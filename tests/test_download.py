"""Exercise actual HTTP requests, ZIP member extraction, and an offline rerun."""
import csv
import hashlib
import io
import re
import struct
import threading
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pdm import data, download


@contextmanager
def serve(files, *, mode="range"):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = files[self.path]
            request_range = self.headers.get("Range")
            requests.append((self.path, request_range))
            if request_range is None or mode == "ignore":
                self.send_response(200)
            else:
                first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request_range).groups())
                assert 0 <= first <= last < len(body)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {first + (mode == 'wrong')}-{last}/{len(body)}")
                body = body[first:last + 1]
                if mode == "short":
                    body = body[:-1]
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def make_sources(tmp_path):
    files, videos, csv_hashes, video_hashes = {}, {}, {}, {}
    records = data.generate_fixture(tmp_path / "fixture")
    for split, n in (("train", 8), ("val", 2), ("test", 3)):
        src = next(row["clip_path"] for row in records if row["split"] == split)
        name = f"{split}.mp4"
        videos[name] = Path(src).read_bytes()
        video_hashes[name] = hashlib.sha256(videos[name]).hexdigest()
        annotations = io.StringIO(newline="")
        writer = csv.writer(annotations)
        writer.writerow(["video", "qid", "type", "question", "a0", "a1", "a2", "a3", "a4", "answer"])
        for kind in ("TN", "CW"):
            for qid in range(n):
                writer.writerow([split, qid, kind, "When?", "A", "B", "C", "D", "E", qid % 5])
        raw = annotations.getvalue().encode()
        files[f"/annotations/{split}.csv"] = raw
        csv_hashes[split] = hashlib.sha256(raw).hexdigest()

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("unused.bin", b"x" * 1_000_000)
        for name, body in videos.items():
            writer.writestr(f"NExT-QA/videos/{name}", body, compress_type=zipfile.ZIP_DEFLATED)
    files["/archive"] = archive.getvalue()
    return files, csv_hashes, video_hashes


def test_selected_download_then_offline_reuse_and_repair(tmp_path, monkeypatch):
    files, csv_hashes, video_hashes = make_sources(tmp_path)
    monkeypatch.setattr(data, "NEXTQA_HASHES", csv_hashes)
    monkeypatch.setattr(download, "VIDEO_HASHES", video_hashes)
    cache = tmp_path / "cache"
    with serve(files) as (url, requests):
        monkeypatch.setattr(download, "ANNOTATION_URL", url + "/annotations")
        monkeypatch.setattr(download, "ARCHIVE_URL", url + "/archive")
        manifest = download.fetch_quick(cache)
        assert len(data.load_manifest(manifest, "nextqa_quick")) == 26
        ranges = [r for path, r in requests if path == "/archive"]
        assert ranges and all(r is not None for r in ranges)
        for r in ranges:
            start, end = map(int, r.split("=")[1].split("-"))
            assert not start < 500_000 < end
        requests.clear()
        # Wrong but decodable video must be replaced.
        (cache / "videos/train.mp4").write_bytes((cache / "videos/val.mp4").read_bytes())
        download.fetch_quick(cache)
        assert requests and all(path == "/archive" for path, _ in requests)
        assert hashlib.sha256((cache / "videos/train.mp4").read_bytes()).hexdigest() == video_hashes["train.mp4"]

    saved = manifest.stat()
    assert download.fetch_quick(cache) == manifest
    assert manifest.stat().st_ino == saved.st_ino
    manifest.write_text("truncated")
    assert download.fetch_quick(cache) == manifest
    assert len(data.load_manifest(manifest, "nextqa_quick")) == 26


@pytest.mark.parametrize("mode", ["ignore", "wrong", "short"])
def test_archive_requires_exact_partial_content(mode):
    with serve({"/archive": b"123456"}, mode=mode) as (url, _):
        with pytest.raises((ValueError, OSError), match="range|Range|206"):
            download.RangeReader(url + "/archive")


def test_bad_annotation_is_not_published(tmp_path, monkeypatch):
    with serve({"/train.csv": b"wrong"}) as (url, _):
        monkeypatch.setattr(download, "ANNOTATION_URL", url)
        with pytest.raises(ValueError, match="SHA-256"):
            download.fetch_quick(tmp_path)
    assert not (tmp_path / "annotations" / "train.csv").exists()
    assert not (tmp_path / "quick.jsonl").exists()


@pytest.mark.parametrize("damage", ["crc", "compressed"])
def test_bad_zip_preserves_previous_cache_file(tmp_path, monkeypatch, damage):
    files, csv_hashes, video_hashes = make_sources(tmp_path)
    bad_zip = bytearray(files["/archive"])
    if damage == "crc":
        # Damage the first selected member's CRC in the central directory.
        signature = b"PK\x01\x02"
        central = bad_zip.index(signature, bad_zip.index(signature) + 4)
        bad_zip[central + 16] ^= 1
    else:
        with zipfile.ZipFile(io.BytesIO(bad_zip)) as archive:
            offset = archive.getinfo("NExT-QA/videos/train.mp4").header_offset
        name_size, extra_size = struct.unpack_from("<HH", bad_zip, offset + 26)
        bad_zip[offset + 30 + name_size + extra_size] = 255  # invalid DEFLATE block
    files["/archive"] = bytes(bad_zip)
    monkeypatch.setattr(data, "NEXTQA_HASHES", csv_hashes)
    monkeypatch.setattr(download, "VIDEO_HASHES", video_hashes)
    old = tmp_path / "cache/videos/train.mp4"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    with serve(files) as (url, _):
        monkeypatch.setattr(download, "ANNOTATION_URL", url + "/annotations")
        monkeypatch.setattr(download, "ARCHIVE_URL", url + "/archive")
        with pytest.raises(ValueError, match="Invalid NExT-QA ZIP"):
            download.fetch_quick(tmp_path / "cache")
    assert old.read_bytes() == b"old"
    assert not (tmp_path / "cache/quick.jsonl").exists()
