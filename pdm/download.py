"""Fetch and verify only the NExT-QA members selected for the quick experiment."""

import hashlib
import io
import re
import time
import zipfile
import zlib
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import data


ANNOTATION_URL = (
    "https://raw.githubusercontent.com/doc-doc/NExT-QA/"
    "2432e9724f88ed9f40010e2989f104570a91de4e/dataset/nextqa"
)
ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH&export=download&confirm=t"
)
# SHA-256 of the 26 selected MP4s after checking the official ZIP member CRCs.
# These pins also verify existing local copies when the machine is offline.
VIDEO_HASHES = {
    "10931176255.mp4": "1d6e8d8351b8d8dc982564172690fea10a76ebb88e589b7c1f30d2f0761e4139",
    "11866348213.mp4": "c3e2ad26a04af3caae3143fc69855532efc86df35e8b2b80f3d8cf5e94e4a497",
    "2457454663.mp4": "61d9dd615d986330812eba4797507b7500d3273f272574fa77c557ce24790137",
    "2530259622.mp4": "7f96998ac8994039d52b0c67a5901aba83df3be967ceed0212792d398e77c762",
    "2664151010.mp4": "09d7125325a88674f22d69b0bda5cab888c57146c58c46de53f743b96e82768c",
    "2867108011.mp4": "59326bb541fcb80a97030cf064bfc1368eb4e0ce6a9bfc513f8592919b0241cd",
    "2984974097.mp4": "ce82cb8da378cfe425204e72c2d9124a9bc4d8d380c2f02d1f9feb4491d68933",
    "3396832512.mp4": "688a8d8bead6388b17c83fc2b75cafef67b2397e4c67c89150518703763df143",
    "3425596660.mp4": "2e1485df7845456e7d01e35f86e5c5405504e67383aed9dca6d05ec662528148",
    "3459862895.mp4": "314916405b1d9457adc02b71bcbc42d8bf497c36444f6c8294258f206f94e3a2",
    "3606510307.mp4": "b5ad9376c2e277541f32913231ff5f2c2138cda136c50f2ea21a0421f6a00638",
    "3763693893.mp4": "6067529d0397bd0a55edfc1718e7c9a136bd6b7cc1d08b194fd6a2efb9b12db8",
    "3799619778.mp4": "d3ba257fd10405981e056390c706ee6401a6386fa26b3525e838f94a1ce10ef6",
    "4166942239.mp4": "3db5b81b633530b7abb5837468f43b917a8d278ccf31f62cbfcf05c7f1a3c766",
    "4789310455.mp4": "d9dc49200a1a9ec1283848be5c54a935e512a62cca6f6be95f849330a6fa3f03",
    "4890651848.mp4": "82bae4e72dee8093c2b309d348858bc77c32d18dbefd0ca283d9ca3a3254a736",
    "4992338194.mp4": "2258ff1927e936a54a13ad1151e36a398385a999374387ec5623f819ccdd22f3",
    "5061117640.mp4": "40f8620cad6012024447e5e29bcd7cc13c89522ad6acc951b2b44a462551b926",
    "5516830700.mp4": "574103ec4258cf8a49b8ba461cdac07cb31f9d52fb262b218491524c6effeb41",
    "5571958942.mp4": "fbc2701edd2a7c51980dab5df4d7f766621b8f8a376d0fad4450a447b31f44aa",
    "5863411262.mp4": "b04db53c50f93a6d50cf2f47919c820de9e9b20b884ef70b1be5e87c7929d853",
    "5866947362.mp4": "0953ca9c3b253ed565ea48b2cb4a135d8d738b01b9ce595ca63f8cd197493aa3",
    "5976071606.mp4": "141847b47dd91ba32a3ca71621a1b0a0a0633efc10bca3edd65d1c0af7bb2a9c",
    "6299469194.mp4": "0277809c1ad9eb326201b0d1f4445d895167bbaf6e12696620a521d9a6f3a222",
    "8015291804.mp4": "2734c6f692e54d470f6076834fe7f5fa9ab943d94059bed8623719b99d566203",
    "9231896269.mp4": "40c99a6d5bc12e14a38a53dedfa122a922cb24c34c651f22df6f0dadf0cb33eb",
}

# The largest pinned CSV is <6 MiB. The largest selected MP4 is <10 MiB.
MAX_BYTES = 32 * 1024 * 1024


def _get(url, limit, byte_range=None):
    headers = {"Accept-Encoding": "identity"}
    if byte_range is not None:
        first, last = byte_range
        headers["Range"] = f"bytes={first}-{last}"
    request = Request(url, headers=headers)
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                total = None
                if byte_range is not None:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if response.status != 206 or not match:
                        raise ValueError(f"Archive server must support HTTP 206 byte ranges: {url}")
                    actual_first, actual_last, total = map(int, match.groups())
                    if (actual_first, actual_last) != byte_range or total <= actual_last:
                        raise ValueError(f"Unexpected Content-Range {content_range!r} for {headers['Range']}")
                body = response.read(limit + 1)
                if len(body) > limit:
                    raise ValueError(f"Response from {url} exceeded {limit} bytes")
                if byte_range is not None and len(body) != last - first + 1:
                    raise ValueError(f"Truncated range {headers['Range']} from {url}")
                return body, total
        except HTTPError as exc:
            if attempt == 2 or exc.code not in (429, 500, 502, 503, 504):
                raise
        except (URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
            if attempt == 2:
                raise OSError(f"Download failed for {url}: {exc}") from exc
        time.sleep(2 ** attempt)


class RangeReader(io.RawIOBase):
    """Read a remote ZIP directory and selected members through bounded HTTP ranges."""

    def __init__(self, url):
        self.url = url
        self.buffer, self.size = _get(url, 1, (0, 0))
        self.start = self.pos = 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        pos = (0, self.pos, self.size)[whence] + offset
        if pos < 0:
            raise OSError("Seek before start of ZIP archive")
        self.pos = pos
        return pos

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.pos + size)
        if end <= self.pos:
            return b""
        if end - self.pos > MAX_BYTES:
            raise ValueError("ZIP directory or compressed member exceeds the quick download limit")
        if self.pos < self.start or end > self.start + len(self.buffer):
            # Read ahead for the variable-length ZIP local header before member data.
            last = min(self.size, max(self.pos + 8192, end)) - 1
            self.buffer, size_now = _get(self.url, last - self.pos + 1, (self.pos, last))
            if size_now != self.size:
                raise ValueError("Remote ZIP archive changed size during download")
            self.start = self.pos
        offset = self.pos - self.start
        self.pos = end
        return self.buffer[offset:end - self.start]


def _valid(path, expected):
    if not path.is_file():
        return False
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() == expected


def _publish(path, body, expected):
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    with data._atomic_path(path) as tmp:
        tmp.write_bytes(body)


def _fetch_videos(directory, missing):
    try:
        with RangeReader(ARCHIVE_URL) as source, zipfile.ZipFile(source) as archive:
            members = {}
            for info in archive.infolist():
                name = Path(info.filename).name
                if name in missing:
                    if name in members:
                        raise ValueError(f"Ambiguous ZIP member {name}")
                    members[name] = info
            absent = missing.difference(members)
            if absent:
                raise ValueError(f"Missing ZIP members: {', '.join(sorted(absent))}")
            for name, info in members.items():
                if max(info.compress_size, info.file_size) > MAX_BYTES:
                    raise ValueError(f"ZIP member {name} exceeds the quick download limit")
                # ZipExtFile verifies the member CRC after decompression.
                with archive.open(info) as member:
                    body = member.read(MAX_BYTES + 1)
                if len(body) != info.file_size:
                    raise ValueError(f"Invalid size of ZIP member {name}")
                _publish(directory / name, body, VIDEO_HASHES[name])
    except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
        raise ValueError(f"Invalid NExT-QA ZIP archive: {exc}") from exc


def fetch_quick(root: Path) -> Path:
    """Return a portable quick manifest, repairing individual missing or invalid files."""
    root = Path(root).expanduser().resolve()
    annotations, videos = root / "annotations", root / "videos"
    for split, expected in data.NEXTQA_HASHES.items():
        name = f"{split}.csv"
        path = annotations / name
        if not _valid(path, expected):
            body, _ = _get(f"{ANNOTATION_URL}/{name}", MAX_BYTES)
            _publish(path, body, expected)

    records = data.select_nextqa(annotations, videos, quick=True)
    names = {f"{r['clip_id']}.mp4" for r in records}
    if names != VIDEO_HASHES.keys():
        raise ValueError("Selected NExT-QA clips differ from the pinned quick profile")
    missing = {name for name in names if not _valid(videos / name, VIDEO_HASHES[name])}
    if missing:
        _fetch_videos(videos, missing)
    path = root / "quick.jsonl"
    if path.is_file():
        try:
            if data.load_manifest(path, "nextqa_quick") == records:
                return path
        except (OSError, ValueError):
            # A stale/partial manifest can be rebuilt from the verified source files.
            pass
    data.prepare_nextqa(annotations, videos, path, quick=True)
    return path
