import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from pdm import data


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    return data.generate_fixture(tmp_path_factory.mktemp("generated"))


def test_fixture_is_balanced_and_verified_against_pixels(records):
    import av
    import numpy as np

    palette = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]])
    counts = Counter()
    first_frames = set()
    for record in records:
        with av.open(record["clip_path"]) as video:
            assert video.streams.video[0].average_rate == 4
            frames = [frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)]
        assert len(frames) == 16 and frames[0].shape == (64, 64, 3)
        first_frames.add(frames[0].tobytes())
        colors = [frame.mean(axis=(0, 1)) / 255 for frame in (frames[0], frames[-1])]
        nearest = [np.linalg.norm(palette - color, axis=1).argmin() for color in colors]
        stage = record["stage"]
        assert nearest[0] != nearest[1]
        assert nearest[stage - 1] == record["answer_index"]
        assert record["evidence_ms"] == ([0, 2000] if stage == 1 else [2000, 4000])
        counts[stage, record["split"], nearest[stage - 1]] += 1
    assert len(records) == len(first_frames) == 120
    for stage in (1, 2):
        for split, count in (("train", 8), ("val", 2), ("test", 2)):
            assert [counts[stage, split, color] for color in range(5)] == [count] * 5


def test_fixture_reproducible_and_manifest_portable(tmp_path, records):
    second = data.generate_fixture(tmp_path / "second")
    assert [(r["sample_id"], r["answer_index"]) for r in records] == [
        (r["sample_id"], r["answer_index"]) for r in second
    ]
    manifest = tmp_path / "report" / "manifest.jsonl"
    data.save_manifest(manifest, second)
    first = json.loads(manifest.read_text().splitlines()[0])
    assert first["clip_path"].startswith("../second/clips/")
    assert data.load_manifest(manifest, "smoke") == second
    clean = data.inference_input(second[0])
    assert set(clean) == {"sample_id", "clip_path", "question", "choices"}


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"answer_index": 5}, "answer_index"),
        ({"answer_index": True}, "answer_index"),
        ({"domain": "last_color"}, "domain"),
        ({"stage": True}, "stage"),
        ({"evidence_ms": [0, 4001]}, "evidence_ms"),
        ({"evidence_ms": [4, 4]}, "evidence_ms"),
        ({"clip_path": "missing.mp4"}, "missing.mp4"),
        ({"choices": "abcde"}, "choices"),
        ({"split": []}, "split"),
        ({"question": ""}, "question"),
    ],
)
def test_invalid_record_fails_with_field(tmp_path, records, change, expected):
    path = tmp_path / "broken.jsonl"
    data.save_manifest(path, records)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(change)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=expected):
        data.load_manifest(path, "smoke")


def test_manifest_rejects_duplicate_split_leak_and_inconsistent_source(tmp_path, records):
    path = tmp_path / "manifest.jsonl"
    train = records[0]
    val = next(r for r in records if r["split"] == "val")
    for extra, message in [
        (train, "duplicate"),
        (dict(train, sample_id="new", split="val"), "split"),
        (dict(train, sample_id="new", clip_path=val["clip_path"]), "path"),
        (dict(val, sample_id="new", clip_id="alias", split="train"), "split"),
    ]:
        data.save_manifest(path, [*records, extra])
        with pytest.raises(ValueError, match=message):
            data.load_manifest(path, "smoke")
    data.save_manifest(path, [r for r in records if r["split"] != "test"])
    with pytest.raises(ValueError, match="test"):
        data.load_manifest(path, "smoke")


def test_sensor_pair_has_identical_frames_and_one_spike(tmp_path):
    import av
    import numpy as np

    record = data.sensor_fixture(tmp_path)
    assert set(record) == {"sample_id", "clip_path", "question", "choices", "sensor_path"}
    with av.open(record["clip_path"]) as video:
        frames = [frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)]
    assert len(frames) == 16 and all(np.array_equal(f, frames[0]) for f in frames)
    values = data.read_sensor(Path(record["sensor_path"]), 4000)
    assert [(t, v) for t, v in values if v != 0] == [(2500, 1)]


@pytest.mark.parametrize("rows", ["1,nan", "2,1\n1,2", "0,0\n0,1", "4001,0", "2.5,4", ""])
def test_invalid_sensor_trace(tmp_path, rows):
    sensor = tmp_path / "bad.csv"
    sensor.write_text("timestamp_ms,value\n" + rows)
    with pytest.raises(ValueError):
        data.read_sensor(sensor, 4000)


def test_nextqa_pins_checksums_and_selects_seeded_subset(tmp_path, monkeypatch, records):
    from shutil import copyfile

    videos = tmp_path / "videos"
    videos.mkdir()
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    checksums = {}
    for split in ("train", "val", "test"):
        copyfile(next(r["clip_path"] for r in records if r["split"] == split), videos / f"{split}.mp4")
        path = annotations / f"{split}.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video", "qid", "type", "question", "a0", "a1", "a2", "a3", "a4", "answer"])
            for kind in ("TN", "CW", "DC"):
                for qid in range(131):
                    writer.writerow([split, qid, kind, "Why?", "z", "x", "c", "v", "b", 3])
        checksums[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    out = tmp_path / "manifest" / "nextqa.jsonl"
    with pytest.raises(ValueError, match="SHA-256"):
        data.prepare_nextqa(annotations, videos, out)
    assert not out.exists()
    monkeypatch.setattr(data, "NEXTQA_HASHES", checksums)
    data.prepare_nextqa(annotations, videos, out)
    subset = data.load_manifest(out, "nextqa")
    assert len(subset) == 640
    for split, limit in (("train", 128), ("val", 64), ("test", 128)):
        for stage, kind in ((1, "TN"), (2, "CW")):
            ids = [f"{split}:{i}:{kind}" for i in range(131)]
            ids.sort(key=lambda sid: hashlib.sha256(f"42:{sid}".encode()).digest())
            assert [r["sample_id"] for r in subset if r["stage"] == stage and r["split"] == split] == ids[:limit]
    assert all(r["choices"] == ["z", "x", "c", "v", "b"] and r["answer_index"] == 3 for r in subset)
    quick_out = out.with_name("quick.jsonl")
    quick = data.prepare_nextqa(annotations, videos, quick_out, quick=True)
    assert data.load_manifest(quick_out, "nextqa_quick") == quick
    for stage in (1, 2):
        for split, limit in (("train", 8), ("val", 2), ("test", 3)):
            def partition(rows):
                return [r for r in rows if r["stage"] == stage and r["split"] == split]
            assert partition(quick) == partition(subset)[:limit]
    with pytest.raises(ValueError, match="64 distinct"):
        data.load_manifest(quick_out, "nextqa")
