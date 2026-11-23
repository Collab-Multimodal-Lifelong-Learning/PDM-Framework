"""Portable VideoQA manifests, local NExT-QA import, and generated CPU media."""

import csv
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import av
import numpy as np


NEXTQA_DOMAINS = ("temporal", "causal")
DOMAINS = {"smoke": ("first_color", "last_color"), "nextqa": NEXTQA_DOMAINS, "nextqa_quick": NEXTQA_DOMAINS}
TRAIN_MINIMUM = {"smoke": 32, "nextqa": 64, "nextqa_quick": 4}
NEXTQA_HASHES = {
    "train": "51ae6e611fa58835f63a0beb325fe647bb197c7e320a31f0bb7a9cf969e082b2",
    "val": "43198bdef8436b8d64a9b75d846b0987c10cbf94ebf4be325c4a4e54634d66b8",
    "test": "cdf46de237b3b3b846ff0209ae1394ad559791b5a0de0bb8c78648f74da83542",
}
COLORS = ("red", "green", "blue", "yellow", "magenta")
RGB = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)], dtype=np.uint8)
QUESTIONS = ("Which color appeared first?", "Which color appeared last?")


@contextmanager
def _atomic_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
        temp = Path(f.name)
    try:
        yield temp
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def inference_input(record):
    """Copy only fields visible to the scorer, including the prediction join key."""
    item = {key: record[key] for key in ("sample_id", "question", "clip_path")}
    item["choices"] = list(record["choices"])
    if "sensor_path" in record:
        item["sensor_path"] = record["sensor_path"]
    return item


def save_manifest(path: Path, records):
    path = Path(path).resolve()
    with _atomic_path(path) as temp, temp.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            row = dict(record)
            for key in ("clip_path", "sensor_path"):
                if key in row:
                    row[key] = os.path.relpath(row[key], path.parent)
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def read_manifest_rows(path: Path) -> list[dict]:
    """Parse JSON Lines and rebase paths without opening the referenced media."""
    path = Path(path).resolve()
    records = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected a JSON object")
                for key in ("clip_path", "sensor_path"):
                    if key == "sensor_path" and key not in record:
                        continue
                    value = record.get(key)
                    if not _text(value) or Path(value).is_absolute():
                        raise ValueError(f"{key} must be a nonempty relative path")
                    record[key] = str((path.parent / value).resolve())
                records.append(record)
            except ValueError as exc:
                raise ValueError(f"{path}, line {line_number}: {exc}") from exc
    return records


def load_manifest(path: Path, profile: str) -> list[dict]:
    return _validate(read_manifest_rows(path), profile)


def _check_fields(record, domains):
    for key in ("sample_id", "clip_id", "clip_path", "question"):
        if not _text(record.get(key)):
            raise ValueError(f"{key} must be a nonempty string")
    if record.get("split") not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    stage = record.get("stage")
    if type(stage) is not int or stage not in (1, 2):
        raise ValueError("stage must be integer 1 or 2")
    if record.get("domain") != domains[stage - 1]:
        raise ValueError(f"domain for stage {stage} must be {domains[stage - 1]}")
    choices = record.get("choices")
    if not isinstance(choices, list) or len(choices) != 5 or not all(map(_text, choices)):
        raise ValueError("choices must be five nonempty strings")
    answer = record.get("answer_index")
    if type(answer) is not int or not 0 <= answer < 5:
        raise ValueError("answer_index must be an integer from 0 to 4")


def _validate(records, profile):
    from .models import video_duration_ms

    if profile not in DOMAINS:
        raise ValueError(f"Unknown profile {profile!r}")
    ids = set()
    clip_paths = {}
    path_splits = {}
    counts = Counter()
    for record in records:
        try:
            _check_fields(record, DOMAINS[profile])
            sid, clip, path, split = (record[k] for k in ("sample_id", "clip_id", "clip_path", "split"))
            if sid in ids:
                raise ValueError(f"duplicate sample_id {sid!r}")
            if clip_paths.setdefault(clip, path) != path:
                raise ValueError(f"clip_id {clip!r} has inconsistent clip_path")
            if path_splits.setdefault(path, split) != split:
                raise ValueError(f"clip {clip!r} occurs in multiple splits")
            ids.add(sid)
            counts[record["stage"], split] += 1
        except ValueError as exc:
            raise ValueError(f"record {record.get('sample_id')!r}: {exc}") from exc
    for stage in (1, 2):
        for split in ("train", "val", "test"):
            required = TRAIN_MINIMUM[profile] if split == "train" else 1
            if counts[stage, split] < required:
                raise ValueError(f"stage {stage} {split} requires at least {required} distinct records")

    durations = {}
    for path in path_splits:
        try:
            durations[path] = video_duration_ms(path)
        except (OSError, ValueError, RuntimeError, av.error.FFmpegError) as exc:
            raise ValueError(f"Cannot read clip {path}: {exc}") from exc
    for record in records:
        duration = durations[record["clip_path"]]
        if "evidence_ms" in record:
            evidence = record["evidence_ms"]
            if not (
                isinstance(evidence, list) and len(evidence) == 2
                and all(type(t) is int for t in evidence)
                and 0 <= evidence[0] < evidence[1] <= duration
            ):
                raise ValueError(f"{record['sample_id']}: invalid evidence_ms for {duration} ms clip")
        if "sensor_path" in record:
            read_sensor(record["sensor_path"], duration)
    return records


def read_sensor(path, duration_ms):
    """Read finite scalar measurements ordered on the clip's millisecond clock."""
    values = []
    try:
        with Path(path).open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, strict=True)
            if not reader.fieldnames or not {"timestamp_ms", "value"} <= set(reader.fieldnames):
                raise ValueError("expected timestamp_ms,value columns")
            for row in reader:
                timestamp, value = int(row["timestamp_ms"]), float(row["value"])
                if not 0 <= timestamp <= duration_ms or (values and timestamp <= values[-1][0]):
                    raise ValueError(f"unordered or out-of-range timestamp at row {reader.line_num}")
                if not math.isfinite(value):
                    raise ValueError(f"non-finite value at row {reader.line_num}")
                values.append((timestamp, value))
            if not values:
                raise ValueError("empty sensor trace")
    except (OSError, TypeError, ValueError, csv.Error) as exc:
        raise ValueError(f"Cannot read sensor {path}: {exc}") from exc
    return values


def _write_clip(path, frames):
    with _atomic_path(path) as temp, av.open(str(temp), "w", format="mp4") as video:
        stream = video.add_stream("libx264", rate=4)
        stream.width, stream.height = 64, 64
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "qp": "0", "threads": "1"}
        for pixels in frames:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                video.mux(packet)
        for packet in stream.encode():
            video.mux(packet)


def _color_frames(first, last, rng):
    # A seeded, low-contrast texture distinguishes clips while preserving their colors.
    texture = rng.integers(0, 36, size=(16, 16), dtype=np.uint8)
    texture = texture.repeat(4, axis=0).repeat(4, axis=1)
    for frame_number in range(16):
        color = RGB[first if frame_number < 8 else last]
        intensity = 255 - np.roll(texture, frame_number, axis=1)
        yield np.minimum(color, intensity[:, :, None])


def generate_fixture(out: Path) -> list[dict]:
    out = Path(out).resolve()
    rng = np.random.default_rng(42)
    records = []
    for stage, domain in enumerate(DOMAINS["smoke"], start=1):
        for split, size in (("train", 40), ("val", 10), ("test", 10)):
            answers = rng.permutation(np.tile(np.arange(5), size // 5))
            for number, label in enumerate(answers):
                answer = int(label)
                other = (answer + int(rng.integers(1, 5))) % 5
                first, last = (answer, other) if stage == 1 else (other, answer)
                clip_id = f"s{stage}-{split}-{number:03d}"
                path = out / "clips" / f"{clip_id}.mp4"
                _write_clip(path, _color_frames(first, last, rng))
                records.append({
                    "sample_id": f"{clip_id}:q01", "clip_id": clip_id, "clip_path": str(path),
                    "split": split, "stage": stage, "domain": domain, "question": QUESTIONS[stage - 1],
                    "choices": list(COLORS), "answer_index": answer,
                    "evidence_ms": [0, 2000] if stage == 1 else [2000, 4000],
                })
    return _validate(records, "smoke")


def sensor_fixture(out: Path) -> dict:
    directory = Path(out).resolve() / "sensor"
    video = directory / "constant.mp4"
    frame = np.full((64, 64, 3), RGB[0], dtype=np.uint8)
    _write_clip(video, (frame for _ in range(16)))
    trace = directory / "trace.csv"
    with _atomic_path(trace) as temp:
        temp.write_text("timestamp_ms,value\n" + "".join(
            f"{t},{int(t == 2500)}\n" for t in range(0, 4000, 250)
        ), encoding="utf-8")
    return {"sample_id": "sensor-pair", "clip_path": str(video), "question": QUESTIONS[0],
            "choices": list(COLORS), "sensor_path": str(trace)}


def select_nextqa(annotations: Path, videos: Path, *, quick=False) -> list[dict]:
    """Select rows from the pinned CSVs before fetching or validating their media."""
    annotations, videos = Path(annotations), Path(videos).resolve()
    sources = {}
    for split, expected in NEXTQA_HASHES.items():
        path = annotations / f"{split}.csv"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {path}; expected {expected}")
        sources[split] = raw.decode("utf-8-sig")

    partitions = defaultdict(list)
    for split, source in sources.items():
        for row in csv.DictReader(io.StringIO(source, newline="")):
            kind = row["type"]
            if not kind.startswith(("T", "C")):
                continue
            stage = 1 if kind.startswith("T") else 2
            clip_id = row["video"]
            sample_id = f"{clip_id}:{row['qid']}:{kind}"
            partitions[stage, split].append({
                "sample_id": sample_id, "clip_id": clip_id, "clip_path": str((videos / f"{clip_id}.mp4").resolve()),
                "split": split, "stage": stage, "domain": DOMAINS["nextqa"][stage - 1],
                "question": row["question"], "choices": [row[f"a{i}"] for i in range(5)],
                "answer_index": int(row["answer"]),
            })
    caps = {"train": 8, "val": 2, "test": 3} if quick else {"train": 128, "val": 64, "test": 128}
    records = []
    for stage in (1, 2):
        for split, cap in caps.items():
            candidates = partitions[stage, split]
            candidates.sort(key=lambda row: hashlib.sha256(f"42:{row['sample_id']}".encode()).digest())
            records.extend(candidates[:cap])
    return records


def prepare_nextqa(annotations: Path, videos: Path, out: Path, *, quick=False) -> list[dict]:
    records = select_nextqa(annotations, videos, quick=quick)
    _validate(records, "nextqa_quick" if quick else "nextqa")
    save_manifest(Path(out), records)
    return records
