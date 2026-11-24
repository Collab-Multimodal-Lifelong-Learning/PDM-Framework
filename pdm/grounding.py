"""Evaluate saved answers and temporal evidence on the same held-out questions."""
import csv
import json
from collections import defaultdict
from statistics import fmean

from .data import _atomic_path, read_manifest_rows
from .download import _get, _publish, _valid
from .experiment import checksum

REVISION = "63772e0256ad9e34b83d6a108f1ba2041b924607"
BASE_URL = f"https://raw.githubusercontent.com/doc-doc/NExT-GQA/{REVISION}/datasets/nextgqa"
HASHES = {
    "test.csv": "0443ede53f521de4cbfb5e4fb02253926b90f81f62041b18bd17207a3511d114",
    "gsub_test.json": "0c16c31699238a0f2b8ace45aeed7f193149ce2f011d6ed4f1ef7170ef5d41e9",
}


def nextgqa_references(directory, tests):
    for name, digest in HASHES.items():
        path = directory / name
        if not path.exists():
            body, _ = _get(f"{BASE_URL}/{name}", 2 * 1024 * 1024)
            _publish(path, body, digest)
        elif not _valid(path, digest):
            raise ValueError(f"unexpected NExT-GQA checksum for {path}")
    with (directory / "test.csv").open(encoding="utf-8", newline="") as f:
        annotations = {f"{r['video_id']}:{r['qid']}:{r['type']}": r for r in csv.DictReader(f)}
    intervals = json.loads((directory / "gsub_test.json").read_text())
    result = {}
    for record in tests:
        sid = record["sample_id"]
        if sid not in annotations:
            continue
        row = annotations[sid]
        choices = [row[f"a{i}"] for i in range(5)]
        if (record["clip_id"] != row["video_id"] or row["question"] != record["question"]
                or choices != record["choices"] or choices[record["answer_index"]] != row["answer"]):
            raise ValueError(f"question, choices or answer differ from NExT-GQA for {sid}")
        spans = intervals[row["video_id"]]["location"][row["qid"]]
        result[sid] = [[round(start * 1000), round(end * 1000)] for start, end in spans]
    return result


def measure(row, references):
    start, end = row["evidence_ms"]
    if not 0 <= start < end or not references:
        raise ValueError(f"invalid temporal evidence for {row['sample_id']}")
    iop = iou = 0.0
    for a, b in references:
        if not 0 <= a <= b:
            raise ValueError(f"invalid reference interval for {row['sample_id']}")
        overlap = max(0, min(end, b) - max(start, a))
        iop = max(iop, overlap / (end - start))
        iou = max(iou, overlap / (max(end, b) - min(start, a)))
    correct = row["choice_index"] == row["answer_index"]
    return dict(correct=correct, grounded_correct=correct and iop >= 0.5, iop=iop, iou=iou)


def summarize(samples):
    return {"count": len(samples), "answer_accuracy": fmean(s["correct"] for s in samples),
            "acc_gqa": fmean(s["grounded_correct"] for s in samples),
            "mean_iop": fmean(s["iop"] for s in samples), "mean_iou": fmean(s["iou"] for s in samples)}


def score_grounding(records, predictions, references, conditions):
    tests = {r["sample_id"]: r for r in records if r["split"] == "test"}
    stages = sorted({r["stage"] for r in tests.values()})
    if not tests.keys() & references.keys():
        raise ValueError("no held-out questions have matching temporal annotations")
    groups = defaultdict(dict)
    for row in predictions:
        sid, method, stage = row["sample_id"], row["condition"], row["stage"]
        record = tests.get(sid)
        if (method not in conditions or stage not in stages or record is None
                or row["task_stage"] != record["stage"] or record["stage"] > stage
                or row["answer_index"] != record["answer_index"] or row["choice_index"] not in range(5)):
            raise ValueError(f"inconsistent held-out prediction for {sid}")
        group = groups[method, stage]
        if sid in group:
            raise ValueError(f"duplicate prediction for {sid}, {method}, stage {stage}")
        group[sid] = row

    summaries, samples = {}, []
    for method in conditions:
        history = []
        for stage in stages:
            group = groups[method, stage]
            expected = {sid for sid, r in tests.items() if r["stage"] <= stage}
            if expected != group.keys():
                raise ValueError(f"missing predictions for {method}, stage {stage}: {sorted(expected - group.keys())}")
            by_task = defaultdict(list)
            matched = []
            for sid in sorted(expected & references.keys()):
                row = group[sid]
                score = measure(row, references[sid])
                by_task[row["task_stage"]].append(score)
                matched.append(score)
                samples.append(dict(score, sample_id=sid, condition=method, stage=stage,
                                    task_stage=row["task_stage"], prediction_ms=row["evidence_ms"],
                                    reference_ms=references[sid]))
            task_scores = {t: summarize(items) for t, items in by_task.items()}
            old = [t for t in task_scores if t < stage]
            forgetting = {}
            for metric in ("answer_accuracy", "acc_gqa", "mean_iop", "mean_iou"):
                drops = []
                for t in old:
                    values = [h["by_task"][t][metric] for h in history if t in h["by_task"]]
                    now = task_scores[t][metric]
                    drops.append(max([now, *values]) - now)
                forgetting[metric] = fmean(drops) if drops else None
            history.append(dict(stage=stage, evaluated_test_count=len(expected),
                                overall=summarize(matched) if matched else None,
                                by_task=task_scores, previous_task_forgetting=forgetting))
        summaries[method] = history
    return {"unannotated_test_ids": sorted(tests.keys() - references.keys()),
            "conditions": summaries, "samples": samples}


def evaluate_saved(out, annotations):
    manifest, predictions = out / "manifest.jsonl", out / "predictions.jsonl"
    info = json.loads((out / "run.json").read_text())
    if info["status"] != "complete" or checksum(manifest) != info["manifest_sha256"]:
        raise ValueError("run is incomplete or its saved manifest has changed")
    records = [r for r in read_manifest_rows(manifest) if r["split"] == "test"]
    if info["profile"] == "smoke":
        refs = {r["sample_id"]: [r["evidence_ms"]] for r in records if "evidence_ms" in r}
        source = {"source": "smoke_fixture"}
    else:
        refs = nextgqa_references(annotations, records)
        source = {"source": "NExT-GQA test", "revision": REVISION, "annotation_sha256": HASHES}
    with predictions.open() as f:
        result = score_grounding(records, map(json.loads, f), refs, info["conditions"])
    result.update(source, manifest_sha256=info["manifest_sha256"], predictions_sha256=checksum(predictions))
    path = out / "grounding.json"
    with _atomic_path(path) as tmp:
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for method, history in result["conditions"].items():
        final = history[-1]["overall"]
        print(f"{method}: Acc@GQA {final['acc_gqa']:.1%}, mean IoP {final['mean_iop']:.1%} "
              f"({final['count']}/{history[-1]['evaluated_test_count']} annotated)")
    print(path)
