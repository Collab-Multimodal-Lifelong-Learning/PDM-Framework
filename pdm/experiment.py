"""Run the same train, checkpoint and held-out evaluation loop on both backends."""
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .continual import SELECTED, TRAINED, retrieve, update
from .data import NEXTQA_HASHES, inference_input, read_manifest_rows, save_manifest
from .models import QWEN_REVISION, VLM_RLAIF_REVISION, build_model, clear_video_cache, predict
from .report import write_report

OASIS_REVISION = "96e67e159397b851c4cc80cae559cef00370a8cf"


def write_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_lines(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def backend(profile):
    return "smoke" if profile == "smoke" else "nextqa"


def trainable_state(model):
    return {name: p.detach().cpu().contiguous() for name, p in model.named_parameters() if p.requires_grad}


def fingerprint(tensors):
    digest = hashlib.sha256()
    for key, tensor in sorted(tensors.items()):
        digest.update(key.encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model, tests, name, stage, memory, device):
    rows, latencies = [], []
    model.eval()
    with torch.inference_mode():
        for record in tests:
            item = inference_input(record)
            sync(device)
            start = time.perf_counter()
            context = retrieve(item, memory) if name == "retrieval" else ()
            answer = predict(model, item, context)
            sync(device)
            latencies.append(1000 * (time.perf_counter() - start))
            rows.append(dict(answer, sample_id=record["sample_id"], condition=name, stage=stage,
                             task_stage=record["stage"], answer_index=record["answer_index"]))
    return rows, latencies


def accuracy(rows):
    return statistics.mean(r["choice_index"] == r["answer_index"] for r in rows)


def evidence_stats(rows, by_id):
    overlaps = []
    for row in rows:
        truth = by_id[row["sample_id"]].get("evidence_ms")
        if truth is not None:
            a, b = row["evidence_ms"]
            c, d = truth
            overlaps.append(max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c)))
    return {"evidence_count": len(overlaps), "mean_evidence_iou": statistics.mean(overlaps) if overlaps else None}


def sampler_comparison(model, tests, baseline):
    changed, hits = [], 0
    with torch.inference_mode():
        for record, ref in zip(tests, baseline, strict=True):
            logits = model.score(inference_input(record), sampler="vlm_rlaif")
            choice = int(logits.argmax())
            hits += choice == record["answer_index"]
            if choice != ref["choice_index"]:
                changed.append(record["sample_id"])
    return {"accuracy_midpoint": accuracy(baseline), "accuracy_vlm_rlaif": hits / len(tests),
            "count": len(tests), "disagreements": len(changed), "disagreement_ids": changed,
            "sample_ids": [r["sample_id"] for r in tests]}


def run_experiment(out, records, config, seed, conditions, device, sensor_item=None):
    device = torch.device(device)
    save_manifest(out / "manifest.jsonl", records)
    by_id = {r["sample_id"]: r for r in records}
    stages = sorted({r["stage"] for r in records})
    lock = Path(__file__).resolve().parent.parent / "uv.lock"
    info = {"status": "running", "profile": config["profile"], "config": config, "seed": seed,
            "device": str(device), "conditions": list(conditions), "command": sys.argv[1:],
            "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "manifest_sha256": checksum(out / "manifest.jsonl"), "lock_sha256": checksum(lock) if lock.exists() else None,
            "model_revision": QWEN_REVISION if backend(config["profile"]) == "nextqa" else None,
            "selector_reference": OASIS_REVISION, "sampler_reference": VLM_RLAIF_REVISION}
    if backend(config["profile"]) == "nextqa":
        info.update(annotations_revision="2432e9724f88ed9f40010e2989f104570a91de4e",
                    expected_annotation_hashes=NEXTQA_HASHES)
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
    write_json(out / "run.json", info)
    # Initialize AdamW's lazy runtime outside the first measured training condition.
    if set(conditions) & TRAINED:
        warmup = torch.zeros(1, device=device, requires_grad=True)
        warmup.grad = torch.zeros_like(warmup)
        torch.optim.AdamW([warmup]).step()
        sync(device)
    predictions, metrics = [], {}
    sensor = sampler = None
    for name in conditions:
        model = build_model(backend(config["profile"]), seed, str(device))
        # Initialize inference kernels with an unlabeled validation input.
        warmup_item = inference_input(next(r for r in records if r["split"] == "val"))
        predict(model.eval(), warmup_item)
        memory, matrix, summaries = [], [], []
        for stage in stages:
            current = [r for r in records if r["stage"] == stage and r["split"] == "train"]
            tests = [r for r in records if r["stage"] <= stage and r["split"] == "test"]
            before = fingerprint(trainable_state(model))
            clear_video_cache()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            sync(device)
            start = time.perf_counter()
            memory, audit = update(model, name, current, memory, config, seed, stage)
            sync(device)
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            weights = trainable_state(model)
            checkpoint = out / "checkpoints" / name / f"stage-{stage}"
            checkpoint.mkdir(parents=True)
            if name in TRAINED:
                save_file(weights, checkpoint / "adapter.safetensors")
            write_lines(checkpoint / "memory.jsonl", ({"sample_id": r["sample_id"]} for r in memory))
            state = dict(audit, condition=name, stage=stage, seed=seed, adapter_before=before,
                         adapter_after=fingerprint(weights), memory_sha256=checksum(checkpoint / "memory.jsonl"),
                         selector="oasis_inspired" if name in SELECTED else None)
            clear_video_cache()
            rows, latencies = evaluate(model, tests, name, stage, memory, device)
            scores = [accuracy([r for r in rows if r["task_stage"] == t]) for t in stages if t <= stage]
            matrix.append(scores + [None] * (len(stages) - len(scores)))
            retained = [max(row[t] for row in matrix if row[t] is not None) - scores[t]
                        for t in range(len(scores) - 1)]
            summaries.append(dict(stage=stage, seen_average=statistics.mean(scores),
                                  forgetting=statistics.mean(retained) if retained else None,
                                  adaptation_gain=None, update_seconds=elapsed, peak_cuda_bytes=peak,
                                  mean_latency_ms=statistics.mean(latencies),
                                  p95_latency_ms=sorted(latencies)[math.ceil(.95*len(latencies))-1],
                                  replay_count=len(audit["replay_ids"]), **evidence_stats(rows, by_id)))
            predictions.extend(rows)
            write_json(checkpoint / "state.json", state)
            print(f"{name} | task {stage} | accuracies " + ", ".join(f"{v:.1%}" for v in scores),
                  file=sys.stderr, flush=True)
            if name == "frozen" and stage == stages[-1] and backend(config["profile"]) == "nextqa":
                sampler = sampler_comparison(model, tests, rows)
            if sensor_item and name == ("pdm" if "pdm" in conditions else conditions[0]) and stage == stages[-1]:
                video_only = {k: v for k, v in sensor_item.items() if k != "sensor_path"}
                context = retrieve(video_only, memory) if name == "retrieval" else ()
                sensor = {"condition": name, "stage": stage, "clip_path": sensor_item["clip_path"],
                          "off": predict(model, video_only, context), "on": predict(model, sensor_item, context)}
        metrics[name] = {"accuracy_matrix": matrix, "stages": summaries}
        del model
    if "frozen" in metrics:
        for entry in metrics.values():
            for i, summary in enumerate(entry["stages"]):
                summary["adaptation_gain"] = entry["accuracy_matrix"][i][i] - metrics["frozen"]["accuracy_matrix"][i][i]
    write_lines(out / "predictions.jsonl", predictions)
    write_json(out / "metrics.json", metrics)
    if sensor:
        write_json(out / "sensor_comparison.json", dict(sensor, clip_path=os.path.relpath(sensor["clip_path"], out)))
    if sampler:
        write_json(out / "sampler_comparison.json", sampler)
    write_report(out, records, predictions, metrics, sensor, sampler)
    write_json(out / "run.json", dict(info, status="complete"))


def restore(out, name, stage):
    info = json.loads((out / "run.json").read_text())
    manifest = out / "manifest.jsonl"
    if info["status"] != "complete" or checksum(manifest) != info["manifest_sha256"]:
        raise ValueError("run is incomplete or its saved manifest has changed")
    if name not in info["conditions"]:
        raise ValueError(f"condition {name!r} was not run")
    checkpoint = out / "checkpoints" / name / f"stage-{stage}"
    state = json.loads((checkpoint / "state.json").read_text())
    if "memory_sha256" in state and checksum(checkpoint / "memory.jsonl") != state["memory_sha256"]:
        raise ValueError("checkpoint memory fingerprint mismatch")
    if backend(info["profile"]) == "nextqa" and info["model_revision"] != QWEN_REVISION:
        raise ValueError(f"this run needs Qwen revision {info['model_revision']}")
    # The checksum authenticates the exact manifest validated at training time.
    # A new prediction needs pixels only for the clip being scored.
    records = read_manifest_rows(manifest)
    by_id = {r["sample_id"]: r for r in records}
    memory = [by_id[json.loads(line)["sample_id"]] for line in (checkpoint / "memory.jsonl").read_text().splitlines()]
    model = build_model(backend(info["profile"]), info["seed"], info["device"])
    if name in TRAINED:
        weights = load_file(checkpoint / "adapter.safetensors")
        params = dict(model.named_parameters())
        if weights.keys() != {k for k, p in params.items() if p.requires_grad}:
            raise ValueError("checkpoint parameters differ from the adapter")
        with torch.no_grad():
            for k, v in weights.items():
                params[k].copy_(v)
    if fingerprint(trainable_state(model)) != state["adapter_after"]:
        raise ValueError("adapter fingerprint mismatch")
    return model.eval(), by_id, memory
