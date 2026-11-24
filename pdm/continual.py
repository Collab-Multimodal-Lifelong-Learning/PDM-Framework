"""Matched-budget continual updates and bounded training memory."""
import hashlib
import math
import re
from collections import defaultdict

import torch
from torch.nn import functional as F

from .data import inference_input
from .models import predict, random_frame_window
from .stream import STREAM_METHODS

DEFAULT_CONDITIONS = ("frozen", "retrieval", "lora", "pdm", "pdm_uniform", "pdm_no_replay")
WINDOW_REPLAY = ("pdm_evidence", "pdm_random_window")
CONDITIONS = (*DEFAULT_CONDITIONS, *WINDOW_REPLAY, *STREAM_METHODS)
TRAINED = {*WINDOW_REPLAY, *STREAM_METHODS, "lora", "pdm", "pdm_uniform", "pdm_no_replay"}
REPLAY = {*WINDOW_REPLAY, "pdm", "pdm_uniform"}
SELECTED = {*WINDOW_REPLAY, "pdm", "pdm_no_replay"}


def seeded_order(records, seed, stage, pass_number=1):
    prefix = f"{seed}:{stage}" if pass_number == 1 else f"{seed}:{stage}:pass:{pass_number}"

    def key(record):
        return hashlib.sha256(f"{prefix}:{record['sample_id']}".encode()).digest()
    return sorted(records, key=key)


def retrieve(item, memory):
    def tokens(text):
        return set(re.findall(r"\w+", text.casefold()))

    query = tokens(item["question"])

    def key(record):
        terms = tokens(record["question"])
        similarity = len(query & terms) / max(1, len(query | terms))
        return -similarity, record["sample_id"]
    return sorted(memory, key=key)[:3]


def gradient_order(gradients, count):
    """Greedy gradient energy minus redundancy; rows already in sample-ID order."""
    energy = gradients.square().sum(dim=1)
    directions = F.normalize(gradients, dim=1)
    scores = energy.clone()
    order = []
    for _ in range(count):
        index = int(scores.argmax())
        order.append(index)
        scores -= (directions @ directions[index]) * energy[index]
        scores[index] = -torch.inf
    return order


def loss(model, record, interval=None):
    item = inference_input(record)
    logits = (model.score(item) if interval is None else model.score(item, interval=interval)).float()
    target = torch.tensor([record["answer_index"]], device=logits.device)
    return F.cross_entropy(logits[None], target)


def select(model, candidates, count):
    model.eval()
    records = sorted(candidates, key=lambda r: r["sample_id"])
    parameters = tuple(model.selector_parameters())
    gradients = []
    for record in records:
        tensors = torch.autograd.grad(loss(model, record), parameters)
        gradients.append(torch.cat([g.detach().flatten().float().cpu() for g in tensors]))
    indices = gradient_order(torch.stack(gradients), count)
    return [records[i] for i in indices]


def train_pass(model, chosen, prior, optimizer, steps, *, use_windows):
    model.train()
    offset = 0
    losses = []
    for step in range(steps):
        replay = prior[step:step + 1]
        current_count = 4 - len(replay)
        batch = [(record, None) for record in chosen[offset:offset + current_count]]
        batch.extend((record, record["replay_window_ms"] if use_windows else None) for record in replay)
        offset += current_count
        optimizer.zero_grad(set_to_none=True)
        for record, interval in batch:
            error = loss(model, record, interval)
            (error / 4).backward()
            losses.append(error.detach())
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(torch.stack(losses).mean())


@torch.inference_mode()
def validation_loss_by_task(model, records):
    model.eval()
    losses = defaultdict(list)
    for record in records:
        losses[record["stage"]].append(loss(model, record))
    return {task: float(torch.stack(values).mean()) for task, values in sorted(losses.items())}


def cache_replay_window(model, record, condition, seed, stage):
    evidence = predict(model, inference_input(record))["evidence_ms"]
    if condition == "pdm_random_window":
        key = f"{seed}:{stage}:{record['sample_id']}:random-window".encode()
        draw = int.from_bytes(hashlib.sha256(key).digest(), "big")
        evidence = random_frame_window(record["clip_path"], evidence[1] - evidence[0], draw)
    return dict(record, replay_window_ms=evidence, source_stage=stage)


def update(model, condition, current, memory, config, seed, stage, *, validation, has_next_task=True):
    info = {"selected_ids": [], "replay_ids": [], "optimizer_steps": 0,
            "training_history": [], "best_pass": None, "window_extraction_count": 0}
    if condition in TRAINED:
        if not validation:
            raise ValueError(f"stage {stage} needs validation records to select an adapter")
        steps = config["steps"]
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=0)
        best_loss = math.inf
        for number in range(1, config.get("passes", 1) + 1):
            prior = seeded_order(memory, seed, stage, number)[:steps] if condition in REPLAY else []
            needed = steps * 4 - len(prior)
            if len(current) < needed:
                raise ValueError(f"stage {stage} needs {needed} distinct training records per pass")
            chosen = (select(model, current, needed) if condition in SELECTED
                      else seeded_order(current, seed, stage, number)[:needed])
            train_loss = train_pass(model, chosen, prior, optimizer, steps, use_windows=condition in WINDOW_REPLAY)
            losses = validation_loss_by_task(model, validation)
            val_loss = math.fsum(losses.values()) / len(losses)
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise ValueError(f"non-finite loss for {condition}, stage {stage}, pass {number}")
            info["optimizer_steps"] += steps
            info["selected_ids"].extend(r["sample_id"] for r in chosen)
            info["replay_ids"].extend(r["sample_id"] for r in prior)
            info["training_history"].append({"pass": number, "optimizer_steps": info["optimizer_steps"],
                                             "train_loss": train_loss, "validation_loss": val_loss,
                                             "validation_loss_by_task": losses})
            if val_loss < best_loss:
                best_loss = val_loss
                info["best_pass"] = number
                best_weights = [p.detach().to("cpu", copy=True) for p in params]
        with torch.no_grad():
            for param, weight in zip(params, best_weights, strict=True):
                param.copy_(weight)
    memory, extractions = remember(model, condition, current, memory, config, seed, stage, has_next_task)
    info["window_extraction_count"] = extractions
    return memory, info


def remember(model, condition, current, memory, config, seed, stage, has_next_task):
    model.eval()
    if condition not in REPLAY and condition != "retrieval":
        return memory, 0
    capacity = config["memory_capacity"]
    incoming = seeded_order(current, seed, stage)[-capacity:]
    extractions = 0
    if has_next_task and condition in WINDOW_REPLAY:
        incoming = [cache_replay_window(model, r, condition, seed, stage) for r in incoming]
        extractions = len(incoming)
    return (memory + incoming)[-capacity:], extractions
