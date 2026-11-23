"""Matched-budget continual updates and bounded training memory."""
import hashlib
import re

import torch
from torch.nn import functional as F

from .data import inference_input

CONDITIONS = ("frozen", "retrieval", "lora", "pdm", "pdm_uniform", "pdm_no_replay")
TRAINED = {"lora", "pdm", "pdm_uniform", "pdm_no_replay"}
REPLAY = {"pdm", "pdm_uniform"}
SELECTED = {"pdm", "pdm_no_replay"}


def seeded_order(records, seed, stage):
    def key(record):
        return hashlib.sha256(f"{seed}:{stage}:{record['sample_id']}".encode()).digest()
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


def loss(model, record):
    logits = model.score(inference_input(record)).float()
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


def update(model, condition, current, memory, config, seed, stage):
    ordered = seeded_order(current, seed, stage)
    info = {"selected_ids": [], "replay_ids": [], "optimizer_steps": 0}
    if condition in TRAINED:
        steps = config["steps"]
        prior = seeded_order(memory, seed, stage)[:steps] if condition in REPLAY else []
        needed = steps * 4 - len(prior)
        if len(ordered) < needed:
            raise ValueError(f"stage {stage} needs {needed} distinct training records")
        chosen = select(model, current, needed) if condition in SELECTED else ordered[:needed]
        info["selected_ids"] = [r["sample_id"] for r in chosen]
        info["replay_ids"] = [r["sample_id"] for r in prior]
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=0)
        model.train()
        offset = 0
        for step in range(steps):
            replay = prior[step:step+1]
            batch = chosen[offset:offset+4-len(replay)]
            offset += len(batch)
            optimizer.zero_grad(set_to_none=True)
            for record in batch + replay:
                (loss(model, record) / 4).backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        info["optimizer_steps"] = steps
    if condition in REPLAY or condition == "retrieval":
        memory = (memory + ordered)[-config["memory_capacity"]:]
    model.eval()
    return memory, info
