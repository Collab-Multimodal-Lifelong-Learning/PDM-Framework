"""One-pass continual VideoQA and question/answer-conditioned clip retrieval."""
import math
import random
from statistics import fmean

import torch
from torch.nn import functional as F

from .data import inference_input

STREAM_METHODS = ("stream_lora", "pdm_stream")


def match_score(logits, answer):
    """Log probability of the supplied answer among the query's five choices."""
    return logits.float().log_softmax(dim=-1)[answer]


def reservoir_slot(rng, seen, size, capacity):
    """Return the insertion/replacement index for the latest distinct arrival."""
    slot = size if size < capacity else rng.randrange(seen)
    return slot if slot < capacity else None


class OnlineLearner:
    def __init__(self, model, method, lr, capacity, seed):
        self.model = model
        self.with_memory = method == "pdm_stream"
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.memory = []
        self.seen = 0
        self.optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                           lr=lr, weight_decay=0)

    def observe(self, item, answer):
        """Update on one labeled arrival. Memory holds only previous training arrivals."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        replay = self.rng.choice(self.memory) if self.memory else None
        audit = dict(sample_id=item["sample_id"], replay_id=None, ranking_loss=None)
        if replay is not None:
            old = self.model.score(inference_input(replay)).float()
            old_ce = -match_score(old, replay["answer_index"])
            distill = F.mse_loss(old, old.new_tensor(replay["teacher_logits"]))
            (0.5 * old_ce + 0.5 * distill).backward()
            audit.update(replay_id=replay["sample_id"], replay_ce=old_ce.item(), distill_mse=distill.item())

        logits = self.model.score(item).float()
        pos = match_score(logits, answer)
        objective = -pos
        audit["qa_ce"] = objective.item()
        # Skip known duplicate positives. Other clips are weak negatives.
        query = item["question"], item["choices"][answer]
        eligible = [
            r for r in self.memory
            if r["clip_path"] != item["clip_path"] and (r["question"], r["choices"][r["answer_index"]]) != query
        ]
        if eligible:
            negative = self.rng.choice(eligible)
            other = self.model.score(dict(item, clip_path=negative["clip_path"])).float()
            ranking = F.softplus(match_score(other, answer) - pos)
            objective = objective + 0.5 * ranking
            audit.update(ranking_loss=ranking.item(), negative_id=negative["sample_id"])
        if not all(math.isfinite(audit[k]) for k in ("qa_ce", "replay_ce", "distill_mse", "ranking_loss")
                   if audit.get(k) is not None):
            raise ValueError(f"non-finite online loss for {item['sample_id']}")
        objective.backward()
        self.optimizer.step()
        self.seen += 1
        if self.with_memory:
            slot = reservoir_slot(self.rng, self.seen, len(self.memory), self.capacity)
            if slot is not None:
                entry = dict(item, answer_index=answer, teacher_logits=logits.detach().cpu().tolist())
                if slot == len(self.memory):
                    self.memory.append(entry)
                else:
                    self.memory[slot] = entry
        return audit

    def update(self, records):
        history = [self.observe(inference_input(r), r["answer_index"]) for r in records]
        self.model.eval()
        return self.memory, dict(online_history=history, stream_seen=self.seen, optimizer_steps=len(history),
                                 selected_ids=[h["sample_id"] for h in history],
                                 replay_ids=[h["replay_id"] for h in history if h["replay_id"] is not None],
                                 window_extraction_count=0, best_pass=None, training_history=[])


@torch.inference_mode()
def rank_clips(model, question, choices, answer, candidates):
    """Return one score per distinct video. The candidate ID breaks exact score ties."""
    rows = []
    for clip_id, path in sorted(candidates.items()):
        item = dict(sample_id=clip_id, clip_path=str(path), question=question, choices=choices)
        score = match_score(model.score(item), answer).item()
        if not math.isfinite(score):
            raise ValueError(f"non-finite retrieval score for {clip_id}")
        rows.append(dict(clip_id=clip_id, score=score))
    return sorted(rows, key=lambda r: (-r["score"], r["clip_id"]))


def evaluate_retrieval(model, tests, stage, seed):
    clips = {r["clip_id"]: r["clip_path"] for r in tests}
    queries = []
    for record in tests:
        target = record["clip_id"]
        pool = sorted(clips.keys() - {target})
        if not pool:
            continue
        rng = random.Random(f"{seed}:{record['sample_id']}")
        candidates = [target, *rng.sample(pool, min(4, len(pool)))]
        ranking = rank_clips(model, record["question"], record["choices"], record["answer_index"],
                             {c: clips[c] for c in candidates})
        rank = next(i for i, r in enumerate(ranking, 1) if r["clip_id"] == target)
        queries.append(dict(sample_id=record["sample_id"], target_clip=target, rank=rank, ranking=ranking))
    return dict(stage=stage, count=len(queries), queries=queries,
                recall_at_1=fmean(q["rank"] == 1 for q in queries) if queries else None,
                mrr=fmean(1 / q["rank"] for q in queries) if queries else None)
