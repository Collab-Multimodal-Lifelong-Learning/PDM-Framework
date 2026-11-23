"""PyAV clip clock, VideoQA model adapters, and answer-conditioned evidence."""

from __future__ import annotations

import logging
import math
import os
from collections import Counter
from functools import lru_cache

import av
import numpy as np
import torch
from torch import nn

from .data import COLORS, QUESTIONS, read_sensor

QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
VLM_RLAIF_REVISION = "610cac72c16ae3862b0d83a0dc3708d73a42a4cc"
MAX_PIXELS = 256 * 28 * 28
LABELS = "ABCDE"


def _file_key(path):
    path = os.path.realpath(path)
    stat = os.stat(path)
    return path, stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=256)
def _timeline(key):
    """Timestamp-only cache. Decode in presentation order and retain fractional ms."""
    path = key[0]
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        pts = []
        last_duration = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Missing video frame timestamp in {path}")
            t = frame.pts * frame.time_base
            if not pts:
                origin = t
            pts.append(float((t - origin) * 1000))
            last_duration = (getattr(frame, "duration", 0) or 0) * frame.time_base
        if not pts or np.any(np.diff(pts) < 0):
            raise ValueError(f"Empty video or invalid presentation timestamps in {path}")
        if not last_duration:
            if len(pts) > 1 and pts[-1] > pts[-2]:
                last_duration = (pts[-1] - pts[-2]) / 1000
            elif stream.average_rate:
                last_duration = 1 / stream.average_rate
            else:
                raise ValueError(f"Cannot determine final frame duration in {path}")
        # Round the physical end upwards to cover a final sub-millisecond frame.
        duration = math.ceil((t - origin + last_duration) * 1000)
        return np.asarray(pts), duration


def video_duration_ms(path):
    return _timeline(_file_key(path))[1]


def _quarters(pts, duration):
    boundaries = [k * duration // 4 for k in range(5)]
    for start, end in zip(boundaries, boundaries[1:]):
        if pts.searchsorted(start) < pts.searchsorted(end):
            yield (start, end)


def sample_indices(pts, interval, sampler="midpoint"):
    """Select eight presentation-order indices strictly inside the half-open interval."""
    pts = np.asarray(pts)
    start, end = interval
    left, right = pts.searchsorted([start, end])
    if right <= left:
        raise ValueError(f"No video frames in interval {interval}")
    if sampler == "vlm_rlaif":
        # Modified from VLM-RLAIF Evaluation/infer_utils.py::load_frames at
        # 610cac72c16ae3862b0d83a0dc3708d73a42a4cc (Apache-2.0, third_party/vlm-rlaif/LICENSE).
        # Sorted frame files become PyAV presentation-order indices, offset into the clip.
        return (left + np.linspace(0, right - left - 1, 8, dtype=int)).tolist()
    if sampler != "midpoint":
        raise ValueError(f"Unknown frame sampler {sampler!r}")
    inside = pts[left:right]
    targets = start + (np.arange(8) + 0.5) * (end - start) / 8
    upper = np.minimum(inside.searchsorted(targets), len(inside) - 1)
    lower = np.maximum(upper - 1, 0)
    midpoint = (inside[lower] + inside[upper]) / 2
    nearest = np.where(targets <= np.nextafter(midpoint, np.inf), lower, upper)
    return (left + nearest).tolist()


@lru_cache(maxsize=256)
def _rgb_means(key):
    with av.open(key[0]) as container:
        means = [frame.to_ndarray(format="rgb24").mean(axis=(0, 1)) / 255
                 for frame in container.decode(video=0)]
    return np.asarray(means, dtype=np.float32)


class SmokeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(6, 5)
        self.base.requires_grad_(False)
        self.lora_a = nn.Linear(6, 2, bias=False)
        self.lora_b = nn.Linear(2, 5, bias=False)
        nn.init.zeros_(self.lora_b.weight)

    def selector_parameters(self):
        return (self.lora_a.weight, self.lora_b.weight)

    def score(self, item, interval=None, context=(), sampler="midpoint"):
        choices = tuple(item["choices"])
        if len(choices) != len(COLORS) or set(choices) != set(COLORS):
            raise ValueError(f"The smoke choices must be a permutation of {', '.join(COLORS)}")
        key = _file_key(item["clip_path"])
        pts, duration = _timeline(key)
        start, end = interval or (0, duration)
        left, middle, right = pts.searchsorted((start, (start + end) / 2, end))
        if left == right:
            raise ValueError(f"No video frames in interval {(start, end)}")
        rgb = _rgb_means(key)
        # A sparse interval can have no frame in one half; use its nearest observed frame.
        features = np.zeros(6, dtype=np.float32)
        question = item["question"].strip().casefold()
        if question == QUESTIONS[0].casefold():
            features[:3] = rgb[left:max(middle, left + 1)].mean(axis=0)
        elif question == QUESTIONS[1].casefold():
            features[3:] = rgb[min(middle, right - 1):right].mean(axis=0)
        else:
            raise ValueError("Smoke profile supports the first-color and last-color questions")
        x = self.base.weight.new_tensor(features)
        # Rank and alpha are both two, giving a LoRA multiplier of one.
        scores = self.base(x) + self.lora_b(self.lora_a(x))
        if choices != COLORS:
            scores = scores[[COLORS.index(choice) for choice in choices]]
        if context:
            counts = Counter(record["choices"][record["answer_index"]] for record in context)
            bonus = scores.new_tensor([counts[choice] for choice in item["choices"]])
            scores = scores + bonus / bonus.sum().clamp_min(1)
        return scores


@lru_cache(maxsize=2)
def _sampled_rgb(key, sampler, extra_interval):
    """Decode one clip once for its sampled full/quarter views. Keep <= 48 small RGB frames."""
    pts, duration = _timeline(key)
    intervals = [(0, duration), *_quarters(pts, duration)]
    if extra_interval:
        intervals.append(extra_interval)
    wanted = {i for interval in intervals for i in sample_indices(pts, interval, sampler)}
    images = {}
    with av.open(key[0]) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                # Bound CPU cache even for 4K inputs, preserving the original aspect.
                scale = min(1.0, math.sqrt(MAX_PIXELS / (frame.width * frame.height)))
                images[index] = frame.reformat(width=max(1, int(frame.width * scale)),
                                               height=max(1, int(frame.height * scale)),
                                               format="rgb24").to_ndarray()
                if len(images) == len(wanted):
                    break
    return images


def clear_video_cache():
    """Start a timed phase with cold frame, timestamp, and RGB feature caches."""
    _timeline.cache_clear()
    _rgb_means.cache_clear()
    _sampled_rgb.cache_clear()


class _QwenVideoWarnings(logging.Filter):
    """Qwen 4.52.4 forwards both consumed options to the inner video processor."""

    reported = False

    def filter(self, record):
        if record.getMessage() in {
            "Unused or unrecognized kwargs: fps, return_tensors.",
            "Unused or unrecognized kwargs: return_tensors, fps.",
        }:
            first = not self.reported
            self.reported = True
            return first
        return True


_qwen_video_warnings = _QwenVideoWarnings()


class QwenModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("The nextqa profile requires a BF16-capable CUDA device")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError(f"The selected device {device} does not support native BF16")
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        logging.getLogger("transformers.image_utils").addFilter(_qwen_video_warnings)
        self.processor = AutoProcessor.from_pretrained(QWEN_MODEL, revision=QWEN_REVISION, use_fast=False)
        ids = [self.processor.tokenizer.encode(c, add_special_tokens=False) for c in LABELS]
        if any(len(parts) != 1 for parts in ids) or len({parts[0] for parts in ids}) != 5:
            raise ValueError("A–E must encode as distinct single tokens")
        self.label_ids = [parts[0] for parts in ids]
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL, revision=QWEN_REVISION, torch_dtype=torch.bfloat16,
            device_map={"": device}, attn_implementation="sdpa",
        )
        self.backbone = get_peft_model(base, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)",
        ))

    def selector_parameters(self):
        decoder = self.backbone.get_base_model().model.language_model
        return (p for p in decoder.layers[-1].parameters() if p.requires_grad)

    def score(self, item, interval=None, context=(), sampler="midpoint"):
        key = _file_key(item["clip_path"])
        pts, duration = _timeline(key)
        interval = tuple(interval or (0, duration))
        ordinary = interval == (0, duration) or interval in _quarters(pts, duration)
        images = _sampled_rgb(key, sampler, None if ordinary else interval)
        indices = sample_indices(pts, interval, sampler)
        frames = np.stack([images[i] for i in indices])
        lines = []
        for record in context:
            answer = record["choices"][record["answer_index"]]
            lines.append(f"Previous question: {record['question']}\nVerified answer: {answer}")
        lines.append(item["question"])
        lines.extend(f"{letter}. {choice}" for letter, choice in zip(LABELS, item["choices"], strict=True))
        lines.append("Answer with the letter of the correct choice.")
        content = [{"type": "video"}, {"type": "text", "text": "\n".join(lines)}]
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                    tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt], videos=[frames], return_tensors="pt",
            videos_kwargs={"fps": 8000 / (interval[1] - interval[0]), "max_pixels": MAX_PIXELS},
        ).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.backbone(**inputs, use_cache=False)
        return output.logits[0, -1, self.label_ids].float()


def build_model(profile, seed=42, device="cpu"):
    device = torch.device(device)
    if profile == "smoke" and device.type != "cpu":
        raise ValueError("The smoke profile requires the CPU")
    devices = [device] if device.type == "cuda" and torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        if profile == "smoke":
            return SmokeModel()
        if profile == "nextqa":
            return QwenModel(device)
        raise ValueError(f"Unknown profile {profile!r}")


@torch.inference_mode()
def predict(model, item, context=(), sampler="midpoint"):
    pts, duration = _timeline(_file_key(item["clip_path"]))
    scores = model.score(item, context=context, sampler=sampler).float()
    if scores.shape != (5,) or not torch.isfinite(scores).all():
        raise ValueError("Expected five finite full-clip logits")
    choice = int(scores.argmax())
    trace = None
    if item.get("sensor_path"):
        samples = read_sensor(item["sensor_path"], duration)
        timestamps, values = np.asarray(samples).reshape(-1, 2).T
        deviations = abs(values - np.median(values)) if len(values) else values
        trace = timestamps, deviations, deviations.max(initial=0)
    windows = []
    for interval in _quarters(pts, duration):
        logits = model.score(item, interval, context, sampler).float()
        if logits.shape != (5,) or not torch.isfinite(logits).all():
            raise ValueError(f"Expected five finite window logits in {interval}")
        video_score = float(logits.softmax(0)[choice])
        sensor_score = 0.0
        if trace is not None:
            ts, dev, peak = trace
            window = dev[(ts >= interval[0]) & (ts < interval[1])]
            sensor_score = float(window.max(initial=0) / peak) if peak else 0.0
        combined = video_score if trace is None else 0.8 * video_score + 0.2 * sensor_score
        windows.append({"evidence_ms": list(interval), "video_score": video_score,
                        "sensor_score": sensor_score, "combined_score": combined})
    winner = max(windows, key=lambda w: w["combined_score"])
    return {"sample_id": item["sample_id"], "choice_index": choice, "answer": item["choices"][choice],
            "choice_scores": scores.tolist(), "evidence_ms": winner["evidence_ms"],
            "evidence_source": "video_score" if trace is None else "video_sensor_score",
            "window_scores": windows}
