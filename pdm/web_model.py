"""Generative browser-action policy using the pinned Qwen visual language model."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import functional as F

from .experiment import fingerprint, trainable_state
from .models import MAX_PIXELS, QWEN_MODEL, QWEN_REVISION, build_model


class WebPolicy:
    """One persistent AdamW optimizer over the language decoder's Q/V LoRA weights."""

    def __init__(self, device, *, seed=42, lr=1e-4):
        self.device = torch.device(device)
        model = build_model("nextqa", seed=seed, device=self.device)
        self.processor = model.processor
        self.backbone = model.backbone
        self.optimizer = torch.optim.AdamW(
            (p for p in self.backbone.parameters() if p.requires_grad), lr=lr, weight_decay=0.0
        )

    def _inputs(self, system, prompt, image):
        from PIL import Image

        if isinstance(image, Image.Image):
            rgb = image.convert("RGB")
        else:
            with Image.open(image) as source:
                rgb = source.convert("RGB")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        ]
        prefix = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.processor(
            text=[prefix], images=[rgb], images_kwargs={"max_pixels": MAX_PIXELS}, return_tensors="pt"
        ).to(self.device)

    def _action_loss(self, row):
        inputs = self._inputs(row["system"], row["prompt"], row["image"])
        tokenizer = self.processor.tokenizer
        action_ids = tokenizer.encode(row["action"], add_special_tokens=False)
        if not action_ids:
            raise ValueError("Browser action is empty")
        prefix = inputs["input_ids"]
        targets = prefix.new_tensor([*action_ids, tokenizer.eos_token_id])
        inputs["input_ids"] = torch.cat([prefix, targets[:-1].unsqueeze(0)], dim=1)
        mask = inputs["attention_mask"]
        inputs["attention_mask"] = torch.cat([mask, mask.new_ones((1, len(action_ids)))], dim=1)
        # LoRA is injected in the decoder modules. Project only the action-predicting
        # positions to the vocabulary, avoiding a full AX-tree-length logits tensor.
        base = self.backbone.get_base_model()
        hidden = base.model(**inputs, use_cache=False).last_hidden_state
        logits = base.lm_head(hidden[0, -len(targets):])
        loss = F.cross_entropy(logits.float(), targets)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite browser action loss")
        return loss

    @torch.inference_mode()
    def action_nll(self, examples) -> float:
        """Score teacher-forced action tokens and EOS with equal weight per example."""
        if not examples:
            raise ValueError("Expected at least one browser action")
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            return sum(self._action_loss(row).item() for row in examples) / len(examples)
        finally:
            self.backbone.train(was_training)

    def update(self, examples):
        """Perform one optimizer step with equal weight per example. Return mean CE."""
        if not examples:
            raise ValueError("Expected at least one browser action")
        self.backbone.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for row in examples:
            loss = self._action_loss(row)
            (loss / len(examples)).backward()
            total += loss.item()
        self.optimizer.step()
        return total / len(examples)

    @torch.inference_mode()
    def generate(self, system, prompt, image, *, max_new_tokens=128):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be an integer from 1 to 128")
        self.backbone.eval()
        inputs = self._inputs(system, prompt, image)
        prefix_length = inputs["input_ids"].shape[1]
        ids = self.backbone.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0,
            repetition_penalty=1.0, use_cache=True,
        )
        return self.processor.tokenizer.decode(
            ids[0, prefix_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

    @staticmethod
    def _metadata():
        return {
            "format": "pdm-web-action-lora-v1",
            "base_model": QWEN_MODEL,
            "model_revision": QWEN_REVISION,
            "max_pixels": str(MAX_PIXELS),
        }

    def save(self, path):
        """Atomically write a safetensors adapter snapshot suitable for inference."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = trainable_state(self.backbone)
        metadata = dict(self._metadata(), adapter_sha256=fingerprint(tensors))
        with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            save_file(tensors, temp_path, metadata=metadata)
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)

    @torch.no_grad()
    def load(self, path):
        """Validate the adapter snapshot, replace its weights and reset optimizer moments."""
        params = {k: p for k, p in self.backbone.named_parameters() if p.requires_grad}
        with safe_open(path, framework="pt", device="cpu") as stream:
            metadata = stream.metadata() or {}
            for key, expected in self._metadata().items():
                if metadata.get(key) != expected:
                    raise ValueError(f"Browser checkpoint metadata mismatch for {key}")
            if set(stream.keys()) != params.keys():
                raise ValueError("Browser checkpoint parameters differ from the adapter")
            tensors = {name: stream.get_tensor(name) for name in params}
            if fingerprint(tensors) != metadata.get("adapter_sha256"):
                raise ValueError("Browser checkpoint adapter checksum mismatch")
            for name, tensor in tensors.items():
                p = params[name]
                if tensor.shape != p.shape or tensor.dtype != p.dtype or not tensor.isfinite().all():
                    raise ValueError(f"Invalid browser checkpoint tensor {name}")
            for name, tensor in tensors.items():
                params[name].copy_(tensor)
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.state.clear()
