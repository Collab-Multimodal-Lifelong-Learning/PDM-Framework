"""Train and serve PDM browser-action adapters, and score them in REAL-Eval."""
import argparse
import base64
import hashlib
import io
import json
import math
import random
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image

from .data import _atomic_path
from .models import QWEN_MODEL, QWEN_REVISION
from .stream import reservoir_slot

DOMAINS = ("opendining", "udriver")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, obj):
    with _atomic_path(path) as temp:
        temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_examples(manifest):
    root = manifest.resolve().parent
    groups = {d: [] for d in DOMAINS}
    counts = Counter()
    episode_domains = {}
    for n, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k].strip()
                for k in ("episode", "domain", "split", "image", "prompt", "action", "system")):
            raise ValueError(f"missing observation or action at line {n}")
        episode, domain = row["episode"], row["domain"]
        if row["split"] != "train" or not episode.startswith("train-") or domain not in groups:
            raise ValueError(f"expected a training demonstration from a supported domain at line {n}")
        if episode_domains.setdefault(episode, domain) != domain:
            raise ValueError(f"episode {episode} belongs to multiple domains")
        image = (root / row["image"]).resolve(strict=True)
        if Path(row["image"]).is_absolute() or not image.is_relative_to(root):
            raise ValueError(f"image must be stored beside manifest at line {n}")
        with Image.open(image) as pic:
            pic.verify()
        counts[episode] += 1
        groups[domain].append(dict(row, image=image, id=f"{episode}:{counts[episode]}", image_sha256=digest(image)))
    if not all(groups.values()):
        raise ValueError("provide verified training demonstrations from both OpenDining and UDriver")
    return groups


def train(manifest, out, policy_factory, *, seed, passes, capacity):
    if passes < 1 or capacity < 1:
        raise ValueError("passes and memory capacity must be positive")
    groups = read_examples(manifest)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", seed=seed, passes=passes, memory_capacity=capacity,
                  model=QWEN_MODEL, revision=QWEN_REVISION, manifest_sha256=digest(manifest),
                  domains=list(DOMAINS), updates={}, checkpoints={"frozen": None}, checkpoint_sha256={},
                  example_images={r["id"]: r["image_sha256"] for rows in groups.values() for r in rows})
    path = out / "training.json"
    write_json(path, report)
    memory = []
    try:
        def fit(name, policy, rows, replay):
            order_rng, sample_rng = random.Random(seed), random.Random(seed + 1)
            history = []
            for epoch in range(passes):
                order = rows.copy()
                order_rng.shuffle(order)
                for index, current in enumerate(order):
                    other = sample_rng.choice(replay) if replay else order[(index + 1) % len(order)]
                    loss = policy.update([current, other])
                    if not math.isfinite(loss):
                        raise ValueError(f"non-finite training loss at {name}, {current['id']}")
                    history.append(dict(epoch=epoch + 1, batch=[current["id"], other["id"]], loss=loss))
            adapter = out / f"{name}.safetensors"
            policy.save(adapter)
            report["updates"][name] = history
            report["checkpoints"][name] = adapter.name
            report["checkpoint_sha256"][adapter.name] = digest(adapter)
            return adapter

        first = groups[DOMAINS[0]]
        rng = random.Random(seed)
        for seen, example in enumerate(first, 1):
            slot = reservoir_slot(rng, seen, len(memory), capacity)
            if slot == len(memory):
                memory.append(example)
            elif slot is not None:
                memory[slot] = example
        report["replay_memory"] = [r["id"] for r in memory]
        policy = policy_factory()
        common = fit("sequential-opendining", policy, first, [])
        del policy
        report["checkpoints"]["replay-opendining"] = common.name
        for method in ("sequential", "replay"):
            policy = policy_factory()
            policy.load(common)
            fit(f"{method}-udriver", policy, groups[DOMAINS[1]], memory if method == "replay" else [])
            del policy
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(path, report)
    return report


def decode_observation(messages):
    if (not isinstance(messages, list) or len(messages) != 2
            or [m["role"] for m in messages] != ["system", "user"]):
        raise ValueError("expected one system message and one multimodal observation")
    system = messages[0]["content"]
    parts = messages[1]["content"]
    images = [part["image_url"]["url"] for part in parts if part["type"] == "image_url"]
    text = "\n\n".join(part["text"] for part in parts if part["type"] == "text")
    if not isinstance(system, str) or not text.strip() or len(images) != 1:
        raise ValueError("expected system text, observation text and one screenshot")
    header, _, encoded = images[0].partition(",")
    if header not in ("data:image/jpeg;base64", "data:image/png;base64"):
        raise ValueError("expected an embedded JPEG or PNG screenshot")
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as img:
        if img.width * img.height > 16_000_000:
            raise ValueError("screenshot exceeds the supported resolution")
        return system, text, img.convert("RGB")


@contextmanager
def serve_policy(policy, port=0):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path != "/v1/models":
                self.send_error(404)
                return
            self.reply(200, {"data": [{"id": QWEN_MODEL, "object": "model"}]})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 12 * 1024 * 1024:
                    raise ValueError("invalid observation size")
                request = json.loads(self.rfile.read(length))
                if request.get("model") != QWEN_MODEL:
                    raise ValueError("unknown model")
                limit = request.get("max_tokens", 128)
                if type(limit) is not int or not 1 <= limit <= 128:
                    raise ValueError("max_tokens must be an integer from 1 to 128")
                obs = decode_observation(request["messages"])
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.reply(400, {"error": {"message": str(exc)}})
                return
            try:
                action = policy.generate(*obs, max_new_tokens=limit)
            except (RuntimeError, ValueError) as exc:
                self.reply(500, {"error": {"message": str(exc)}})
                return
            self.reply(200, {"id": "pdm", "object": "chat.completion", "created": 0, "model": QWEN_MODEL,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": action},
                                          "finish_reason": "stop"}]})

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1"
        finally:
            server.shutdown()
            thread.join()


def checkpoints_from(run):
    record = json.loads((run / "training.json").read_text())
    if record["status"] != "complete" or record["revision"] != QWEN_REVISION:
        raise ValueError("training is incomplete or uses a different model revision")
    checkpoints = {}
    for name, filename in record["checkpoints"].items():
        path = (run / filename) if filename else None
        if path is not None and digest(path) != record["checkpoint_sha256"][filename]:
            raise ValueError(f"adapter fingerprint mismatch for {name}")
        checkpoints[name] = path
    return checkpoints


def score_demos(run, manifest, policy_factory):
    """Descriptive fit on the original demonstrations, after all training is finished."""
    checkpoints = checkpoints_from(run)
    groups = read_examples(manifest)
    metadata = json.loads((run / "training.json").read_text())
    images = {r["id"]: r["image_sha256"] for rows in groups.values() for r in rows}
    if digest(manifest) != metadata["manifest_sha256"] or images != metadata["example_images"]:
        raise ValueError("training demonstrations differ from the original run")
    unique, scores = {}, {}
    for name, adapter in checkpoints.items():
        if adapter not in unique:
            policy = policy_factory()
            if adapter is not None:
                policy.load(adapter)
            unique[adapter] = {d: policy.action_nll(rows) for d, rows in groups.items()}
            del policy
        scores[name] = unique[adapter]
    write_json(run / "demonstration-fit.json", scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--manifest", required=True, type=Path)
    tr.add_argument("--out", required=True, type=Path)
    tr.add_argument("--passes", type=int, default=2)
    tr.add_argument("--capacity", type=int, default=8)
    tr.add_argument("--seed", type=int, default=42)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--run", required=True, type=Path)
    evaluation.add_argument("--real-eval", default="real-eval")
    evaluation.add_argument("--out", required=True, type=Path)
    evaluation.add_argument("--max-steps", type=int, default=25)
    evaluation.add_argument("--tasks", nargs="+", help="held-out official v2 tasks across both sites")
    fit = sub.add_parser("score-demos", help="measure adapter fit on the original training demonstrations")
    fit.add_argument("--run", required=True, type=Path)
    fit.add_argument("--manifest", required=True, type=Path)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8017)
    serve.add_argument("--checkpoint", type=Path, help="saved action adapter (omit for the frozen baseline)")
    for cmd in (tr, evaluation, fit, serve):
        cmd.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from .web_model import WebPolicy
    try:
        if args.command == "train":
            train(args.manifest, args.out, lambda: WebPolicy(args.device, seed=args.seed),
                  seed=args.seed, passes=args.passes, capacity=args.capacity)
            print(args.out / "training.json")
            return 0

        if args.command == "score-demos":
            score_demos(args.run, args.manifest, lambda: WebPolicy(args.device))
            print(args.run / "demonstration-fit.json")
            return 0

        @contextmanager
        def serve_checkpoint(path):
            policy = WebPolicy(args.device)
            if path is not None:
                policy.load(path)
            with serve_policy(policy, getattr(args, "port", 0)) as url:
                yield url

        if args.command == "serve":
            with serve_checkpoint(args.checkpoint) as url:
                print(f"Serving {QWEN_MODEL} at {url}", flush=True)
                threading.Event().wait()
        else:
            from .web_eval import evaluate
            kwargs = {"tasks": args.tasks} if args.tasks else {}
            result = evaluate(args.out, args.real_eval, checkpoints_from(args.run), serve_checkpoint, args.max_steps, **kwargs)
            print(args.out / "web-summary.json")
            if result["infrastructure_errors"]:
                print("Some episodes were unavailable. Rerun with the same output directory to retry.", file=sys.stderr)
                return 2
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"pdm-web: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
