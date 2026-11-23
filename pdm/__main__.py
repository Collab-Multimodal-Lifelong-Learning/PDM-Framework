"""Command-line entry point for PDM VideoQA."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from .continual import CONDITIONS, retrieve
from .data import generate_fixture, inference_input, load_manifest, prepare_nextqa, sensor_fixture
from .experiment import checksum, restore, run_experiment, write_json
from .models import build_model, predict


def media_options(parser):
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--choice", action="append", help="repeat for each of five ordered choices")
    parser.add_argument("--sensor-csv", type=Path)


def arguments():
    parser = argparse.ArgumentParser(prog="pdm-videoqa", description="Continual multiple-choice VideoQA")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run the generated CPU or real-video GPU demo")
    demo.add_argument("--profile", choices=("smoke", "nextqa-quick"), default="smoke")
    demo.add_argument("--device", help="CPU for smoke or a CUDA device for real video")
    demo.add_argument("--data-dir", type=Path, default=Path.home() / ".cache/pdm-framework/nextqa",
                      help="download cache for the real-video demo")
    run = commands.add_parser("run", help="train and evaluate using a saved manifest")
    for sub in (demo, run):
        sub.add_argument("--out", required=True, type=Path, help="a new run directory")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--device")
    prep = commands.add_parser("prepare-nextqa", help="import the pinned NExT-QA annotations")
    prep.add_argument("--annotations", type=Path, required=True)
    prep.add_argument("--videos", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--quick", action="store_true", help="select 8 train, 2 validation, 3 test questions per task")
    prediction = commands.add_parser("predict", help="answer with a saved condition and checkpoint")
    prediction.add_argument("--run-dir", type=Path, required=True)
    prediction.add_argument("--condition", choices=CONDITIONS, default="pdm")
    prediction.add_argument("--stage", type=int, choices=[1, 2], default=2, help="how many tasks have been seen")
    prediction.add_argument("--sample-id")
    media_options(prediction)
    ask = commands.add_parser("ask", help="score a new video directly with pretrained Qwen")
    ask.add_argument("--device", default="cuda:0")
    media_options(ask)
    return parser.parse_args()


def new_input(args):
    if not args.clip or not args.question or not args.question.strip() or len(args.choice or ()) != 5:
        raise ValueError("supply --clip, --question and exactly five --choice arguments")
    if not all(s.strip() for s in args.choice):
        raise ValueError("choices must be nonempty")
    path = args.clip.resolve(strict=True)
    key = json.dumps([checksum(path), args.question, args.choice], ensure_ascii=False)
    item = {"sample_id": hashlib.sha256(key.encode()).hexdigest(), "clip_path": str(path),
            "question": args.question, "choices": args.choice}
    if args.sensor_csv:
        item["sensor_path"] = str(args.sensor_csv.resolve(strict=True))
    return item


def read_config(path):
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or set(config) != {"profile", "steps", "learning_rate", "memory_capacity"}:
        raise ValueError("configuration needs profile, steps, learning_rate and memory_capacity")
    if config["profile"] not in ("smoke", "nextqa", "nextqa_quick"):
        raise ValueError("unknown model/data profile")
    for key in ("steps", "memory_capacity"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    lr = config["learning_rate"]
    if type(lr) not in (float, int) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning_rate must be positive and finite")
    return config


def execute(args):
    if args.command == "prepare-nextqa":
        records = prepare_nextqa(args.annotations, args.videos, args.out, quick=args.quick)
        print(f"{len(records)} questions → {args.out}")
        return
    if args.command in ("predict", "ask"):
        context = ()
        if args.command == "ask":
            item = new_input(args)
            model = build_model("nextqa", 42, args.device)
        else:
            model, records, memory = restore(args.run_dir, args.condition, args.stage)
            if args.sample_id:
                if any((args.clip, args.question, args.choice, args.sensor_csv)):
                    raise ValueError("use either --sample-id or the new-clip arguments")
                if args.sample_id not in records:
                    raise ValueError(f"unknown sample ID {args.sample_id!r}")
                item = inference_input(records[args.sample_id])
            else:
                item = new_input(args)
            if args.condition == "retrieval":
                context = retrieve(item, memory)
        result = predict(model.eval(), item, context)
        print(json.dumps(dict(result, sample_id=item["sample_id"]), allow_nan=False))
        return
    demo = args.command == "demo"
    path = Path(__file__).parent / "configs" / f"{args.profile}.json" if demo else args.config
    config = read_config(path)
    device = args.device or ("cpu" if config["profile"] == "smoke" else "cuda:0")
    if len(args.conditions) != len(set(args.conditions)):
        raise ValueError("conditions must be unique")
    # Exclusive creation prevents concurrent writers and keeps earlier results intact.
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        sensor = None
        if demo and config["profile"] == "smoke":
            records = generate_fixture(out)
            sensor = sensor_fixture(out)
        else:
            if demo:
                from .download import fetch_quick
                manifest = fetch_quick(args.data_dir)
            else:
                manifest = args.manifest
            records = load_manifest(manifest, config["profile"])
        run_experiment(out, records, config, args.seed, args.conditions, device, sensor)
    except Exception as error:
        recorded = out / "run.json"
        info = json.loads(recorded.read_text()) if recorded.exists() else {}
        write_json(recorded, dict(info, status="failed", error=str(error)))
        raise
    print(out / "report.html")


def main():
    try:
        execute(arguments())
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"pdm-videoqa: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
