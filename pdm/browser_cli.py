"""Run one browser episode against a locally served PDM policy."""

import argparse
from http.client import HTTPConnection
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from urllib.parse import urlsplit

from .common import QWEN_MODEL, _atomic_path


def _environment():
    packages = {pkg: version(pkg) for pkg in ("agisdk", "gymnasium", "playwright")}
    if packages["agisdk"] != "0.3.5":
        raise ValueError("The protocol requires agisdk==0.3.5.")
    return dict(python=platform.python_version(), **packages)


def _check_endpoint(endpoint):
    url = urlsplit(endpoint)
    if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("The model endpoint must use local HTTP.")
    connection = HTTPConnection(url.hostname, url.port, timeout=10)
    try:
        connection.request("GET", url.path.rstrip("/") + "/models")
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Model endpoint returned HTTP {response.status} {response.reason}")
        models = json.load(response)["data"]
        if not any(model.get("id") == QWEN_MODEL for model in models):
            raise ValueError(f"Model endpoint does not serve {QWEN_MODEL}")
    finally:
        connection.close()


def main(argv=None):
    from .web_eval import SITES, _OFFICIAL_IDS

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    episode = commands.add_parser("episode", help="score one official v2 instruction")
    episode.add_argument("--task", choices=sorted(_OFFICIAL_IDS), required=True)
    ask = commands.add_parser("ask", help="run your instruction in a fresh browser")
    ask.add_argument("--goal", required=True)
    ask.add_argument("--site", choices=SITES, required=True)
    for cmd in (episode, ask):
        cmd.add_argument("--endpoint", default="http://127.0.0.1:8017/v1")
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--max-steps", type=int, default=25)
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.command == "ask" and not args.goal.strip():
        parser.error("--goal must contain an instruction")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "record.json").exists():
        parser.error("This output already contains an episode. Choose a fresh directory.")

    custom = args.command == "ask"
    site = args.site if custom else _OFFICIAL_IDS[args.task]
    task = f"ask-{site}" if custom else args.task
    try:
        _check_endpoint(args.endpoint)
        environment = _environment()
        from .browser import run_episode

        record = run_episode(task, endpoint=args.endpoint, out=args.out, max_steps=args.max_steps,
                             goal=args.goal if custom else None, site=site if custom else None)
        record["environment"] = environment
    except Exception as exc:
        record = dict(task_id=task, site=site, status="error", reward=None, steps=0,
                      end="error", error=f"{type(exc).__name__}: {exc}")

    path = args.out / "record.json"
    with _atomic_path(path) as temp:
        temp.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(path)
    if record.get("answer"):
        print(record["answer"])
    if record["status"] in {"skipped", "error", "missing"}:
        print(record.get("error") or record["status"], file=sys.stderr)
        return 2
    return 1 if custom and record["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
