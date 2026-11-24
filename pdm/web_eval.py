"""Evaluate continual browser policies with the official REAL v2 task scripts."""

import fcntl
import hashlib
import json
import shutil
import subprocess
import traceback
from pathlib import Path
from urllib.parse import urlsplit

SITES = ("opendining", "udriver")
DEFAULT_TASKS = ("v2.opendining-5", "v2.udriver-4")
CHECKPOINTS = ("frozen", "sequential-opendining", "sequential-udriver", "replay-opendining", "replay-udriver")
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
# Possible, script-scored tasks in the REAL v2 catalogue packaged with agisdk 0.3.5.
_OFFICIAL_IDS = {
    f"v2.{site}-{number}": site
    for site, numbers in {
        "opendining": (1, 2, 4, 5, 6, 7, 9, 12, 17, 20, 24),
        "udriver": (2, 3, 4, 6, 7, 9, 11, 13, 14, 15, 16, 17, 19, 20),
    }.items()
    for number in numbers
}
_SITE_URLS = {
    "opendining": "https://real-opendining-fdcw98zrb-real-sites.vercel.app/",
    "udriver": "https://real-udriver-3ufk0q1v4-real-sites.vercel.app/",
}
_FINISHED = {"success", "fail"}


def _save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _bind(path, desired):
    """Only reuse results from exactly the same settings and checkpoint."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != desired:
            raise ValueError(f"Evaluation inputs changed at {path}. Use a fresh evaluation directory.")
    else:
        _save(path, desired)


def _read_episode(out, name, task, missing_reason):
    location = Path("real-eval", name, "episodes", task, "record.json")
    site = _OFFICIAL_IDS[task]
    episode = dict(task_id=task, site=site, record=str(location), status="missing",
                   reward=None, error=missing_reason)
    try:
        record = json.loads((out / location).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return episode
    except (OSError, ValueError) as exc:
        return dict(episode, status="error", error=f"Cannot read episode record: {exc}")
    # One OpenDining task was published on the stable replica. All others use versioned replicas.
    official_url = "https://real-opendining.vercel.app" if task == "v2.opendining-17" else _SITE_URLS[site]
    if (not isinstance(record, dict) or record.get("task_id") != task or record.get("site") != site
            or record.get("kind") != "real" or record.get("version") != "v2"
            or record.get("llm_judged") is not False or record.get("url_override")
            or record.get("start_url", "").rstrip("/") != official_url.rstrip("/")):
        raise ValueError(f"Record does not match its official REAL v2 task: {out / location}")
    status = record.get("status")
    if status not in _FINISHED | {"error", "skipped"}:
        raise ValueError(f"Unknown REAL-Eval status {status!r} in {out / location}")
    return {**episode, "status": status, "reward": record["reward"] if status in _FINISHED else None,
            **{key: record.get(key) for key in ("steps", "end", "elapsed", "error", "goal", "start_url")}}


def summarize(episodes, tasks):
    """Compute rates over scored episodes and changes over paired, scored tasks."""
    matrix = {}
    indexed = {}
    errors = []
    for name, rows in episodes.items():
        indexed[name] = {row["task_id"]: row for row in rows}
        matrix[name] = {}
        for site in SITES:
            selected = [row for row in rows if row["site"] == site]
            scored = [row for row in selected if row["status"] in _FINISHED]
            successes = sum(row["status"] == "success" for row in scored)
            matrix[name][site] = dict(successes=successes, scored=len(scored), planned=len(selected),
                                      success_rate=successes / len(scored) if scored else None)
        errors.extend(dict(checkpoint=name, **row) for row in rows if row["status"] not in _FINISHED)

    def paired_change(first, last, site):
        matched = [task for task in tasks if _OFFICIAL_IDS[task] == site
                   and all(indexed[name][task]["status"] in _FINISHED for name in (first, last))]
        first_success = sum(indexed[first][task]["status"] == "success" for task in matched)
        last_success = sum(indexed[last][task]["status"] == "success" for task in matched)
        return dict(task_ids=matched, scored_pairs=len(matched),
                    change=(last_success - first_success) / len(matched) if matched else None)

    forgetting = {}
    adaptation = {}
    for method in ("sequential", "replay"):
        first, last = (f"{method}-{site}" for site in SITES)
        if first in indexed and last in indexed:
            pair = paired_change(first, last, "opendining")
            change = pair.pop("change")
            forgetting[method] = dict(**pair, forgetting=-change if change is not None else None)
        if "frozen" in indexed:
            adaptation[method] = {site: paired_change("frozen", name, site)
                                  for name, site in zip((first, last), SITES) if name in indexed}
    return dict(success_matrix=matrix, opendining_forgetting=forgetting, adaptation_vs_frozen=adaptation,
                infrastructure_errors=errors, episodes=episodes)


def _invoke(folder, executable, config, tasks, checkpoint, serve_checkpoint):
    """Leave endpoint startup failures and CLI exits visible even without a REAL record."""
    returncode = None
    reason = "Harness exited without an episode record. See harness.log."
    try:
        with serve_checkpoint(checkpoint) as endpoint:
            url = urlsplit(endpoint)
            if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("The browser policy endpoint must be local HTTP.")
            _save(folder / "pdm-config.json", config)
            command = [executable, "run", "--version", "v2", "--task", *tasks, "--config",
                       str(folder / "pdm-config.json"), "--base-url", endpoint, "--results-dir",
                       str(folder.parent), "--name", folder.name]
            with (folder / "harness.log").open("a", encoding="utf-8") as log:
                returncode = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            if returncode:
                reason = f"REAL-Eval exited with code {returncode}. See harness.log."
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        with (folder / "harness.log").open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
    _save(folder / "pdm-invocation.json", dict(returncode=returncode, missing_reason=reason))


def evaluate(out, real_eval, checkpoints, serve_checkpoint, max_steps=25, *, tasks=DEFAULT_TASKS,
             model_name=MODEL_NAME, task_timeout=240, max_tokens=128, seed=42):
    """Run matched browser episodes. serve_checkpoint(path_or_None) yields a local /v1 URL.

    A repeated call may add checkpoints, or retry unavailable episodes with the same
    inputs. Each official harness run retains its original records, traces and screenshots.
    """
    out = Path(out).resolve()
    executable = shutil.which(str(real_eval))
    if executable is None:
        raise FileNotFoundError(f"Cannot find the REAL-Eval executable: {real_eval}")
    tasks = tuple(task if task.startswith("v2.") else f"v2.{task}" for task in tasks)
    if (len(tasks) != len(set(tasks)) or any(task not in _OFFICIAL_IDS for task in tasks)
            or {_OFFICIAL_IDS[task] for task in tasks} != set(SITES)):
        raise ValueError("Select distinct official v2 OpenDining and UDriver tasks, at least one per site.")
    if not checkpoints or set(checkpoints) - set(CHECKPOINTS):
        raise ValueError(f"Expected checkpoint names from {CHECKPOINTS}")
    if any((name == "frozen") != (checkpoint is None) for name, checkpoint in checkpoints.items()):
        raise ValueError("Use None for frozen and a saved adapter path for each trained checkpoint.")
    if min(max_steps, max_tokens, task_timeout) < 1:
        raise ValueError("Step, token and timeout budgets must be positive.")
    config = {
        "model": dict(name=model_name, temperature=0.0, top_p=None, top_k=None, enable_thinking=None,
                      max_tokens=max_tokens, seed=seed),
        "agent": dict(prompt_style="basic", use_screenshot=True, axtree_max_chars=6000,
                      jpeg_quality=75, parse_retries=0, site_guides={}, task_hints={}, agentlab_hint_sites=[],
                      system_prompt_extra="Return only one browser action in a fenced action block."),
        "browser": dict(max_steps=max_steps, headless=True, record_video=False, width=1280, height=720),
        "run": dict(workers=1, task_timeout=task_timeout, site_url_overrides={}),
    }
    executable = str(Path(executable).resolve())
    protocol = dict(version="v2", sdk="agisdk==0.3.5", executable=executable, tasks=list(tasks), config=config)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        root = out / "real-eval"
        _bind(out / "web-protocol.json", protocol)
        for name, checkpoint in checkpoints.items():
            folder = root / name
            folder.mkdir(parents=True, exist_ok=True)
            digest = None
            if checkpoint is not None:
                with Path(checkpoint).open("rb") as weights:
                    digest = hashlib.file_digest(weights, "sha256").hexdigest()
            _bind(folder / "pdm-checkpoint.json", dict(name=name, sha256=digest))
            rows = [_read_episode(out, name, task, "No episode record yet") for task in tasks]
            if all(row["status"] in _FINISHED for row in rows):
                continue
            _invoke(folder, executable, config, tasks, checkpoint, serve_checkpoint)
        episodes = {}
        environments = {}
        for name in CHECKPOINTS:
            folder = root / name
            if (folder / "pdm-checkpoint.json").exists():
                info_path = folder / "run.json"
                if info_path.exists():
                    env = json.loads(info_path.read_text(encoding="utf-8"))["environment"]
                    if env.get("agisdk") != "0.3.5":
                        raise ValueError(f"Unexpected REAL SDK version in {info_path}")
                    if environments and env != next(iter(environments.values())):
                        raise ValueError(f"REAL-Eval runtime differs across checkpoints in {info_path}")
                    environments[name] = env
                invocation = folder / "pdm-invocation.json"
                reason = (json.loads(invocation.read_text())["missing_reason"] if invocation.exists()
                          else "Evaluation interrupted before the harness finished.")
                episodes[name] = [_read_episode(out, name, task, reason) for task in tasks]
        result = dict(protocol=protocol, environments=environments, **summarize(episodes, tasks))
        _save(out / "web-summary.json", result)
        return result
