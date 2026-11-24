"""Evaluate continual browser policies with the official REAL v2 task scripts."""

import fcntl
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
from pathlib import Path
from .common import QWEN_MODEL, QWEN_REVISION, _atomic_path

SITES = ("opendining", "udriver")
DEFAULT_TASKS = ("v2.opendining-5", "v2.udriver-4")
CHECKPOINTS = ("frozen", "sequential-opendining", "sequential-udriver", "replay-opendining", "replay-udriver")
EPISODE_TIMEOUT = 240
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
    with _atomic_path(path) as temporary:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                             encoding="utf-8")


def _bind(path, desired):
    """Only reuse results from exactly the same settings and checkpoint."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != desired:
            raise ValueError(f"Evaluation inputs changed at {path}. Use a fresh evaluation directory.")
    else:
        _save(path, desired)


def _read_episode(out, name, task):
    location = Path("checkpoints", name, "episodes", task, "record.json")
    site = _OFFICIAL_IDS[task]
    path = out / location
    if not path.exists():
        return dict(task_id=task, site=site, record=str(location), status="missing", reward=None,
                    error="Evaluation interrupted before the episode was recorded.")
    row = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(row, dict) or row.get("task_id") != task or row.get("site") != site:
        raise ValueError(f"Record does not match the official task at {path}")
    status = row.get("status")
    if status in _FINISHED:
        official_url = "https://real-opendining.vercel.app" if task == "v2.opendining-17" else _SITE_URLS[site]
        if (row.get("start_url") or "").rstrip("/") != official_url.rstrip("/"):
            raise ValueError(f"Record does not match the official task URL at {path}")
        reward = row.get("reward")
        if type(reward) not in (int, float) or not math.isfinite(reward):
            raise ValueError(f"Invalid scored reward at {path}")
        if type(row.get("steps")) is not int or row["steps"] < 0 or not row.get("end"):
            raise ValueError(f"Missing episode outcome at {path}")
        env = row.get("environment")
        if not isinstance(env, dict) or env.get("agisdk") != "0.3.5":
            raise ValueError(f"Unexpected browser runtime at {path}")
        _bind(out / "browser-runtime.json", env)
    elif status not in {"missing", "error", "skipped"} or row.get("reward") is not None:
        raise ValueError(f"Invalid episode outcome at {path}")
    return dict(row, record=str(location))


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


def _error(folder, task, reason):
    _save(folder / "record.json", dict(task_id=task, site=_OFFICIAL_IDS[task], status="error",
                                       reward=None, steps=None, end="error", error=reason))


def _run_episode(executable, endpoint, task, folder, max_steps):
    command = [executable, "-m", "pdm.browser_cli", "episode", "--task", task, "--out", str(folder),
               "--endpoint", endpoint, "--max-steps", str(max_steps)]
    with (folder / "episode.log").open("a", encoding="utf-8") as log:
        # Use this checkout's policy code even when the browser interpreter lives in a separate venv.
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parent.parent, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=EPISODE_TIMEOUT)
        except BaseException:
            # Chromium and Playwright are descendants of the worker; reap the entire episode on timeout.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    path = folder / "record.json"
    if code and path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") in {"error", "skipped"}:
            return
    if code or not path.exists():
        _error(folder, task, f"Browser worker exited with code {code} without a valid outcome. See episode.log.")


def _prepare_retry(folder):
    # Keep evidence from earlier attempts, and never mistake it for output from a new worker.
    if folder.exists():
        archive = folder.parent.parent / "attempts" / folder.name
        archive.mkdir(parents=True, exist_ok=True)
        index = 1
        while (archive / str(index)).exists():
            index += 1
        folder.rename(archive / str(index))
    folder.mkdir(parents=True)


def evaluate(out, browser_python, checkpoints, serve_checkpoint, max_steps=25, *, tasks=DEFAULT_TASKS):
    """Evaluate matched official tasks, retrying only episodes without a score."""
    out = Path(out).resolve()
    tasks = tuple(task if task.startswith("v2.") else f"v2.{task}" for task in tasks)
    if (len(tasks) != len(set(tasks)) or any(task not in _OFFICIAL_IDS for task in tasks)
            or {_OFFICIAL_IDS[task] for task in tasks} != set(SITES)):
        raise ValueError("Select distinct official v2 OpenDining and UDriver tasks, at least one per site.")
    if not checkpoints or set(checkpoints) - set(CHECKPOINTS):
        raise ValueError(f"Expected checkpoint names from {CHECKPOINTS}")
    if any((name == "frozen") != (checkpoint is None) for name, checkpoint in checkpoints.items()):
        raise ValueError("Use None for frozen and a saved adapter path for each trained checkpoint.")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    executable = shutil.which(str(browser_python))
    if executable is None:
        raise FileNotFoundError(f"Cannot find browser Python interpreter: {browser_python}")
    # Keep the venv's path intact. Resolving the executable symlink would select the base installation.
    executable = os.path.abspath(executable)
    root = Path(__file__).parent
    source_files = ("browser.py", "browser_cli.py", "web_model.py", "web_eval.py")
    protocol = dict(version="v2", sdk="agisdk==0.3.5", browser_python=executable, tasks=list(tasks),
                    model=QWEN_MODEL, revision=QWEN_REVISION, decoding="greedy", max_steps=max_steps,
                    task_timeout_seconds=EPISODE_TIMEOUT,
                    source_sha256={file: hashlib.sha256((root / file).read_bytes()).hexdigest()
                                   for file in source_files})
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _bind(out / "web-protocol.json", protocol)
        for name, checkpoint in checkpoints.items():
            folder = out / "checkpoints" / name
            digest = None
            if checkpoint is not None:
                with Path(checkpoint).open("rb") as weights:
                    digest = hashlib.file_digest(weights, "sha256").hexdigest()
            _bind(folder / "pdm-checkpoint.json", dict(name=name, sha256=digest))
            pending = [task for task in tasks if _read_episode(out, name, task)["status"] not in _FINISHED]
            if not pending:
                continue
            for task in pending:
                _prepare_retry(folder / "episodes" / task)
            try:
                with serve_checkpoint(checkpoint) as endpoint:
                    for task in pending:
                        episode = folder / "episodes" / task
                        try:
                            _run_episode(executable, endpoint, task, episode, max_steps)
                        except subprocess.TimeoutExpired:
                            _error(episode, task, f"Episode timeout after {EPISODE_TIMEOUT} seconds. See episode.log.")
                        except (OSError, ValueError) as exc:
                            _error(episode, task, f"{type(exc).__name__}: {exc}")
            except Exception as exc:
                for task in pending:
                    episode = folder / "episodes" / task
                    if not (episode / "record.json").exists():
                        _error(episode, task, f"{type(exc).__name__}: {exc}")

        episodes = {name: [_read_episode(out, name, task) for task in tasks] for name in CHECKPOINTS
                    if (out / "checkpoints" / name / "pdm-checkpoint.json").exists()}
        result = dict(protocol=protocol, **summarize(episodes, tasks))
        _save(out / "web-summary.json", result)
        return result
