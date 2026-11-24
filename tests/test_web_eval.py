import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from pdm import web_eval


def test_forgetting_only_uses_paired_scored_episodes():
    tasks = ("v2.opendining-5", "v2.opendining-7", "v2.udriver-4")

    def rows(statuses):
        return [dict(task_id=task, site=task.split(".")[1].split("-")[0], status=status)
                for task, status in zip(tasks, statuses)]

    report = web_eval.summarize({
        "frozen": rows(["fail", "success", "skipped"]),
        "sequential-opendining": rows(["success", "fail", "fail"]),
        "sequential-udriver": rows(["fail", "error", "success"]),
    }, tasks)
    first = report["success_matrix"]["sequential-opendining"]["opendining"]
    assert first == dict(successes=1, planned=2, scored=2, success_rate=0.5)
    assert report["success_matrix"]["frozen"]["udriver"]["success_rate"] is None
    assert report["opendining_forgetting"]["sequential"] == {
        "task_ids": ["v2.opendining-5"], "scored_pairs": 1, "forgetting": 1.0,
    }
    assert report["adaptation_vs_frozen"]["sequential"]["udriver"]["change"] is None
    assert len(report["infrastructure_errors"]) == 2


def test_cli_conditions_resume_and_checkpoint_identity(tmp_path, monkeypatch):
    invoked = []
    status = {"v2.opendining-5": "success", "v2.udriver-4": "skipped"}

    @contextmanager
    def serve(_):
        yield "http://127.0.0.1:9000/v1"

    def harness(command, **kwargs):
        assert command[1:6] == ["run", "--version", "v2", "--task", "v2.opendining-5"]
        assert command[6] == "v2.udriver-4"
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert config["agent"]["prompt_style"] == "basic"
        assert config["agent"]["use_screenshot"] is True
        assert config["agent"]["task_hints"] == {}
        assert config["run"]["site_url_overrides"] == {}
        assert config["agent"]["parse_retries"] == config["model"]["temperature"] == 0
        name = command[command.index("--name") + 1]
        root = Path(command[command.index("--results-dir") + 1]) / name
        for task, result in status.items():
            site = task.split(".")[1].split("-")[0]
            folder = root / "episodes" / task
            folder.mkdir(exist_ok=True, parents=True)
            row = dict(task_id=task, site=site, start_url=web_eval._SITE_URLS[site], kind="real", version="v2",
                       llm_judged=False, url_override=None, status=result, reward=float(result == "success"))
            (folder / "record.json").write_text(json.dumps(row))
        environment = dict(agisdk="0.3.5", realeval="0.1.0", git_commit="mock", playwright="mock")
        (root / "run.json").write_text(json.dumps(dict(environment=environment)))
        invoked.append(name)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(web_eval.subprocess, "run", harness)
    run = tmp_path / "evaluation"
    web_eval.evaluate(run, "true", {"frozen": None}, serve)
    assert len(json.loads((run / "web-summary.json").read_text())["infrastructure_errors"]) == 1
    status["v2.udriver-4"] = "fail"
    web_eval.evaluate(run, "true", {"frozen": None}, serve)
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"first adapter")
    checkpoints = {"frozen": None, "sequential-opendining": weights}
    scores = web_eval.evaluate(run, "true", checkpoints, serve)
    assert invoked == ["frozen", "frozen", "sequential-opendining"]
    assert set(scores["success_matrix"]) == set(checkpoints)
    assert not scores["infrastructure_errors"]
    with pytest.raises(ValueError, match="inputs changed"):
        web_eval.evaluate(run, "true", checkpoints, serve, max_steps=20)
    weights.write_bytes(b"different adapter")
    with pytest.raises(ValueError, match="inputs changed"):
        web_eval.evaluate(run, "true", checkpoints, serve)


@pytest.mark.parametrize("tasks", [
    ("v1.opendining-5", "v2.udriver-4"), ("udriver-4",), ("opendining-3000", "udriver-4"),
    ("opendining-5", "udriver-4", "v2.udriver-4"),
])
def test_rejects_nonofficial_or_unmatched_task_selections(tmp_path, tasks):
    with pytest.raises(ValueError, match="official"):
        web_eval.evaluate(tmp_path, "true", {"frozen": None}, None, tasks=tasks)


def test_rejects_raw_records_for_alternate_sites(tmp_path):
    run = tmp_path / "real-eval/frozen/episodes/v2.udriver-4"
    run.mkdir(parents=True)
    (run / "record.json").write_text(json.dumps(dict(task_id="v2.udriver-4", site="udriver",
        kind="real", version="v2", llm_judged=False, start_url="https://unrelated.example", status="success", reward=1)))
    with pytest.raises(ValueError, match="official"):
        web_eval._read_episode(tmp_path, "frozen", "v2.udriver-4", "")


def test_endpoint_failure_is_reported_for_all_planned_tasks(tmp_path):
    @contextmanager
    def failed(_):
        raise RuntimeError("policy could not load")
        yield

    scores = web_eval.evaluate(tmp_path, "true", {"frozen": None}, failed)
    assert len(scores["infrastructure_errors"]) == 2
    assert all("policy could not load" in row["error"] for row in scores["infrastructure_errors"])
    assert all(cell["scored"] == 0 for cell in scores["success_matrix"]["frozen"].values())
