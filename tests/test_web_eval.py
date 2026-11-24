import json
import sys
import types
from contextlib import contextmanager

import pytest

from pdm import browser_cli, web_eval


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


@contextmanager
def serve(_):
    yield "http://127.0.0.1:9000/v1"


def test_cli_conditions_resume_and_checkpoint_identity(tmp_path):
    # A separate interpreter executable follows the real parent / worker file contract.
    executable = tmp_path / "browser-python"
    executable.write_text(f"#!{sys.executable}\n" + """
import json, sys
from pathlib import Path
assert sys.argv[1:4] == ["-m", "pdm.browser_cli", "episode"]
task = sys.argv[sys.argv.index("--task") + 1]
site = task.split(".")[1].split("-")[0]
out = Path(sys.argv[sys.argv.index("--out") + 1])
assert int(sys.argv[sys.argv.index("--max-steps") + 1]) == 25
assert sys.argv[sys.argv.index("--endpoint") + 1].endswith("/v1")
calls = Path(__file__).with_name("calls")
with calls.open("a") as log:
    print(out.parent.parent.name, task, file=log)
if site == "udriver" and not Path(__file__).with_name("ready").exists():
    print("network unavailable")
    sys.exit(2)
urls = {"opendining": "https://real-opendining-fdcw98zrb-real-sites.vercel.app/",
        "udriver": "https://real-udriver-3ufk0q1v4-real-sites.vercel.app/"}
(out / "record.json").write_text(json.dumps(dict(task_id=task, site=site, start_url=urls[site],
    status="fail", reward=0, steps=2, end="done", environment={"agisdk":"0.3.5"})))
""")
    executable.chmod(0o755)
    out = tmp_path / "evaluation"
    scores = web_eval.evaluate(out, executable, {"frozen": None}, serve)
    [error] = scores["infrastructure_errors"]
    assert error["task_id"] == "v2.udriver-4"
    assert "network unavailable" in (out / error["record"]).with_name("episode.log").read_text()
    assert error["reward"] is None
    (tmp_path / "ready").touch()
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"first adapter")
    checkpoints = {"frozen": None, "sequential-opendining": weights}
    scores = web_eval.evaluate(out, executable, checkpoints, serve)
    assert (tmp_path / "calls").read_text().splitlines() == [
        "frozen v2.opendining-5", "frozen v2.udriver-4", "frozen v2.udriver-4",
        "sequential-opendining v2.opendining-5", "sequential-opendining v2.udriver-4",
    ]
    assert not scores["infrastructure_errors"]
    assert scores == json.loads((out / "web-summary.json").read_text())
    with pytest.raises(ValueError, match="inputs changed"):
        web_eval.evaluate(out, executable, checkpoints, serve, max_steps=20)
    weights.write_bytes(b"different adapter")
    with pytest.raises(ValueError, match="inputs changed"):
        web_eval.evaluate(out, executable, checkpoints, serve)


@pytest.mark.parametrize("tasks", [
    ("v1.opendining-5", "v2.udriver-4"), ("udriver-4",), ("opendining-3000", "udriver-4"),
    ("opendining-5", "udriver-4", "v2.udriver-4"), ("train-dining", "v2.udriver-4"),
])
def test_rejects_nonofficial_or_unmatched_task_selections(tmp_path, tasks):
    with pytest.raises(ValueError, match="official"):
        web_eval.evaluate(tmp_path, sys.executable, {"frozen": None}, None, tasks=tasks)


def test_rejects_raw_records_for_alternate_sites(tmp_path):
    run = tmp_path / "checkpoints/frozen/episodes/v2.udriver-4"
    run.mkdir(parents=True)
    (run / "record.json").write_text(json.dumps(dict(task_id="v2.udriver-4", site="udriver",
        start_url="https://unrelated.example", status="success", reward=1, steps=2, end="done")))
    with pytest.raises(ValueError, match="official"):
        web_eval._read_episode(tmp_path, "frozen", "v2.udriver-4")


def test_endpoint_failure_is_reported_for_all_planned_tasks(tmp_path):
    @contextmanager
    def failed(_):
        raise RuntimeError("policy could not load")
        yield

    scores = web_eval.evaluate(tmp_path, sys.executable, {"frozen": None}, failed)
    assert len(scores["infrastructure_errors"]) == 2
    assert all("policy could not load" in row["error"] for row in scores["infrastructure_errors"])
    assert all(cell["scored"] == 0 for cell in scores["success_matrix"]["frozen"].values())


def test_timeout_record_is_unscored(tmp_path, monkeypatch):
    executable = tmp_path / "slow"
    executable.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(10)\n")
    executable.chmod(0o755)
    monkeypatch.setattr(web_eval, "EPISODE_TIMEOUT", 0.1)
    scores = web_eval.evaluate(tmp_path / "evaluation", executable, {"frozen": None}, serve)
    assert len(scores["infrastructure_errors"]) == 2
    assert all(row["reward"] is None and "timeout" in row["error"] for row in scores["infrastructure_errors"])


def test_custom_request_writes_model_answer(tmp_path, monkeypatch):
    endpoint = "http://localhost:9000/v1"
    module = types.ModuleType("pdm.browser")

    def run(task_id, *, site, goal, endpoint, out, max_steps):
        assert site == "opendining" and goal == "Check my reservation."
        assert endpoint.endswith("/v1") and max_steps == 4
        return dict(task_id=task_id, site=site, status="completed", reward=None,
                    end="message", steps=2, answer="Your reservation is at noon.")

    module.run_episode = run
    monkeypatch.setitem(sys.modules, "pdm.browser", module)
    monkeypatch.setattr(browser_cli, "_environment", lambda: {"agisdk": "0.3.5"})
    monkeypatch.setattr(browser_cli, "_check_endpoint", lambda url: None)
    rc = browser_cli.main(["ask", "--goal", "Check my reservation.", "--site", "opendining",
                           "--endpoint", endpoint, "--max-steps", "4", "--out", str(tmp_path)])
    assert rc == 0
    result = json.loads((tmp_path / "record.json").read_text())
    assert result["answer"] == "Your reservation is at noon." and result["reward"] is None


def test_endpoint_non_200(tmp_path):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_error(503)

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.handle_request)
        thread.start()
        try:
            rc = browser_cli.main(["episode", "--task", "v2.udriver-4", "--endpoint",
                f"http://127.0.0.1:{server.server_port}/v1", "--out", str(tmp_path)])
        finally:
            thread.join(timeout=5)
            assert not thread.is_alive()
    record = json.loads((tmp_path / "record.json").read_text())
    assert rc and record["status"] == "error" and record["reward"] is None
    assert "503" in record["error"]
