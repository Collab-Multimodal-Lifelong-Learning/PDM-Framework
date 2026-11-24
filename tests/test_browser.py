import json
from types import SimpleNamespace

import pytest

from pdm import browser


@pytest.mark.parametrize(("source", "expected"), [
    ('```action\nfill("12", "tea")\n```', "fill('12', 'tea')"),
    ("noop()", "noop(1000)"),
    ("scroll(0,-300)", "scroll(0, -300)"),
    ("go_back()", "go_back()"),
])
def test_parse_one_command(source, expected):
    assert browser.parse_action(source) == expected


@pytest.mark.parametrize("source", [
    "click('3')\nclick('4')", "click('3'); noop()", "click(id='3')",
    "click(x)", "click(get_id())", "click('3.0')", "noop(True)", "noop(60000)",
    "upload_file('3','photo')", "goto('other')", "agent.click('3')", "click('3', '4')",
    "Click the checkbox", "scroll(0, '20')",
])
def test_parse_rejects_unsupported_commands_and_arguments(source):
    with pytest.raises(ValueError, match="supported browser command"):
        browser.parse_action(source)


def test_budget_exhaustion_runs_official_scorer_and_retains_model_evidence(monkeypatch, tmp_path):
    state = {"finalstate": {"reservations": []}}
    scorer_calls = []

    def evaluate(actual_state, answer):
        scorer_calls.append((actual_state, answer))
        return 0.0, True, "", {"results": [[False, {"output": "FAILURE", "return_code": 0}]]}

    raw = {"goal": "book lunch", "url": "https://example.invalid/", "last_action_error": ""}
    task = SimpleNamespace(url=raw["url"], get_finish_json=lambda timeout: state,
                           evaluator=SimpleNamespace(evaluate=evaluate))
    actions = []

    def step(action):
        actions.append(action)
        return raw, 0.0, False, False, {}

    env = SimpleNamespace(task=task, step=step, reset=lambda seed: (raw, {}), close=lambda: None)
    replies = iter(["unsupported", "noop(0)"])
    monkeypatch.setattr(browser, "make_env", lambda *args: env)
    monkeypatch.setattr(browser, "page_view", lambda *args: ("", "sample prompt", b"jpg"))
    monkeypatch.setattr(browser, "request_action", lambda *args: next(replies))
    result = browser.run_episode("v2.opendining-5", endpoint="http://localhost/v1",
                                 out=tmp_path, max_steps=2)
    assert (result["status"], result["reward"], result["steps"], result["end"]) == ("fail", 0.0, 2, "max_steps")
    assert actions == ["noop(0)"]
    assert scorer_calls == [(state, "")]
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert trace[0]["error"] and trace[1]["action"] == "noop(0)"
    assert json.loads((tmp_path / "state.json").read_text()) == state
    assert (tmp_path / "02.jpg").read_bytes() == b"jpg"
    assert (tmp_path / "score.json").is_file()

    state.clear()
    monkeypatch.setattr(browser, "request_action", lambda *args: "send_msg_to_user('finished')")
    empty = browser.run_episode("v2.opendining-5", endpoint="http://localhost/v1", out=tmp_path / "empty")
    assert (empty["status"], empty["reward"]) == ("error", None)
    assert "no final state" in empty["error"]
    assert len(scorer_calls) == 1


def test_scorer_diagnostics_distinguish_a_broken_script_from_a_failed_task():
    task = SimpleNamespace(evaluator=SimpleNamespace(evaluate=lambda *_:
                           (0.0, True, "", {"results": [[False, {"return_code": 1, "stderr": "broken"}]]})))
    with pytest.raises(RuntimeError, match="SDK scorer error"):
        browser.score_official(task, {}, "")
