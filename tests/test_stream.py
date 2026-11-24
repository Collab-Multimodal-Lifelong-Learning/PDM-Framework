import json
from pathlib import Path

from test_workflow import cli


def test_stream_updates_and_both_inference_paths(tmp_path):
    out = tmp_path / "stream"
    result = cli("demo", "--memory-capacity", 4, "--conditions", "stream_lora", "pdm_stream", "--out", out)
    assert result.returncode == 0, result.stderr
    records = [json.loads(s) for s in (out / "manifest.jsonl").read_text().splitlines()]
    ordered = [r["sample_id"] for r in records if r["split"] == "train"]
    seen = []
    for stage in (1, 2):
        root = out / "checkpoints/pdm_stream" / f"stage-{stage}"
        state = json.loads((root / "state.json").read_text())
        assert state["adapter_before"] != state["adapter_after"]
        for step in state["online_history"]:
            if seen:
                assert step["replay_id"] in seen
                assert step["distill_mse"] >= 0
            if step["ranking_loss"] is not None:
                assert step["ranking_loss"] >= 0 and step["negative_id"] in seen
            seen.append(step["sample_id"])
        assert state["stream_seen"] == len(seen)
        entries = list(map(json.loads, (root / "memory.jsonl").read_text().splitlines()))
        assert any(h["ranking_loss"] is not None for h in state["online_history"])
        assert len(entries) == 4
        assert all(len(r["teacher_logits"]) == 5 for r in entries)
    assert seen == ordered
    queries = json.loads((out / "retrieval.json").read_text())["pdm_stream"][-1]["queries"]
    assert len(queries) == 20
    test_ids = {r["clip_id"] for r in records if r["split"] == "test"}
    for q in queries:
        assert len(q["ranking"]) == 5
        assert {r["clip_id"] for r in q["ranking"]} <= test_ids
    sample = next(r for r in records if r["split"] == "test")
    clips = {r["clip_id"]: out / r["clip_path"] for r in records}
    q = next(q for q in queries if q["sample_id"] == sample["sample_id"])
    args = ["retrieve-video", "--run-dir", out, "--question", sample["question"],
            "--query-answer", sample["choices"][sample["answer_index"]]]
    for c in sample["choices"]:
        args.extend(["--choice", c])
    for r in reversed(q["ranking"]):
        args.extend(["--clip", clips[r["clip_id"]]])
    restored = cli(*args)
    assert restored.returncode == 0, restored.stderr
    assert [Path(r["clip_id"]).resolve() for r in json.loads(restored.stdout)] == [
        clips[r["clip_id"]].resolve() for r in q["ranking"]]
    answer = cli("predict", "--run-dir", out, "--condition", "pdm_stream", "--sample-id", sample["sample_id"])
    assert answer.returncode == 0 and json.loads(answer.stdout)["evidence_ms"]


def test_online_learner_needs_no_task_fields(tmp_path):
    from pdm.data import generate_fixture, inference_input
    from pdm.models import build_model
    from pdm.stream import OnlineLearner

    rows = generate_fixture(tmp_path)
    rows = [next(r for r in rows if r["stage"] == t and r["split"] == "train") for t in (2, 1)]
    learner = OnlineLearner(build_model("smoke"), "pdm_stream", 0.001, 1, 5)
    for r in rows:
        learner.observe(inference_input(r), r["answer_index"])
    assert learner.seen == 2 and len(learner.memory) == 1
