import json
import subprocess
import sys



def cli(*args):
    return subprocess.run([sys.executable, "-m", "pdm", *map(str, args)],
                          text=True, capture_output=True)


def test_demo_train_evaluate_and_restore(tmp_path):
    run = tmp_path / "run"
    result = cli("demo", "--out", run)
    assert result.returncode == 0, result.stderr
    info = json.loads((run / "run.json").read_text())
    assert info["status"] == "complete"
    metrics = json.loads((run / "metrics.json").read_text())
    assert set(metrics) == {"frozen", "retrieval", "lora", "pdm", "pdm_uniform", "pdm_no_replay"}
    for name, entry in metrics.items():
        assert entry["accuracy_matrix"][0][1] is None
        assert len(entry["accuracy_matrix"][1]) == 2
        first, last = entry["stages"]
        assert first["replay_count"] == 0
        assert last["replay_count"] == (8 if name in ("pdm", "pdm_uniform") else 0)
    records = [json.loads(line) for line in (run / "manifest.jsonl").read_text().splitlines()]
    by_id = {r["sample_id"]: r for r in records}
    predictions = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 180
    assert all(by_id[p["sample_id"]]["split"] == "test" for p in predictions)
    sample = next(p for p in predictions if p["condition"] == "pdm" and p["stage"] == 2)
    result = cli("predict", "--run-dir", run, "--condition", "pdm", "--stage", 2,
                 "--sample-id", sample["sample_id"])
    assert result.returncode == 0, result.stderr
    restored = json.loads(result.stdout)
    for key in ("answer", "choice_scores", "evidence_ms"):
        assert restored[key] == sample[key]
    state = json.loads((run / "checkpoints/pdm/stage-2/state.json").read_text())
    assert all(by_id[i]["split"] == "train" and by_id[i]["stage"] == 1 for i in state["replay_ids"])
    assert state["adapter_before"] != state["adapter_after"]
    sensor = json.loads((run / "sensor_comparison.json").read_text())
    assert sensor["off"]["evidence_ms"] == [0, 1000]
    assert sensor["on"]["evidence_ms"] == [2000, 3000]
    assert (run / "report.html").is_file()
    assert cli("demo", "--out", run).returncode != 0
    # A restored model only needs its selected input's pixels.
    removed = next(r for r in records if r["split"] == "train")
    (run / removed["clip_path"]).unlink()
    assert cli("predict", "--run-dir", run, "--sample-id", sample["sample_id"]).returncode == 0
    with (run / "manifest.jsonl").open("a") as file:
        file.write("\n")
    assert cli("predict", "--run-dir", run, "--sample-id", sample["sample_id"]).returncode != 0


def test_selector_prefers_complementary_gradients():
    import torch
    from pdm.continual import gradient_order

    grads = torch.tensor([[2., 0.], [1., 0.], [0., 1.]])
    assert gradient_order(grads, 2) == [0, 2]
    assert gradient_order(torch.zeros(3, 2), 3) == [0, 1, 2]


def test_repeated_training_uses_seen_validation_tasks(tmp_path):
    run = tmp_path / "repeated"
    result = cli("demo", "--passes", 3, "--conditions", "pdm", "pdm_uniform", "--out", run)
    assert result.returncode == 0, result.stderr
    manifest = {r["sample_id"]: r for r in map(json.loads, (run / "manifest.jsonl").read_text().splitlines())}
    metrics = json.loads((run / "metrics.json").read_text())
    for name in ("pdm", "pdm_uniform"):
        previous = None
        for task in (1, 2):
            state = json.loads((run / f"checkpoints/{name}/stage-{task}/state.json").read_text())
            if previous is not None:
                assert state["adapter_before"] == previous["adapter_after"]
            previous = state
            history = state["training_history"]
            assert [row["pass"] for row in history] == [1, 2, 3]
            assert state["optimizer_steps"] == 24
            assert state["best_pass"] == min(history, key=lambda row: row["validation_loss"])["pass"]
            for row in history:
                assert set(row["validation_loss_by_task"]) == {str(t) for t in range(1, task + 1)}
            assert len(state["replay_ids"]) == (24 if task == 2 else 0)
            assert all(manifest[i]["split"] == "train" and manifest[i]["stage"] == 1
                       for i in state["replay_ids"])
            stage_metrics = metrics[name]["stages"][task - 1]
            assert stage_metrics["training_history"] == history
            assert stage_metrics["best_pass"] == state["best_pass"]


def test_update_restores_best_pass_when_later_validation_worsens():
    import torch

    from pdm.continual import update

    class BinaryScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def score(self, _item):
            return self.weight * self.weight.new_tensor([1., -1.])

    item = {"clip_path": "fixture", "question": "Which one?", "choices": ["A", "B"]}
    training = [dict(item, sample_id=f"train{i}", answer_index=0, stage=1) for i in range(4)]
    validation = [dict(item, sample_id="val", answer_index=1, stage=1)]
    config = {"steps": 1, "learning_rate": 0.1, "memory_capacity": 4}
    single, repeated = BinaryScorer(), BinaryScorer()
    update(single, "lora", training, [], config, 42, 1, validation=validation)
    _, result = update(repeated, "lora", training, [], dict(config, passes=3), 42, 1, validation=validation)
    assert result["optimizer_steps"] == 3
    assert result["best_pass"] == 1
    assert result["training_history"][-1]["validation_loss"] > result["training_history"][0]["validation_loss"]
    torch.testing.assert_close(repeated.weight, single.weight)


def test_grounded_replay_uses_previous_checkpoint_and_matched_windows(tmp_path):
    run = tmp_path / "grounded"
    methods = ("pdm", "pdm_random_window", "pdm_evidence")
    result = cli("demo", "--conditions", *methods, "--out", run)
    assert result.returncode == 0, result.stderr
    states = {}
    for name in methods:
        root = run / "checkpoints" / name
        first = json.loads((root / "stage-1/state.json").read_text())
        last = json.loads((root / "stage-2/state.json").read_text())
        memory = {r["sample_id"]: r for r in map(json.loads, (root / "stage-1/memory.jsonl").read_text().splitlines())}
        assert last["adapter_before"] == first["adapter_after"]
        assert last["optimizer_steps"] == 8
        if name != "pdm":
            assert first["window_extraction_count"] == len(memory)
            assert last["window_extraction_count"] == 0
            for key in last["replay_ids"]:
                assert memory[key]["source_stage"] == 1
                assert key.startswith("s1-train-")
        states[name] = first, last, memory
    full, random, evidence = (states[n] for n in methods)
    assert full[1]["replay_ids"] == random[1]["replay_ids"] == evidence[1]["replay_ids"]
    assert [s[0]["shared_training_from"] for s in (full, random, evidence)] == [None, "pdm", "pdm"]
    assert full[0]["adapter_after"] == random[0]["adapter_after"] == evidence[0]["adapter_after"]
    assert random[1]["adapter_after"] != evidence[1]["adapter_after"]
    for key in evidence[1]["replay_ids"]:
        start, end = evidence[2][key]["replay_window_ms"]
        a, b = random[2][key]["replay_window_ms"]
        assert 0 <= a < b <= 4000
        assert b - a == end - start
    key = evidence[1]["replay_ids"][0]
    teacher = cli("predict", "--run-dir", run, "--condition", methods[-1], "--stage", 1, "--sample-id", key)
    assert teacher.returncode == 0, teacher.stderr
    assert json.loads(teacher.stdout)["evidence_ms"] == evidence[2][key]["replay_window_ms"]
