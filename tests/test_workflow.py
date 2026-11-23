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
