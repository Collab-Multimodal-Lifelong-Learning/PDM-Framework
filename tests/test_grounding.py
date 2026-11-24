import json

import pytest

from test_workflow import cli


def test_grounding_uses_matched_rows_and_tracks_prior_task_loss():
    from pdm.grounding import score_grounding

    def question(key, stage):
        return {"sample_id": key, "split": "test", "stage": stage, "answer_index": 2}
    questions = [question("a", 1), question("b", 2), question("no_span", 1)]
    spans = {"a": [[0, 1000], [4000, 5000]], "b": [[0, 500]]}
    def prediction(key, stage, task, window, choice=2):
        return dict(sample_id=key, stage=stage, task_stage=task, condition="pdm",
                    choice_index=choice, answer_index=2, evidence_ms=window)
    rows = [prediction("a", 1, 1, [4000, 6000]), prediction("no_span", 1, 1, [0, 500]),
            prediction("a", 2, 1, [2000, 3000]), prediction("b", 2, 2, [0, 500], 0),
            prediction("no_span", 2, 1, [0, 500])]
    result = score_grounding(questions, rows, spans, ["pdm"])
    assert result["unannotated_test_ids"] == ["no_span"]
    first, last = result["conditions"]["pdm"]
    assert first["overall"]["acc_gqa"] == 1
    assert first["overall"]["mean_iou"] == first["overall"]["mean_iop"] == 0.5
    assert last["overall"]["count"] == 2
    assert last["overall"]["acc_gqa"] == 0
    assert last["overall"]["answer_accuracy"] == last["overall"]["mean_iop"] == 0.5
    assert last["previous_task_forgetting"]["acc_gqa"] == 1.0
    assert last["previous_task_forgetting"]["answer_accuracy"] == 0
    with pytest.raises(ValueError, match="missing"):
        score_grounding(questions, rows[:-1], spans, ["pdm"])
    with pytest.raises(ValueError, match="duplicate"):
        score_grounding(questions, rows + [rows[0]], spans, ["pdm"])


def test_cpu_grounding_cli_scores_existing_predictions(tmp_path):
    run = tmp_path / "cpu"
    training = cli("demo", "--out", run, "--conditions", "pdm_evidence")
    assert training.returncode == 0, training.stderr
    result = cli("evaluate-grounding", "--run-dir", run)
    assert result.returncode == 0, result.stderr
    report = json.loads((run / "grounding.json").read_text())
    assert report["source"] == "smoke_fixture"
    assert report["conditions"]["pdm_evidence"][-1]["overall"]["count"] == 20
    assert len(report["samples"]) == 30


def test_nextgqa_joins_official_questions_and_converts_multiple_spans(tmp_path, monkeypatch):
    import hashlib

    from pdm import grounding

    (tmp_path / "test.csv").write_text(
        "video_id,qid,type,question,a0,a1,a2,a3,a4,answer\n7,1,TC,What?,a,b,c,d,e,b\n"
    )
    (tmp_path / "gsub_test.json").write_text(json.dumps({"7": {"location": {"1": [[0.2, 0.7], [1.3, 2]]}}}))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    monkeypatch.setattr(grounding, "HASHES", hashes)
    row = {"sample_id": "7:1:TC", "clip_id": "7", "question": "What?", "choices": list("abcde"), "answer_index": 1}
    assert grounding.nextgqa_references(tmp_path, [row, dict(row, sample_id="7:1:CW")]) == {
        "7:1:TC": [[200, 700], [1300, 2000]]}
    with pytest.raises(ValueError, match="differ"):
        grounding.nextgqa_references(tmp_path, [dict(row, choices=list("bacde"))])
    (tmp_path / "test.csv").write_text("damaged")
    with pytest.raises(ValueError, match="checksum"):
        grounding.nextgqa_references(tmp_path, [row])
