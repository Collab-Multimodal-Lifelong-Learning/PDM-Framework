"""Video adapter contracts, including the pinned VLM-RLAIF sampler parity cases."""

import av
import numpy as np
import pytest
import torch

from pdm.models import build_model, predict, sample_indices, video_duration_ms


def make_clip(path, colors):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=4)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0"}
        for color in colors:
            pixels = np.full((64, 64, 3), color, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return {"sample_id": path.stem, "clip_path": str(path),
            "question": "Which color appeared first?",
            "choices": ["red", "green", "blue", "yellow", "magenta"]}


# load_frames in Evaluation/infer_utils.py at VLM-RLAIF 610cac72c16ae3862b0d83a0dc3708d73a42a4cc
@pytest.mark.parametrize(("n", "expected"), [
    (1, [0] * 8), (3, [0, 0, 0, 0, 1, 1, 1, 2]),
    (8, list(range(8))), (16, [0, 2, 4, 6, 8, 10, 12, 15]),
])
def test_source_frame_indices(n, expected):
    assert sample_indices(np.arange(n), (0, n), "vlm_rlaif") == expected


def test_midpoint_ties_earlier_and_excludes_outside_pts():
    assert sample_indices(np.arange(21), (2, 18), "midpoint") == list(range(3, 18, 2))
    pts = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    assert sample_indices(pts, (0, 16), "midpoint") == list(range(8))
    assert sample_indices(pts, (5, 7), "midpoint") == [3] * 8


def test_smoke_masks_other_half_and_updates_only_adapter(tmp_path):
    first, second = (255, 0, 0), (0, 0, 255)
    red_blue = make_clip(tmp_path / "red-blue.mp4", [first] * 8 + [second] * 8)
    red = make_clip(tmp_path / "red.mp4", [first] * 16)
    model = build_model("smoke", 42, "cpu")
    a = model.score(red_blue)
    assert a.shape == (5,)
    torch.testing.assert_close(a, model.score(red))
    red_blue["question"] = red["question"] = "Which color appeared last?"
    assert not torch.allclose(model.score(red_blue), model.score(red))
    replica = build_model("smoke", 42, "cpu")
    torch.testing.assert_close(model.score(red_blue), replica.score(red_blue), rtol=0, atol=0)
    prior = {k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad}
    assert sum(p.numel() for p in model.selector_parameters()) == 2 * (6 + 5)
    optimizer = torch.optim.AdamW(model.selector_parameters(), lr=0.01, weight_decay=0)
    torch.nn.functional.cross_entropy(a.unsqueeze(0), torch.tensor([0])).backward()
    optimizer.step()
    assert not torch.allclose(model.score(red_blue), replica.score(red_blue))
    for k, p in model.named_parameters():
        if k in prior:
            torch.testing.assert_close(p, prior[k], atol=0, rtol=0)


def test_retrieval_bonus_normalizes_only_matching_verified_answers(tmp_path):
    item = make_clip(tmp_path / "context.mp4", [(0, 150, 40)] * 16)
    model = build_model("smoke", 42, "cpu")
    ctx = [{"choices": ["blue", "yellow", "pink"], "answer_index": i} for i in (0, 1, 0, 2)]
    torch.testing.assert_close(model.score(item, context=ctx) - model.score(item),
                               torch.tensor([0, 0, 2/3, 1/3, 0]))


def test_sensor_changes_only_evidence_and_earliest_tie_wins(tmp_path):
    item = make_clip(tmp_path / "sensor.mp4", [(30, 160, 200)] * 16)
    assert video_duration_ms(item["clip_path"]) == 4000
    model = build_model("smoke", 42, "cpu")
    off = predict(model, item)
    sensor = tmp_path / "trace.csv"
    sensor.write_text("timestamp_ms,value\n0,0\n1000,0\n2500,10\n3000,0\n")
    on = predict(model, dict(item, sensor_path=str(sensor)))
    assert off["choice_scores"] == on["choice_scores"]
    assert off["evidence_ms"] == [0, 1000] and off["evidence_source"] == "video_score"
    assert on["evidence_ms"] == [2000, 3000] and on["evidence_source"] == "video_sensor_score"
    assert [w["sensor_score"] for w in on["window_scores"]] == [0, 0, 1, 0]


def test_sparse_clip_scores_only_nonempty_quarters_and_breaks_choice_ties(tmp_path):
    item = make_clip(tmp_path / "single-frame.mp4", [(20, 50, 90)])
    model = build_model("smoke", 17, "cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    sensor = tmp_path / "constant.csv"
    sensor.write_text("timestamp_ms,value\n0,2\n249,2\n")
    outcome = predict(model, dict(item, sensor_path=sensor))
    assert outcome["choice_index"] == 0 and outcome["answer"] == item["choices"][0]
    assert [w["evidence_ms"] for w in outcome["window_scores"]] == [[0, 62]]
    assert outcome["window_scores"][0]["sensor_score"] == 0


def test_midpoint_tie_on_fractional_millisecond_pts():
    # These PTS are 1/1500 and 2/1500 seconds into a variable-rate stream.
    assert sample_indices([2/3, 4/3], (0, 16), "midpoint")[0] == 0


def test_smoke_logits_follow_displayed_choice_order(tmp_path):
    item = make_clip(tmp_path / "ordered.mp4", [(190, 80, 20)] * 16)
    model = build_model("smoke", 42, "cpu")
    context = [{"choices": ["red"], "answer_index": 0}]
    canonical = model.score(item, context=context)
    perm = [2, 0, 4, 1, 3]
    reordered = dict(item, choices=[item["choices"][i] for i in perm])
    torch.testing.assert_close(model.score(reordered, context=context), canonical[perm])
    with pytest.raises(ValueError, match="smoke choices"):
        model.score(dict(item, choices=["orange", "green", "blue", "yellow", "magenta"]))


def test_clearing_video_cache_starts_a_cold_decode(tmp_path, monkeypatch):
    import pdm.models as adapters
    item = make_clip(tmp_path / "cached.mp4", [(40, 90, 120)] * 16)
    model = build_model("smoke", 42, "cpu")
    readers = []
    av_open = av.open

    def counted_open(path, *args, **kwargs):
        readers.append(path)
        return av_open(path, *args, **kwargs)

    monkeypatch.setattr(av, "open", counted_open)

    def score_views():
        model.score(item)
        adapters._sampled_rgb(adapters._file_key(item["clip_path"]), "midpoint", None)

    score_views()
    initial_decodes = len(readers)
    assert initial_decodes > 0
    score_views()
    assert len(readers) == initial_decodes
    adapters.clear_video_cache()
    score_views()
    assert len(readers) == 2 * initial_decodes


def test_known_processor_warning_is_shown_once_while_other_warnings_survive(caplog):
    import logging
    from pdm.models import _QwenVideoWarnings
    logger = logging.getLogger("test.pdm.qwen")
    warning_filter = _QwenVideoWarnings()
    logger.addFilter(warning_filter)
    try:
        logger.warning("Unused or unrecognized kwargs: fps, return_tensors.")
        logger.warning("Unused or unrecognized kwargs: return_tensors, fps.")
        logger.warning("Unused or unrecognized kwargs: max_pixels.")
        logger.warning("Unused or unrecognized kwargs: max_pixels.")
    finally:
        logger.removeFilter(warning_filter)
    assert caplog.messages == ["Unused or unrecognized kwargs: fps, return_tensors.",
                               "Unused or unrecognized kwargs: max_pixels.",
                               "Unused or unrecognized kwargs: max_pixels."]
