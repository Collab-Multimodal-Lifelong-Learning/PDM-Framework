import errno
import json
import os
from html import unescape
from html.parser import HTMLParser
import shutil
import subprocess
from urllib.parse import unquote

import pytest

from pdm.report import write_report


class EmbeddedResults(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reading = False
        self.results = None

    def handle_starttag(self, tag, attrs):
        self.reading = tag == "script" and dict(attrs).get("type") == "application/json"

    def handle_endtag(self, tag):
        if tag == "script":
            self.reading = False

    def handle_data(self, data):
        if self.reading:
            self.results = json.loads(data)


def make_clip(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    f"color={color}:size=64x64:rate=4:duration=4", "-pix_fmt", "yuv420p",
                    str(path)], check=True)
    return path


@pytest.fixture
def report(tmp_path):
    out = tmp_path / "run"
    media = make_clip(out / "clips" / "sun #2?.mp4", "red")
    record = {
        "sample_id": "q:7", "clip_id": "sun", "clip_path": media,
        "split": "test", "stage": 1, "domain": "first_color",
        "question": "Which </script><i>color</i>?",
        "choices": ["red", "green", "blue", "yellow", "magenta"], "answer_index": 1,
    }
    pred = {
        "sample_id": "q:7", "condition": "frozen", "stage": 1, "task_stage": 1,
        "choice_index": 0, "answer": "red", "evidence_ms": [0, 1000],
        "choice_scores": [2, 1, 0, 0, 0], "window_scores": [],
    }
    later = {**pred, "stage": 2}
    trained = {**later, "condition": "pdm", "choice_index": 1, "answer": "green",
               "choice_scores": [1, 2, 0, 0, 0], "evidence_ms": [2001, 3000]}
    write_report(out, [record, {**record, "sample_id": "training-only", "split": "train"}],
                 [pred, later, trained], {"frozen": {"accuracy_matrix": [[0.5, None], [0.25, 1]],
                 "stages": [{"stage": 2, "seen_average": 0.625, "forgetting": 0.25,
                             "adaptation_gain": 0, "mean_evidence_iou": 0.375, "evidence_count": 12,
                             "replay_count": 0, "mean_latency_ms": 3,
                             "peak_cuda_bytes": None, "p95_latency_ms": 8, "update_seconds": 0.2}]}},
                 {"clip_path": media, "off": pred, "on": trained},
                 {"accuracy_midpoint": 1, "accuracy_vlm_rlaif": 0.5, "disagreements": 1,
                  "count": 2, "disagreement_ids": ["q:7"], "sample_ids": ["q:7", "q:9"]})
    return out / "report.html"


def test_offline_report_preserves_untrusted_text_and_rebases_local_media(report):
    parser = EmbeddedResults()
    parser.feed(report.read_text())
    record, = parser.results["records"]
    assert record["question"] == "Which </script><i>color</i>?"
    assert record["clip_path"] == "clips/sun%20%232%3F.mp4"
    assert parser.results["sensor"]["clip_path"] == record["clip_path"]
    assert str(report.parent.parent) not in report.read_text()


@pytest.mark.parametrize("cross_device", [False, True])
def test_moved_run_keeps_distinct_external_clips_and_shares_repeated_sources(tmp_path, monkeypatch, cross_device):
    originals = tmp_path / "originals"
    day = make_clip(originals / "day" / "view.mp4", "blue")
    night = make_clip(originals / "night" / "view.mp4", "black")
    if cross_device:
        def unavailable_link(*args, **kwargs):
            raise OSError(errno.EXDEV, "Cross-device link")
        monkeypatch.setattr(os, "link", unavailable_link)

    records = [{"sample_id": str(i), "clip_path": path}
               for i, path in enumerate((day, day, night))]
    out = tmp_path / "run"
    write_report(out, records, [{"sample_id": row["sample_id"]} for row in records],
                 {}, {"clip_path": night, "on": {}, "off": {}}, None)
    expected = [path.read_bytes() for path in (day, day, night)]
    moved = out.rename(tmp_path / "relocated")
    shutil.rmtree(originals)
    parser = EmbeddedResults()
    parser.feed((moved / "report.html").read_text())
    paths = [moved / unquote(r["clip_path"]) for r in parser.results["records"]]
    assert paths[0] == paths[1] and paths[0] != paths[2]
    assert all(path.is_relative_to(moved / "media") for path in paths)
    assert [path.read_bytes() for path in paths] == expected
    assert parser.results["sensor"]["clip_path"] == parser.results["records"][2]["clip_path"]
    assert len([path for path in (moved / "media").rglob("*") if path.is_file()]) == 2


def test_offline_browser_switches_stage_and_method(report, tmp_path):
    chrome = next((path for name in ("google-chrome", "chromium", "chromium-browser")
                   if (path := shutil.which(name))), None)
    if chrome is None:
        pytest.skip("Chrome is unavailable for the optional offline browser check")
    # Exercise the exported document via file://, including its inline event handlers.
    interaction = """<script>
      const select = (id, value) => {
        const el = document.getElementById(id);
        el.value = value;
        el.dispatchEvent(new Event('change'));
      };
      select('stage', '1');
      const before = document.querySelector('#answers tbody').innerText;
      select('stage', '2');
      select('method', 'pdm');
      for (const summary of document.querySelectorAll('#sampler details summary')) summary.click();
      const observed = [before, document.querySelector('#answers tbody').innerText,
                        document.getElementById('play-evidence').textContent,
                        document.getElementById('question-title').textContent,
                        document.querySelector('#choices').innerText,
                        document.getElementById('results').innerText,
                        document.getElementById('paired').innerText];
      document.body.replaceChildren(document.createTextNode(JSON.stringify(observed)));
    </script>"""
    probe = report.with_name("probe.html")
    probe.write_text(report.read_text().replace("</body>", interaction + "</body>"))
    result = subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu",
                             "--disable-dev-shm-usage", "--disable-background-networking",
                             f"--user-data-dir={tmp_path / 'browser'}", "--dump-dom",
                             probe.as_uri()], capture_output=True, text=True, timeout=30, check=True)
    body = unescape(result.stdout.split("<body>", 1)[1].split("</body>", 1)[0])
    before, after, play, question, choices, results, paired = json.loads(body)
    assert "red" in before and "green" not in before
    assert "red" in after and "green" in after
    assert "2.001" in play and "3.00" in play
    assert question == "Which </script><i>color</i>?"
    assert all(label in choices for label in ("red", "green", "Label", "Prediction"))
    assert "62.5%" in results and "25.0%" in results
    assert "37.5%" in results and "12" in results
    assert all(text in paired for text in ("green", "q:7", "q:9"))
