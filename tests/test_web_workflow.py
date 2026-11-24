import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image


def test_web_training_uses_prior_domain_and_shared_first_adapter(tmp_path):
    from pdm.web import score_demos, train

    rows = []
    for domain in ("opendining", "udriver"):
        for index in range(3):
            name = f"{domain}-{index}.jpg"
            Image.new("RGB", (3, 2), (index, 0, 0)).save(tmp_path / name)
            rows.append(dict(domain=domain, episode="train-" + domain, split="train", system="act",
                             prompt=str(index), image=name, action=f"click('{index}')"))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(map(json.dumps, rows)))

    class Policy:
        def __init__(self):
            self.n = 0

        def update(self, batch):
            assert len(batch) == 2
            assert all(Path(x["image"]).is_file() for x in batch)
            self.n += 1
            return 0.5

        def save(self, path):
            path.write_text(str(self.n))

        def load(self, path):
            self.n = int(path.read_text())

        def action_nll(self, rows):
            return float(len(rows) + self.n)

    report = train(manifest, tmp_path / "out", Policy, seed=42, passes=2, capacity=2)
    assert report["status"] == "complete"
    assert report["checkpoints"]["sequential-opendining"] == report["checkpoints"]["replay-opendining"]
    s = report["updates"]["sequential-udriver"]
    r = report["updates"]["replay-udriver"]
    assert len(s) == len(r) == 6
    assert all(t['batch'][1].startswith('train-opendining:') for t in r)
    assert all(t['batch'][1].startswith('train-udriver:') for t in s)
    assert [t['batch'][0] for t in s] == [t['batch'][0] for t in r]
    score_demos(tmp_path / "out", manifest, Policy)
    scores = json.loads((tmp_path / "out" / "demonstration-fit.json").read_text())
    assert scores["frozen"]["udriver"] == 3.0
    assert scores["replay-udriver"]["udriver"] == 15.0
    Image.new("RGB", (3, 2), "red").save(tmp_path / rows[0]["image"])
    with pytest.raises(ValueError, match="differ from the original"):
        score_demos(tmp_path / "out", manifest, Policy)
    rows[0]["split"] = "test"
    manifest.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError):
        train(manifest, tmp_path / "other", Policy, seed=42, passes=1, capacity=2)


def test_chat_transport_matches_training_prompt_and_image():
    from pdm.web import decode_observation

    buf = io.BytesIO()
    Image.new("RGB", (4, 6)).save(buf, "JPEG")
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    user = [{"type": "text", "text": "tree"}, {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "history"}]
    system, prompt, img = decode_observation([{"role":"system","content":"rules"},{"role":"user","content":user}])
    assert (system, prompt, img.size) == ("rules", "tree\n\nhistory", (4, 6))
    with pytest.raises(ValueError):
        decode_observation([{"role":"user","content":user}])
