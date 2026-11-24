# PDM Framework

[한국어 요약](README_ko.md) · [Continual VideoQA experiments](VIDEOQA.md)

PDM (Pre-trained, Data, Memory) teaches a web agent to make restaurant reservations, then to book rides. Before training and after each service is learned, REAL-Eval runs the agent on a restaurant request and a ride request it did not train on. A booking counts when the official REAL v2 evaluator accepts the simulated site's final state. This is a service-oriented application of research project 2022-0-00113. The video-language application and its existing checkpoints, commands and benchmarks are documented separately.

## What is trained

| Component | Implementation |
| --- | --- |
| Pre-trained model | Qwen2.5-VL-3B-Instruct reads the screenshot, user instruction, accessibility tree and action history. Rank-8 LoRA weights on the language decoder's query/value projections learn to generate one BrowserGym action. [Model](pdm/web_model.py) |
| Data | A scripted demonstrator completes two custom OpenDining bookings and two custom UDriver trips. The collector checks the browser outcome and writes the observation *before* each executed action. It publishes a manifest only after all four trajectories pass. [Collection](pdm/web_collect.py) |
| Memory | Reservoir sampling retains up to eight distinct steps from the restaurant demonstrations. During ride training, the replay condition pairs each current step with one stored restaurant step. Sequential LoRA pairs it with another ride step. Both conditions receive the same current-step order, example count and optimizer-step budget. [Training](pdm/web.py) |
| Evaluation | The same browser policy endpoint serves the frozen backbone and the saved adapters. The official v2 script scores each held-out episode. Results include per-site success, paired change in restaurant success after ride training, and infrastructure errors. [Evaluation](pdm/web_eval.py) · [REAL benchmark](https://www.realevals.xyz/) |

The target is the demonstrator's action tokens and EOS. Training minimizes cross-entropy on those tokens. A single restaurant adapter seeds both ride conditions, each with a fresh AdamW optimizer. This experiment uses known service boundaries. Web action generation uses separate weights from the five-choice VideoQA model.

## Observed results

Seed 42 used 10 restaurant actions and 29 ride actions. Each branch received 20 restaurant updates and 58 ride updates with two examples per update. The two restaurant rows share weights and ran in separate browsers. The booking columns count official held-out tasks completed. The loss columns measure mean action-token cross-entropy on the original demonstrations, averaged equally over actions.

| Checkpoint | Restaurant bookings | Ride bookings | Restaurant action loss | Ride action loss |
| --- | ---: | ---: | ---: | ---: |
| Frozen | 0/1 | 0/1 | 0.496 | 0.484 |
| After restaurant training (sequential) | 0/1 | 0/1 | 0.204 | 0.406 |
| After restaurant training (replay) | 0/1 | 0/1 | 0.204 | 0.406 |
| After ride training (sequential) | 0/1 | 0/1 | 0.186 | 0.111 |
| After ride training (replay) | 0/1 | 0/1 | 0.075 | 0.143 |

Across the two official tasks, all ten episodes received scores and none completed the booking. In a separate fresh browser, the final replay adapter repeated the first training instruction and completed a Vintage Bites reservation for two at 12:30 PM in five actions. The simulated site's booking record confirmed the result. Replay improved fit to the restaurant demonstrations. This small comparison provides no evidence of better held-out task success or retention. [Episode results, checkpoint hashes and reproduction settings](benchmarks/real-v2-web.json)

## Reproduce the web experiment

Run from the repository root on Linux x86-64 with Python 3.11, uv and a native BF16 CUDA GPU. The two Python environments serve different purposes. PDM trains and serves the model. The pinned REAL-Eval harness controls Chromium and scores the web episodes. Its source is a **private submodule** and fetching it requires access to the original repository. PDM training on an existing compatible manifest and serving an adapter use the main repository alone.

```bash
git submodule update --init REAL-Eval
uv sync --project REAL-Eval --locked --python 3.11
REAL-Eval/.venv/bin/playwright install chromium --only-shell
uv sync --locked --extra gpu --python 3.11
```

1. **Collect verified training demonstrations.** The practice websites and official evaluation deployments require network access. Use fresh output directories for collection and training.

   ```bash
   REAL-Eval/.venv/bin/python -m pdm.web_collect --output runs/web-training-demo
   ```

2. **Train.** The manifest contains a training episode ID, service domain, split, system text, observation text, relative JPEG path and target action. The run saves adapter weights and fingerprints, exact batch membership, losses, and the replay memory IDs. Model and processor downloads use the pinned Hugging Face revision on first load.

   ```bash
   .venv/bin/python -m pdm.web train --manifest runs/web-training-demo/manifest.jsonl \
     --out runs/web-train-42 --passes 2 --capacity 8 --seed 42
   ```

3. **Evaluate the saved checkpoints in the browser.** Evaluation starts and stops its own loopback inference endpoint. The quick comparison uses the official birthday restaurant request (`v2.opendining-5`) and the Chestnut Street to Rooftop 25 ride request (`v2.udriver-4`). Both are separate from the training instructions. Each condition gets greedy decoding, the same basic prompt and screenshot, and the same step limit. The evaluator uses the site's state checks and supplies no task-specific hints.

   ```bash
   .venv/bin/python -m pdm.web evaluate --run runs/web-train-42 --out runs/web-real-42 \
     --real-eval REAL-Eval/.venv/bin/real-eval --max-steps 25
   ```

   Open `runs/web-real-42/real-eval/replay-udriver/index.html` for the final replay agent's screenshots and actions. `web-summary.json` includes the site success matrix and the number of scored tasks behind each figure. Rerunning with the same directory retries unavailable episodes and reuses finished ones. Use `--tasks` to select additional official v2 tasks from both services.

To run **inference** with the trained replay adapter on a free-form instruction, start the server in one terminal and run the harness in another. The first command loads the saved weights. The second uses the same observation layout as evaluation and writes its own trace. Free-form instructions do not receive an official benchmark score.

```bash
.venv/bin/python -m pdm.web serve --checkpoint runs/web-train-42/replay-udriver.safetensors --port 8017
# In a second terminal, from the same directory
REAL-Eval/.venv/bin/real-eval ask \
  "Reserve a table for two at the first restaurant in the home page's list of available restaurants. Choose its first offered time today. Use phone 4155550146 and email diner@example.com." \
  --site opendining \
  --config runs/web-real-42/real-eval/replay-udriver/pdm-config.json \
  --model Qwen/Qwen2.5-VL-3B-Instruct --base-url http://127.0.0.1:8017/v1 --results-dir runs/web-inference --no-video
```

To inspect **fit to the training demonstrations**, run `.venv/bin/python -m pdm.web score-demos --manifest runs/web-training-demo/manifest.jsonl --run runs/web-train-42`. The resulting action-token loss describes imitation of the supplied examples. The held-out browser success rate measures whether the model completes a new instruction.

## Tests

```bash
uv run --locked --extra cpu python -m pytest -q
(cd REAL-Eval && .venv/bin/python -m pytest -q)
```

## Acknowledgment

This research is associated with the Korean national R&D project **“Developing a Sustainable Collaborative Multi-modal Lifelong Learning Framework” (Project No. 2022-0-00113)**, conducted within the *Human-Centered AI Core Technology Development* program, supported by the Ministry of Science and ICT (MSIT) and administered by the Institute of Information & Communications Technology Planning & Evaluation (IITP).
