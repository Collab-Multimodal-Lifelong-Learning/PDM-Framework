# PDM-Framework

[한국어 안내](README_ko.md)

PDM-Framework trains a multimodal browser agent across two simulated services. On **OpenDining**, the agent follows a restaurant reservation request. On **UDriver**, it follows a ride request. You can give a saved agent your own instruction, compare checkpoints on the same benchmark tasks, and inspect the browser actions behind each booking.

## How the agent works

At each step, the policy receives the user's instruction, a screenshot, the current URL, a numbered accessibility tree and recent actions. It generates one command, such as clicking a time slot or filling a pickup field. Chromium executes the command and returns the next observation. The trace preserves the model's response, the parsed action and any browser error. [Browser runtime](pdm/browser.py)

| Site | Example instruction | What to inspect in the final state |
| --- | --- | --- |
| `opendining` | Reserve a table for two at the first available time. | Restaurant, party size and reservation time |
| `udriver` | Book a Comfort ride from Local Edition to Golden Farmer Market and pay with cash. | Pickup, destination, vehicle class and payment method |

**PDM** stands for *Pre-trained, Data, Memory*. The pre-trained policy is Qwen2.5-VL-3B-Instruct with a rank-8 LoRA adapter. The training data pairs each demonstrated action with the screen and page elements visible immediately before it. A scripted demonstrator verifies each booking against the site's record before publishing the training manifest. [Policy](pdm/web_model.py) · [Demonstration collector](pdm/web_collect.py)

The agent learns restaurant reservations first. Two training methods then start from that same adapter. In **sequential training**, each update uses two ride actions. In **replay**, it uses one ride action and one earlier restaurant action drawn from a bounded memory. Both methods see the same current-action order and receive the same number of updates. Evaluating both services after the switch lets you examine new learning and retention separately. [Training loop](pdm/web.py)

## Install

Run from the repository root on Linux x86-64 with Python 3.11 and uv. Training and inference require a CUDA GPU with native BF16 support. The browser environment uses Chromium and the public AGI SDK. It needs network access to the simulated sites. The first model load also downloads the pinned Qwen weights.

```bash
uv sync --locked --extra gpu --python 3.11
uv sync --project browser-env --locked --python 3.11
browser-env/.venv/bin/python -m playwright install --with-deps --only-shell chromium
```

## Collect demonstrations and train

Use fresh output directories. The collector writes the manifest and screenshots together. Training saves the shared restaurant adapter and one final adapter for each ride-training method. Here `--capacity 8` sets the maximum number of restaurant actions kept for replay.

```bash
.venv/bin/pdm-agent collect --out runs/demos
.venv/bin/pdm-agent train --manifest runs/demos/manifest.jsonl \
  --out runs/agent --passes 2 --capacity 8 --seed 42
```

## Give the agent an instruction

`ask` loads the checkpoint, starts a local policy endpoint and opens a fresh browser on the selected site. Supply your own request with `--goal` and a new session directory with `--out`.

```bash
.venv/bin/pdm-agent ask --checkpoint runs/agent/replay-udriver.safetensors \
  --site udriver --out runs/my-ride \
  --goal "Book a Comfort ride from Local Edition to Golden Farmer Market and pay with cash."
```

For this request, inspect `runs/my-ride/trace.jsonl` to see how the agent selected a vehicle and `runs/my-ride/state.json` to check the recorded trip. A free-form request has no benchmark score.

## Evaluate checkpoints

`evaluate` runs the frozen policy, the restaurant checkpoint for each method and the two final ride adapters. Each condition gets the same held-out requests and action budget. The SDK scores each task using the site's final state and the agent's answer. The summary reports per-service success and the change in restaurant success after ride training. Unavailable episodes are counted separately. [Evaluator and supported task IDs](pdm/web_eval.py)

```bash
.venv/bin/pdm-agent evaluate --run runs/agent --out runs/evaluation \
  --tasks v2.opendining-5 v2.udriver-4 --max-steps 25
```

Select at least one supported task from each service. Rerunning the command with the same settings and output directory reuses scored episodes and retries unfinished ones. To measure how closely each checkpoint predicts the demonstrated actions, use `.venv/bin/pdm-agent score-demos --run runs/agent --manifest runs/demos/manifest.jsonl`.

## Inspect an episode

For the evaluation command above, the final replay agent's ride episode lives under `runs/evaluation/checkpoints/replay-udriver/episodes/v2.udriver-4/`.

| File | What it tells you |
| --- | --- |
| `record.json` | The task, final answer, step count, reward and reason the episode ended |
| `trace.jsonl` and numbered JPEGs | What the agent saw and which action it attempted at each step |
| `state.json` | What the simulated service recorded, including the booking details |
| `score.json` | The task scorer's checks for a benchmark episode |
| `runs/evaluation/web-summary.json` | Success counts and retention comparisons across checkpoints |

## Acknowledgment

This research is associated with the Korean national R&D project **“Developing a Sustainable Collaborative Multi-modal Lifelong Learning Framework” (Project No. 2022-0-00113)**, conducted within the *Human-Centered AI Core Technology Development* program, supported by the Ministry of Science and ICT (MSIT) and administered by the Institute of Information & Communications Technology Planning & Evaluation (IITP).
