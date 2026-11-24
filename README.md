# PDM-Framework

[한국어 요약](README_ko.md) · [Recorded benchmark](benchmarks/agent-42.json)

PDM (*Pre-trained, Data, Memory*) trains a browser agent to reserve tables on OpenDining and then book rides on UDriver. Both are simulated services. The experiment tests whether replaying earlier restaurant actions during ride training helps the agent complete a restaurant request afterward.

## What the experiment showed

The table counts completed bookings on **two held-out requests** with seed 42. Each checkpoint condition ran once per request, with a limit of 25 browser actions. The two methods shared the restaurant checkpoint and evaluated it in separate browser sessions.

| Checkpoint | Birthday dinner reservation | Ride on a new route |
| --- | ---: | ---: |
| Frozen model | 0/1 | 0/1 |
| After restaurant training (shared weights, two sessions per request) | 0/2 | 0/2 |
| After ride training, sequential | 0/1 | 1/1 |
| After ride training, with replay | 0/1 | 1/1 |

Both final adapters booked the new ride in 15 actions. None of the five dinner attempts succeeded, so this run provides no evidence that replay improves held-out restaurant success. The ten episodes all received scores. [Per-episode scores and adapter hashes](benchmarks/agent-42.json)

**Repeating a training request.** In a separate check, the final replay adapter was asked twice to book the first listed restaurant for two. Both sessions produced a reservation in six actions. Only one matched the request. It booked Vintage Bites at 12:30 PM. The other booked the second restaurant, Evening Delight, at 1:00 PM.

## How the agent learns

| Component | Implementation |
| --- | --- |
| **Pre-trained model** | Qwen2.5-VL-3B-Instruct reads the instruction, screenshot, accessibility tree and action history. A rank-8 LoRA adapter learns to generate one browser action at a time. [Policy](pdm/web_model.py) |
| **Data** | The collector verifies the site's final records for two reservations and two rides. Each training example pairs the observation before a step with the action taken. This run yielded 12 restaurant actions and 29 ride actions. [Collector](pdm/web_collect.py) |
| **Memory** | Reservoir sampling keeps eight restaurant actions. During ride training, replay pairs a current ride action with a stored restaurant action. Sequential training pairs it with another ride action. [Training loop](pdm/web.py) |

The shared restaurant adapter received 24 updates. Each ride-training method then received 58 updates, using a fresh AdamW optimizer and the same current-action order. On the original demonstrations, replay reduced the final mean restaurant action loss from 0.177 (sequential) to 0.122. Ride action losses were 0.120 and 0.136, respectively. These losses measure how closely the model imitates demonstrated actions. The booking scores above use the simulated site's final state. [Browser runtime](pdm/browser.py) · [Evaluator](pdm/web_eval.py)

## Install

Run from the repository root on Linux x86-64 with Python 3.11 and uv. Training and inference require a CUDA GPU with native BF16 support. Collection and evaluation need Chromium and network access to the simulated sites. The first model load downloads the pinned Qwen weights.

```bash
uv sync --locked --extra gpu --python 3.11
uv sync --project browser-env --locked --python 3.11
browser-env/.venv/bin/python -m playwright install --with-deps --only-shell chromium
```

## Collect, train and evaluate

Use fresh collection and training directories. Both training methods run from the same manifest. Evaluation starts its own local policy server.

```bash
.venv/bin/pdm-agent collect --out runs/agent-demos

.venv/bin/pdm-agent train --manifest runs/agent-demos/manifest.jsonl \
  --out runs/agent-42 --passes 2 --capacity 8 --seed 42

.venv/bin/pdm-agent evaluate --run runs/agent-42 --out runs/evaluation-42 --max-steps 25
```

Open `runs/evaluation-42/web-summary.json` for per-service scores and the number of episodes scored. To inspect the final replay agent's ride, read `runs/evaluation-42/checkpoints/replay-udriver/episodes/v2.udriver-4/trace.jsonl` alongside the screenshots and `state.json` in that directory.

To reproduce the demonstration losses, run the following command. It writes `demonstration-fit.json` next to the saved adapters in `runs/agent-42/`.

```bash
.venv/bin/pdm-agent score-demos --run runs/agent-42 --manifest runs/agent-demos/manifest.jsonl
```

## Run an instruction

After training, use `ask` to load a saved adapter and open a fresh browser. The following example repeats the practice request described above.

```bash
.venv/bin/pdm-agent ask --checkpoint runs/agent-42/replay-udriver.safetensors \
  --site opendining --out runs/practice-agent \
  --goal "Reserve a table for two at the first restaurant in the home page's list of available restaurants. Choose its first offered time today. Use phone 4155550146 and email diner@example.com."
```

Read `runs/practice-agent/trace.jsonl` to inspect the clicks and `runs/practice-agent/state.json` to verify which restaurant and time were booked.

## Tests

```bash
uv run --isolated --locked --extra cpu python -m pytest -q
```

## Acknowledgment

This research is associated with the Korean national R&D project **“Developing a Sustainable Collaborative Multi-modal Lifelong Learning Framework” (Project No. 2022-0-00113)**, conducted within the *Human-Centered AI Core Technology Development* program, supported by the Ministry of Science and ICT (MSIT) and administered by the Institute of Information & Communications Technology Planning & Evaluation (IITP).
