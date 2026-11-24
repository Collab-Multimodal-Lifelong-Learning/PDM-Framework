# PDM-Framework

[한국어 요약](README_ko.md)

PDM (*Pre-trained, Data, Memory*) is a continual learner for browser agents. It learns to reserve tables on OpenDining, then book rides on UDriver. The experiment asks whether replaying a small number of restaurant actions during ride training helps the agent retain its first skill. Both services run on simulated websites.

## What the agent learns

| Component | Implementation |
| --- | --- |
| **Pre-trained model** | Qwen2.5-VL-3B-Instruct reads the user instruction, screenshot, accessibility tree and previous actions. A rank-8 LoRA adapter on the decoder's query and value projections predicts one browser action. [Policy](pdm/web_model.py) |
| **Data** | A scripted demonstrator completes two restaurant bookings and two trips. The collector checks the site's final booking records, then saves the observations and executed actions as training pairs. [Collector](pdm/web_collect.py) |
| **Memory** | At the service boundary, the learner keeps up to eight restaurant steps by reservoir sampling. During ride training, replay pairs each new step with a remembered restaurant step. The sequential baseline pairs it with another ride step. [Training](pdm/web.py) |

Both methods start from the same restaurant adapter, use a fresh AdamW optimizer for ride training and get the same current-action order and update budget. The loss is cross-entropy on the demonstrated action tokens and their end-of-sequence token. The browser executes the generated actions and the task's own state checks determine booking success. [Browser runtime](pdm/browser.py) · [Evaluation](pdm/web_eval.py)

## Measured results

Seed 42 produced 41 verified action examples (12 restaurant, 29 ride). The shared restaurant adapter received 24 updates, then each ride-training method received 58. Final action losses on the restaurant demonstrations were 0.177 for sequential training and 0.122 for replay. On the ride demonstrations the losses were 0.120 and 0.136, respectively.

| Browser check | Outcome |
| --- | --- |
| Repeat the first restaurant instruction twice with the final replay adapter | Both fresh browsers produced a booking in six actions. One booked the requested first restaurant, Vintage Bites. The other booked the second restaurant, Evening Delight. |
| Two unseen requests, one per service | **2/10 successful episodes** across five checkpoint conditions. Both final adapters booked the held-out ride in 15 actions each. The frozen model and the shared restaurant adapter failed that ride. All five restaurant attempts failed. |

The two methods ran their shared restaurant checkpoint in separate browsers. With one held-out request per service and identical final scores, this run shows successful transfer to a new ride request for both methods. It does not establish a task-success advantage for replay. [Recorded scores, practice receipts and checkpoint hashes](benchmarks/agent-42.json)

## Install

Run from the repository root on Linux x86-64 with Python 3.11 and uv. Model training and inference require a CUDA GPU with native BF16 support. The first model load downloads the pinned Qwen weights. Browser collection and evaluation require Chromium and network access to the simulated sites.

```bash
uv sync --locked --extra gpu --python 3.11
uv sync --project browser-env --locked --python 3.11
browser-env/.venv/bin/python -m playwright install --with-deps --only-shell chromium
```

The main environment trains and serves the agent. The browser environment runs the public AGI SDK and Chromium. [Model dependencies](pyproject.toml) · [Browser dependencies](browser-env/pyproject.toml)

## Collect, train and evaluate

Use fresh directories for collection and training. The evaluator launches the policy endpoint, tests the saved checkpoints on the same held-out restaurant and ride requests, and writes per-episode results.

```bash
.venv/bin/pdm-agent collect --out runs/agent-demos

.venv/bin/pdm-agent train --manifest runs/agent-demos/manifest.jsonl \
  --out runs/agent-42 --passes 2 --capacity 8 --seed 42

.venv/bin/pdm-agent evaluate --run runs/agent-42 --out runs/evaluation-42 \
  --browser-python browser-env/.venv/bin/python --max-steps 25

.venv/bin/pdm-agent score-demos --run runs/agent-42 --manifest runs/agent-demos/manifest.jsonl
```

| Output | Contents |
| --- | --- |
| `runs/agent-demos/manifest.jsonl` | One row for each observed training action, with its screenshot path |
| `runs/agent-42/training.json` | Update losses, replay selections, adapter filenames and hashes |
| `runs/agent-42/demonstration-fit.json` | Mean action loss per checkpoint on the training examples |
| `runs/evaluation-42/web-summary.json` | Held-out booking successes, scored episodes and change in restaurant success after ride training |

## Run an instruction

Run a natural-language request with the final replay adapter. The command loads the weights, opens a fresh browser, and writes the model's actions, screenshots, and the site's final state.

```bash
.venv/bin/pdm-agent ask --checkpoint runs/agent-42/replay-udriver.safetensors \
  --site opendining --out runs/practice-agent \
  --goal "Reserve a table for two at the first restaurant in the home page's list of available restaurants. Choose its first offered time today. Use phone 4155550146 and email diner@example.com."
```

This example repeats a training request. Inspect `runs/practice-agent/trace.jsonl` for the action sequence and `state.json` in the same directory for the site's booking record. Use `evaluate` for state-scored held-out tasks.

## Tests

```bash
uv run --isolated --locked --extra cpu python -m pytest -q
```

## Acknowledgment

This research is associated with the Korean national R&D project **“Developing a Sustainable Collaborative Multi-modal Lifelong Learning Framework” (Project No. 2022-0-00113)**, conducted within the *Human-Centered AI Core Technology Development* program, supported by the Ministry of Science and ICT (MSIT) and administered by the Institute of Information & Communications Technology Planning & Evaluation (IITP).
