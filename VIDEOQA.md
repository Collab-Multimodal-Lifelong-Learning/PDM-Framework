# PDM Framework · Continual VideoQA

[PDM web agent and project overview](README.md) · [한국어 요약](README_ko.md)

PDM (Pre-trained, Data, Memory) uses one video-language adapter to answer a question about a clip, locate supporting frames, and retrieve clips for a supplied question and answer. The online learner processes labeled examples once in arrival order. A second experiment measures how the choice of replay frames affects retained answers and temporal evidence.

## Online learning and inference with one adapter

Project 2022-0-00113 describes a shared multimodal learner for multiple downstream tasks (research plan, p. 24), memory and downstream prediction losses (p. 41), and video learning without task-boundary signals (p. 48). This repository implements a **video–language application** of those goals. The quick benchmark shifts from temporal to causal questions within one video dataset. Cross-dataset transfer, time-series forecasting, and the project's broader application demonstrations remain outside the evidence from this example.

| Component | What happens when a labeled clip arrives |
| --- | --- |
| **P · Pre-trained** | Qwen2.5-VL encodes the eight frames, question, and five choices. A rank-8 LoRA adapter on the language decoder learns both answer classification and clip matching. |
| **D · Data** | The learner consumes the training rows in manifest order. It receives the clip, question, choices and verified answer. One AdamW optimizer persists across arrivals and reporting checkpoints. |
| **M · Memory** | A bounded reservoir samples training examples from the entire stream. Each entry refers to its original clip and holds the answer and the five logits observed before its update. Old clips must remain accessible during replay. |

For each arrival, `pdm_stream` minimizes answer cross-entropy. When memory is available, it also minimizes cross-entropy and logit MSE on one earlier example. An eligible older clip supplies a negative for a two-clip ranking loss. The ranking score is the log probability of the query's supplied answer given the candidate clip and the five choices. Replay CE, distillation MSE and ranking loss each have weight 0.5 relative to current QA CE. Known duplicate question/answer positives and the source clip are excluded from negative sampling. `stream_lora` uses the same arrivals, initial weights and number of optimizer steps with current QA CE alone. The additional forward passes cost time and memory. [Online learner and retrieval](pdm/stream.py)

The implementation adapts the stored-logit replay and reservoir ideas from the DER baseline in the collected SDP repository and the positive-versus-negative retrieval objective from Axis-LoRA. PDM uses its existing video scorer for both tasks. [SDP repository's DER trainer](https://github.com/Collab-Multimodal-Lifelong-Learning/sdp/blob/c917fb2b2a50b498840ba5edc810d21d65be8d16/methods/der.py) · [Axis-LoRA contrastive objective](https://github.com/Collab-Multimodal-Lifelong-Learning/axis-lora/blob/2dec422685355c3d0d5ec220ae52925e1dd359c8/axis_lora/losses.py)

Run from this repository with Python 3.11 and a native BF16 NVIDIA GPU. Training writes the adapter at the midpoint and end of the stream. These indices also set the evaluation schedule (temporal questions first, then temporal and causal). The learner sees no task IDs or boundary signal, uses the latest weights at each checkpoint, and performs one update per arrival independently of the offline `steps` and `passes` settings.

```bash
uv run --locked --extra gpu python -m pdm demo --profile nextqa-quick --device cuda:0 \
  --conditions stream_lora pdm_stream --memory-capacity 4 --out runs/pdm-stream-42
uv run --locked --extra gpu python -m pdm evaluate-grounding --run-dir runs/pdm-stream-42
uv run --locked --extra gpu python -m pdm predict --run-dir runs/pdm-stream-42 \
  --condition pdm_stream --sample-id '3606510307:9:TC'
```

For a new clip, `predict` accepts `--clip`, `--question`, and five `--choice` arguments. For retrieval, supply the answer as part of the query and at least two candidate clips. This example uses clips already cached by the quick demo.

```bash
videos="$HOME/.cache/pdm-framework/nextqa/videos"
uv run --locked --extra gpu python -m pdm retrieve-video --run-dir runs/pdm-stream-42 \
  --question "what does the man in blue hold in his hands" --query-answer camera \
  --choice camera --choice bottle --choice cap --choice "toy plane" --choice "his baby" \
  --clip "$videos/3606510307.mp4" --clip "$videos/3459862895.mp4"
```

The retrieval evaluation ranks the source clip against up to four other distinct held-out clips. **Recall@1** is the fraction of queries that rank their source first. **MRR** averages the reciprocal source rank. Another clip can also fit the description. The reference identifies the recorded source. `retrieval.json` stores every candidate, score and rank. Each checkpoint's `state.json` stores per-arrival losses and replay IDs. The video report displays retrieval and QA metrics, and `grounding.json` scores estimated time intervals against human annotations.

### Observed online run

With seed 42 on NExT-QA Quick, each method took 16 optimizer steps on the same 16 training arrivals. Online PDM held four examples and performed 15 replay and 15 clip-ranking updates. Each final test query ranked its source among five candidate clips. Four questions also had NExT-GQA human time annotations.

| Final checkpoint | Correct QA | Source clip ranked first | Retrieval MRR | Correct answer with human evidence overlap (IoP ≥ 0.5) |
| --- | ---: | ---: | ---: | ---: |
| Online LoRA | 3/6 | 5/6 | 0.867 | 2/4 |
| Online PDM | 3/6 | 5/6 | 0.867 | 1/4 |

Both methods updated their adapters. The combined objective tied QA and retrieval, and scored lower on grounded answers in this small run. The test set and single seed support a functional demonstration only.

For a CPU check with generated color clips, run `uv run --locked --extra cpu python -m pdm demo --conditions stream_lora pdm_stream --memory-capacity 4 --out runs/pdm-stream-cpu`. The CPU backend verifies the same training, checkpoint and inference paths using a small RGB model.

## Train and evaluate temporal evidence replay

Run these commands from this repository on Linux x86-64 with Python 3.11, uv, and a native BF16 NVIDIA GPU with a CUDA 12.8-compatible driver. Choose a fresh output directory for training. The first command downloads the pinned Qwen model and the 26 selected NExT-QA clips. The second downloads two small NExT-GQA test annotation files and scores the saved predictions on CPU. Repeating the second command reuses the saved predictions and annotations.

```bash
uv run --locked --extra gpu python -m pdm demo --profile nextqa-quick --device cuda:0 \
  --conditions pdm pdm_random_window pdm_evidence --seed 42 --out runs/grounded-42
uv run --locked --extra gpu python -m pdm evaluate-grounding --run-dir runs/grounded-42
```

| Training condition | Frames revisited for a previous training question |
| --- | --- |
| `pdm` | Eight frames sampled across the full clip |
| `pdm_random_window` | Eight frames inside a seeded, frame-anchored random window of the same length as the estimated evidence |
| `pdm_evidence` | Eight frames inside the window selected by the previous task's retained checkpoint |

After the temporal task, the selected checkpoint predicts each incoming memory question's answer from the full clip. It scores that answer in each nonempty quarter and stores the highest-scoring interval alongside the training ID. The random control computes the same estimate to obtain the duration, then draws a frame as the start and clips the start to fit the window. During the causal task, a replay update uses three current full clips and one prior example with its original training answer as the cross-entropy target. The predicted interval can therefore be imperfect, including when the earlier model answered incorrectly.

The first method in this comparison trains the temporal-task adapter once. All participating replay methods load those exact weights before building their own memory. Their causal-task adapters train separately. On GPU this also removes variation from independently repeating the first backward passes. All three methods use the same FIFO capacity, replay IDs for a given seed, current-task gradient selection rule, frames per training example, and optimizer-step budget. Each window method pays the extra inference cost of estimating the intervals. The measured `update_seconds` includes extraction. Only the first method pays the shared training cost. The resource table identifies reused training and counts extractions. The human temporal annotations enter the workflow only in the standalone evaluator.

**Read the evaluation.** Open `report.html` for per-question predictions and per-pass training losses, and `grounding.json` for matched IDs, predicted and human intervals, task-level scores and forgetting. The evaluator checks video/question/type IDs, question text, choice order and the correct answer. Questions outside the annotated subset are listed separately. Each checkpoint is scored on all seen test tasks. [Official annotations](https://github.com/doc-doc/NExT-GQA/tree/63772e0256ad9e34b83d6a108f1ba2041b924607/datasets/nextgqa)

- **Answer accuracy** measures correct choices. The run report covers all test questions. The grounding report also gives accuracy on just the annotated subset.
- **Mean IoP** averages the fraction of the predicted interval covered by a human interval. **Mean IoU** uses the union of the two intervals as the denominator. For several valid human spans, the evaluator takes the best overlap for each metric.
- **Acc@GQA** counts a question when the answer is correct and IoP is at least 0.5. **Previous-task forgetting** is each earlier task's best score through the current checkpoint minus its current score, averaged over earlier tasks with annotations. The JSON reports fractions. [NExT-GQA reference evaluator](https://github.com/doc-doc/NExT-GQA/blob/63772e0256ad9e34b83d6a108f1ba2041b924607/code/TempGQA/eval_ground.py)

To run the same comparison locally on generated clips with known evidence, use the CPU commands below. For offline NExT-GQA evaluation, pass `--annotations /path/to/nextgqa` with the pinned `test.csv` and `gsub_test.json`.

```bash
uv run --locked --extra cpu python -m pdm demo --out runs/grounded-cpu \
  --conditions pdm pdm_random_window pdm_evidence
uv run --locked --extra cpu python -m pdm evaluate-grounding --run-dir runs/grounded-cpu
```

### Observed GPU results

One seed-42 quick run used 8 training, 2 validation and 3 test questions per task. Four of the six held-out questions had NExT-GQA intervals, two from each task. The table reports results after learning causal questions. All three conditions used the same temporal-task adapter and replayed the same training IDs. The random and estimated windows had identical lengths for all eight stored examples.

| Replay view | Correct answers, all test questions | Correct answer and IoP ≥ 0.5, annotated subset | Mean IoP, annotated subset | Earlier temporal task's mean IoP drop |
| --- | ---: | ---: | ---: | ---: |
| Full clip | 3/6 | 2/4 | 47.0% | 0.0 pp |
| Random window | 2/6 | 1/4 | 28.9% | 11.6 pp |
| Estimated evidence | 3/6 | 2/4 | 41.2% | 11.6 pp |

Evidence replay matched full-clip replay on correct grounded answers. Full-clip replay retained more overlap on the earlier task. Each annotated answer moves Acc@GQA by 25 percentage points in this run. A larger test set and multiple seeds are needed to estimate the effect.

## Offline replay and checkpoint selection

| Component | What the code does | Source |
| --- | --- | --- |
| **P · Pre-trained model** | Qwen2.5-VL scores five choices from eight sampled video frames. It chooses an answer on the full clip, then rescores four time windows to find the strongest support for that answer. Training updates rank-8 LoRA weights on the language decoder's query and value projections. | [Video scoring and evidence selection](pdm/models.py) |
| **D · Data and selection** | The importer builds clip-disjoint train, validation, and test splits. Before each training pass, PDM ranks current-task questions by the size of their loss gradients on the last decoder block's LoRA weights and penalizes directions similar to those of questions already selected. | [Dataset and model inputs](pdm/data.py) · [Gradient-based selection](pdm/continual.py) |
| **M · Memory** | A bounded FIFO stores labeled training questions. PDM replays one earlier example per four-example update while memory permits. The retrieval-only baseline uses up to three stored question/answer pairs as inference context. | [Memory, retrieval, and replay](pdm/continual.py) |

The offline controller trains each task in configurable passes. After each pass it computes the mean validation loss, giving equal weight to every seen task. The adapter with the lowest validation loss is used for held-out testing and carried into the next task. AdamW state persists across passes and resets between tasks. [Experiment and checkpoint code](pdm/experiment.py) · [Results viewer](pdm/report.py)

## Run the application

Use Linux x86-64, Python 3.11, and uv. Run the commands from this repository. Each experiment creates a new output directory. Keep the generated HTML and its media together for playback.

**CPU demo.** One command generates 120 four-second color clips, runs the six baseline conditions, saves the adapters, and creates a local video viewer. This fixture uses a rank-2 LoRA adapter on RGB features and exercises video decoding, backpropagation, and replay. Its model supports the two color questions shown in the viewer.

```bash
uv run --locked --python 3.11 --extra cpu python -m pdm demo --out runs/my-cpu-demo
```

Open `runs/my-cpu-demo/report.html`. Select a clip and question, compare methods before and after the second task, then choose **Play evidence**. The highlighted interval is the highest-scoring quarter of the clip for the predicted answer. The CPU report also includes a paired example where an aligned sensor event changes the selected interval.

**Real-video demo.** On an NVIDIA GPU with native BF16 and a CUDA 12.8-compatible driver, use the same application with a pretrained model. The first run downloads the pinned Qwen weights and the selected official NExT-QA videos. The video cache defaults to `~/.cache/pdm-framework/nextqa` and can be changed with `--data-dir`. [Qwen model](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3) · [NExT-QA data and attribution](https://github.com/doc-doc/NExT-QA/tree/2432e9724f88ed9f40010e2989f104570a91de4e)

```bash
uv run --locked --python 3.11 --extra gpu python -m pdm demo \
  --profile nextqa-quick --device cuda:0 --out runs/my-videoqa-demo
```

Open `runs/my-videoqa-demo/report.html`. The real-video demo uses 8 training, 2 validation, and 3 test questions **per task**. Each test answer represents one third of that task's accuracy. Use this run to inspect individual answers and verify integration. For a larger experiment use the benchmark command below.

Each offline trained condition in the quick GPU demo runs three passes by default. Under **Training and validation history**, the viewer lists the training and validation losses for each pass and marks the checkpoint selected for held-out evaluation.

### Answer a new question

The GPU `ask` command accepts a local video and exactly five choices. It prints the selected answer, five raw logits, the evidence interval in milliseconds, and the scores for the candidate windows.

```bash
uv run --locked --extra gpu python -m pdm ask --device cuda:0 \
  --clip /path/to/video.mp4 --question "What happens after the door opens?" \
  --choice "Someone enters" --choice "The lights go out" --choice "The door closes" \
  --choice "A bell rings" --choice "The camera moves"
```

To ask with a trained model, use `predict --run-dir RUN --condition pdm --stage 2` with the same new-clip arguments or a saved sample ID. Here `--stage` selects the checkpoint after the first or second **learning task**. The example below reproduces a prediction from the CPU demo.

```bash
uv run --locked --extra cpu python -m pdm predict \
  --run-dir runs/my-cpu-demo --condition pdm --stage 2 --sample-id s1-test-000:q01
```

For a real held-out example, reload the PDM checkpoint and ask what the man in blue is holding.

```bash
uv run --locked --extra gpu python -m pdm predict \
  --run-dir runs/my-videoqa-demo --condition pdm --stage 2 --sample-id '3606510307:9:TC'
```

### Run a larger benchmark

For a local copy of the official annotations and media, use the pinned train/val/test CSVs and videos named `<video_id>.mp4` under one directory. The importer verifies annotation checksums, clips, unique question IDs, and split isolation. The larger profile selects up to 128 training, 64 validation, and 128 test questions per task using a stable hash of each ID. The subsets stay fixed when the optimizer seed changes. [Official annotations](https://github.com/doc-doc/NExT-QA/tree/2432e9724f88ed9f40010e2989f104570a91de4e/dataset/nextqa)

```bash
uv run --locked --extra gpu python -m pdm prepare-nextqa \
  --annotations /data/nextqa/annotations --videos /data/nextqa/videos --out runs/data/nextqa.jsonl
uv run --locked --extra gpu python -m pdm run --config configs/nextqa.json \
  --manifest runs/data/nextqa.jsonl --device cuda:0 --seed 42 --out runs/nextqa-42 \
  --conditions pdm pdm_random_window pdm_evidence
uv run --locked --extra gpu python -m pdm evaluate-grounding --run-dir runs/nextqa-42
```

## Compare learning methods

| Report condition | Current-task sample selection | Previous-task memory |
| --- | --- | --- |
| `frozen` | Fixed starting weights | Empty |
| `retrieval` | Fixed starting weights | Retrieved context at inference |
| `lora` | Seeded order | Empty |
| `pdm` | Gradient magnitude and diversity | Replay during training |
| `pdm_uniform` | Seeded order | Replay during training |
| `pdm_no_replay` | Gradient magnitude and diversity | Empty |

Compare `pdm` with `pdm_uniform` to measure the effect of gradient-based selection with replay on. Compare `pdm` with `pdm_no_replay` to measure the effect of allocating part of the batch to replay while keeping the selection rule.

Outside the three-way replay comparison, each baseline starts independently. The offline trained conditions receive the same optimizer-step and effective-batch budgets. Each pass takes one AdamW step in the quick GPU demo, 16 in the larger benchmark, and eight in the CPU demo. The GPU presets use three passes per task and the CPU preset uses one. Use `--passes` to set the number of passes and `--conditions` to choose methods. The default runs the six baseline conditions in the table. Pass the explicit three-condition list above for the temporal evidence replay experiment.

In a configuration file, `steps` sets the number of optimizer updates per pass and `passes` sets the number of checkpoint candidates per task. PDM draws replay examples from the memory of earlier tasks. Validation records are used to score checkpoints. Held-out test questions are scored after the task checkpoint has been selected.

## Inspect the results

| Output | Contents |
| --- | --- |
| `report.html` | Video playback, answers by method and checkpoint, annotated and estimated intervals when available, accuracy and resource tables |
| `run.json`, `manifest.jsonl` | Run status, resolved configuration, environment, revisions, checksums, and the exact input records |
| `metrics.json`, `predictions.jsonl` | Per-pass training and validation losses, test accuracy for each task, individual predictions, evidence scores and labels |
| `checkpoints/<condition>/stage-<n>/` | Selected adapter for trained methods, training-memory IDs, parameter fingerprints, selected pass, and all attempted training and replay IDs |
| `grounding.json` | Standalone evaluation of saved predictions, including coverage, official annotation revision for NExT-GQA, per-example overlap, and previous-task forgetting |
| `sensor_comparison.json`, `sampler_comparison.json` | The paired sensor example on CPU and the frozen frame-sampler comparison on GPU, when enabled |

The accuracy matrix has one row per checkpoint and one column per task. **Forgetting** is the earlier task's best observed accuracy minus its current accuracy. **Adaptation gain** compares the newly learned task with the frozen model at the same checkpoint. The report also includes inference latency, update time including selection and validation, peak allocated GPU memory, and evidence IoU for samples with labeled intervals. Timing uses an untimed validation forward pass and initializes the optimizer runtime before measurement. Video caches start empty for each timed update and held-out evaluation. NExT-QA answer labels support answer accuracy. For human-labeled NExT-GQA overlap, run `evaluate-grounding` on the same directory.

For a new clip, optional `--sensor-csv` accepts strictly increasing `timestamp_ms,value` measurements on the clip clock. The model chooses the answer from video, then ranks evidence windows using 80% video confidence and 20% normalized sensor deviation.

## Research code and tests

The four sibling collections hold 60 pinned study repositories. [Vision](https://github.com/Collab-Multimodal-Lifelong-Learning/Continual-Learning-Vision-Domain) · [Language](https://github.com/Collab-Multimodal-Lifelong-Learning/Continual-Learning-Language-Domain) · [Time series](https://github.com/Collab-Multimodal-Lifelong-Learning/Continual-Learning-Time-Series-Domain) · [Multimodal integration](https://github.com/Collab-Multimodal-Lifelong-Learning/Continual-Learning-and-Multimodal-Integration)

DynaTokens studies adaptation and forgetting in continual VideoQA. Hidden Forgetting examines how answer accuracy can mask changes in multimodal evidence use. These motivate evaluating PDM on both the answer and the temporal interval. The experiment implemented here compares which part of a remembered video to replay. [DynaTokens](https://aclanthology.org/2026.emnlp-main.2425/) · [Hidden Forgetting](https://arxiv.org/abs/2607.02020)

PDM integrates the VLM-RLAIF frame-index sampling rule as a paired comparison against timestamp sampling. The gradient-based selection rule is inspired by OASIS. Both adaptations are implemented in this repository. [VLM-RLAIF source](https://github.com/Collab-Multimodal-Lifelong-Learning/vlm-rlaif/blob/610cac72c16ae3862b0d83a0dc3708d73a42a4cc/Evaluation/infer_utils.py) · [Sampler license](third_party/vlm-rlaif/LICENSE) · [OASIS paper](https://aclanthology.org/2026.acl-long.158/) · [Reference trainer](https://github.com/Collab-Multimodal-Lifelong-Learning/oasis/blob/96e67e159397b851c4cc80cae559cef00370a8cf/selection_methods/oasis.py)

```bash
uv run --locked --extra cpu python -m pytest -q
```

The viewer tests use the system FFmpeg executable. A locally available Chromium browser enables the offline interaction test. The tests cover split validation, answer-label isolation, video sampling, evidence selection, gradient ranking, real optimizer updates, replay provenance, checkpoint reload, and the result viewer.

## Acknowledgment

This research is associated with the Korean national R&D project **“Developing a Sustainable Collaborative Multi-modal Lifelong Learning Framework” (Project No. 2022-0-00113)**, conducted within the *Human-Centered AI Core Technology Development* program, supported by the Ministry of Science and ICT (MSIT) and administered by the Institute of Information & Communications Technology Planning & Evaluation (IITP).
