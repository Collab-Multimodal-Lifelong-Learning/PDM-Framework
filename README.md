# PDM Framework · Continual VideoQA

PDM answers a multiple-choice question about a video, locates an estimated evidence interval, and learns from a stream of labeled questions. Its first application uses Qwen2.5-VL on NExT-QA. Train on temporal questions, update on causal questions, then ask the original temporal questions again. The report shows which answers changed, how much earlier-task accuracy was retained, and what the updates cost.

## How PDM learns and answers

| Component | What the code does | Source |
| --- | --- | --- |
| **P · Pre-trained model** | Qwen2.5-VL scores five choices from eight sampled video frames. It chooses an answer on the full clip, then rescores four time windows to find the strongest support for that answer. Training updates rank-8 LoRA weights on the language decoder's query and value projections. | [Video scoring and evidence selection](pdm/models.py) |
| **D · Data and selection** | The importer builds clip-disjoint train, validation, and test splits. For PDM training, the selector ranks current questions by the size of their loss gradients on the last decoder block's LoRA weights and penalizes directions already represented in the batch. | [Dataset and model inputs](pdm/data.py) · [Gradient-based selection](pdm/continual.py) |
| **M · Memory** | A bounded FIFO stores labeled training questions. PDM replays one earlier example per four-example update while memory permits. The retrieval-only baseline uses up to three stored question/answer pairs as inference context. | [Memory, retrieval, and replay](pdm/continual.py) |

The controller saves the adapter and replay IDs after each task and evaluates all tasks seen so far on fixed test clips. In the verified GPU run, the causal-task update used **three new causal questions and one earlier temporal question**. The adapter weights changed. Final accuracy on the six held-out questions matched the frozen model at 3/6. [Experiment and checkpoint code](pdm/experiment.py) · [Results viewer](pdm/report.py)

## Run the application

Use Linux x86-64, Python 3.11, and uv. Run the commands from this repository. Each experiment creates a new output directory. Keep the generated HTML and its media together for playback.

**CPU demo.** One command generates 120 four-second color clips, runs all six conditions, saves the adapters, and creates a local video viewer. This fixture uses a rank-2 LoRA adapter on RGB features and exercises video decoding, backpropagation, and replay. Its model supports the two color questions shown in the viewer.

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

**Verified run (seed 42).** On an RTX PRO 6000, all six methods answered 2/3 temporal questions and 1/3 causal questions correctly after both tasks. Switching the frozen model's frame sampler changed two predictions (3/6 correct with timestamp sampling, 2/6 with frame-index sampling).

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

For a real held-out example, reload the PDM checkpoint and ask what the man in blue is holding. The verified prediction was **“camera”**, with estimated evidence from **0 to 7.249 seconds**.

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
  --manifest runs/data/nextqa.jsonl --device cuda:0 --seed 42 --out runs/nextqa-42
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

Each condition starts independently. The trained conditions receive the same optimizer-step and effective-batch budgets. The small real-video demo takes one AdamW step per task, the larger benchmark takes 16, and the CPU demo takes eight. `--conditions frozen pdm` runs a selection of methods. The default runs all six.

## Inspect the results

| Output | Contents |
| --- | --- |
| `report.html` | Video playback, answers by method and checkpoint, annotated and estimated intervals when available, accuracy and resource tables |
| `run.json`, `manifest.jsonl` | Run status, resolved configuration, environment, revisions, checksums, and the exact input records |
| `metrics.json`, `predictions.jsonl` | Accuracy for every evaluated task/checkpoint pair, individual predictions, evidence scores and labels |
| `checkpoints/<condition>/stage-<n>/` | Adapter tensors for trained methods, training-memory IDs, parameter fingerprints, selected and replayed IDs |
| `sensor_comparison.json`, `sampler_comparison.json` | The paired sensor example on CPU and the frozen frame-sampler comparison on GPU, when enabled |

The accuracy matrix has one row per checkpoint and one column per task. **Forgetting** is the earlier task's best observed accuracy minus its current accuracy. **Adaptation gain** compares the newly learned task with the frozen model at the same checkpoint. The report also includes inference latency, update time including selection, peak allocated GPU memory, and evidence IoU for samples with labeled intervals. Timing uses an untimed validation forward pass and initializes the optimizer runtime before measurement. Video caches start empty for each timed update and held-out evaluation. NExT-QA answer labels support answer accuracy. The viewer's NExT-QA evidence intervals are model estimates.

For a new clip, optional `--sensor-csv` accepts strictly increasing `timestamp_ms,value` measurements on the clip clock. The model chooses the answer from video, then ranks evidence windows using 80% video confidence and 20% normalized sensor deviation.

## Research code and tests

The four sibling collections hold 60 pinned study repositories. [Vision](https://github.com/Collab-Multimodal-Lifelong-Learning/Continual-Learning-Vision-Domain) · [Language](https://github.com/Collab-Multimodal-Lifelong-Learning/Continual-Learning-Language-Domain) · [Time series](https://github.com/Collab-Multimodal-Lifelong-Learning/Continual-Learning-Time-Series-Domain) · [Multimodal integration](https://github.com/Collab-Multimodal-Lifelong-Learning/Continual-Learning-and-Multimodal-Integration)

PDM integrates the VLM-RLAIF frame-index sampling rule as a paired comparison against timestamp sampling. Its OASIS-inspired selector uses per-example LoRA gradients at the start of each task and penalizes redundancy with previously selected examples. These adapters run inside this repository without sibling checkouts. [VLM-RLAIF source](https://github.com/Collab-Multimodal-Lifelong-Learning/vlm-rlaif/blob/610cac72c16ae3862b0d83a0dc3708d73a42a4cc/Evaluation/infer_utils.py) · [Sampler license](third_party/vlm-rlaif/LICENSE) · [OASIS paper](https://aclanthology.org/2026.acl-long.158/) · [Reference trainer](https://github.com/Collab-Multimodal-Lifelong-Learning/oasis/blob/96e67e159397b851c4cc80cae559cef00370a8cf/selection_methods/oasis.py)

```bash
uv run --locked --extra cpu python -m pytest -q
```

The viewer tests use the system FFmpeg executable. A locally available Chromium browser enables the offline interaction test. The tests cover split validation, answer-label isolation, video sampling, evidence selection, gradient ranking, real optimizer updates, replay provenance, checkpoint reload, and the result viewer.
