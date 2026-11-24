# PDM-Framework 한글 요약

[English README](README.md) · [실험 결과 원본](benchmarks/agent-42.json)

PDM은 **식당 예약을 익힌 웹 에이전트가 차량 호출을 배운 뒤에도 식당을 예약할 수 있는지** 실험하는 연속학습 프레임워크입니다. 에이전트는 자연어 요청과 모의 사이트의 화면을 보고 클릭·입력 같은 행동을 하나씩 실행합니다.

## 무엇을 확인했나

학습에 쓰지 않은 요청은 생일 저녁 식당 예약과 새 경로의 차량 호출, 각 한 건입니다. 시드 42에서 각 조건에 최대 25번의 행동을 허용하고 사이트의 최종 예약 기록으로 채점했습니다.

| 평가 조건 | 생일 저녁 예약 성공 | 새 경로 차량 호출 성공 |
| --- | ---: | ---: |
| 학습 전 | 0/1 | 0/1 |
| 식당 학습 직후 (같은 가중치로 요청별 두 번 실행) | 0/2 | 0/2 |
| 차량까지 순차 학습한 모델 | 0/1 | 1/1 |
| 차량 학습 중 식당 행동을 다시 학습한 모델 | 0/1 | 1/1 |

두 최종 모델 모두 **15번의 행동으로 새 경로의 차량 호출을 완료**했습니다. 생일 저녁 예약에는 모두 실패해, 이번 실험에서 메모리가 새 예약 요청의 성공률을 높였다는 근거는 얻지 못했습니다.

학습 때 본 “첫 번째 식당을 예약해 달라”는 요청도 메모리 모델에 두 번 주었습니다. 두 번 모두 예약은 접수됐지만 요청대로 첫 번째 식당을 예약한 것은 한 번이었습니다. 다른 한 번은 두 번째 식당을 골랐습니다.

## 어떻게 학습하나

- **P · 사전학습 모델**은 Qwen2.5-VL-3B-Instruct입니다. 화면·접근성 트리·요청·행동 이력을 입력으로 사용해 LoRA 어댑터가 다음 행동을 예측합니다.
- **D · 데이터**는 완료 기록을 확인한 시연 4건입니다. 식당 예약 2건에서 행동 12개, 차량 호출 2건에서 행동 29개를 수집했습니다.
- **M · 메모리**에는 식당 행동 8개를 보관합니다. 같은 식당 모델에서 출발해 한쪽은 차량 행동만, 다른 쪽은 차량 행동과 과거 식당 행동을 짝지어 각각 58회 갱신합니다.

## 실행과 결과 확인

저장소 루트에서 두 Python 환경과 Chromium을 설치한 뒤 실행합니다. 학습과 추론에는 BF16을 지원하는 CUDA GPU가 필요합니다. [설치 명령](README.md#install)

```bash
.venv/bin/pdm-agent collect --out runs/agent-demos
.venv/bin/pdm-agent train --manifest runs/agent-demos/manifest.jsonl \
  --out runs/agent-42 --passes 2 --capacity 8 --seed 42
.venv/bin/pdm-agent evaluate --run runs/agent-42 --out runs/evaluation-42 --max-steps 25
```

과제별 성공 여부는 `runs/evaluation-42/web-summary.json`에서 확인합니다. 저장한 모델에 직접 요청하려면 `pdm-agent ask`를 사용합니다. 실행 폴더의 `trace.jsonl`에는 모델이 수행한 행동이, `state.json`에는 사이트의 최종 상태가 기록됩니다. [저장한 모델로 식당 예약하기](README.md#run-an-instruction)

## Acknowledgment

This research is associated with the Korean national R&D project **“Developing a Sustainable Collaborative Multi-modal Lifelong Learning Framework” (Project No. 2022-0-00113)**, conducted within the *Human-Centered AI Core Technology Development* program, supported by the Ministry of Science and ICT (MSIT) and administered by the Institute of Information & Communications Technology Planning & Evaluation (IITP).
