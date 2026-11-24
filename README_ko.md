# PDM-Framework 한글 요약

[English README](README.md)

PDM은 **식당 예약을 익힌 웹 에이전트가 차량 호출을 배운 뒤에도 예약을 수행할 수 있는지** 실험하는 연속학습 프레임워크입니다. 사용자가 자연어로 요청하면 모델이 모의 사이트의 화면과 접근성 트리를 읽고 클릭·입력 등 다음 행동을 하나씩 생성합니다.

| 구성 | 구현 |
| --- | --- |
| **P · 사전학습 모델** | Qwen2.5-VL-3B-Instruct의 LoRA 어댑터를 갱신해 다음 브라우저 행동을 예측합니다. |
| **D · 데이터** | 식당 예약 2건과 차량 호출 2건을 시연합니다. 사이트의 완료 기록을 확인한 뒤 각 행동과 그 직전의 관측값을 학습 예시로 저장합니다. |
| **M · 메모리** | 식당 행동을 최대 8개 보관합니다. 차량 호출을 학습할 때 과거 식당 행동을 함께 사용하는 방법과 차량 행동만 사용하는 방법을 같은 업데이트 횟수로 비교합니다. |

## 확인한 결과

시드 42에서 식당 행동 12개와 차량 행동 29개를 수집했습니다. 저장한 메모리 모델에 학습 때 본 요청을 새 브라우저에서 두 번 주자, 두 번 모두 6번의 행동으로 예약을 접수했습니다. **요청한 첫 번째 식당을 예약한 경우는 1/2회**였습니다. 다른 한 번은 두 번째 식당을 예약했습니다.

학습에 없던 생일 저녁 예약과 새 경로 차량 호출을 다섯 조건에서 실행한 결과는 **10회 중 2회 성공**이었습니다. 순차 학습 모델과 메모리 재학습 모델이 각각 15번의 행동으로 새 경로의 차량 호출을 완료했습니다. 생일 저녁 예약은 다섯 조건 모두 실패했습니다. 이 두 과제에서 메모리 재학습의 성공률 우위는 확인되지 않았습니다. [실측 결과와 모델 해시](benchmarks/agent-42.json)

## 사용법

- **설치**에는 Python 3.11, uv, BF16을 지원하는 CUDA GPU, Chromium이 필요합니다. 모델 환경과 브라우저 환경을 각각 설치합니다. [설치 명령](README.md#install)
- **학습·평가**는 시연 수집, 어댑터 학습, 미학습 과제 평가 순서로 실행합니다. `training.json`에는 갱신별 학습 손실과 메모리 항목이, `web-summary.json`에는 실제 과제 성공 횟수와 채점 건수가 기록됩니다. [실행 명령과 결과 경로](README.md#collect-train-and-evaluate)
- **추론**은 `pdm-agent ask`에 저장한 어댑터, 사이트, 요청을 주면 새 브라우저에서 실행됩니다. 결과 폴더의 `trace.jsonl`로 행동을, `state.json`으로 사이트의 예약 기록을 확인할 수 있습니다. [식당 예약 예시](README.md#run-an-instruction)

## Acknowledgment

This research is associated with the Korean national R&D project **“Developing a Sustainable Collaborative Multi-modal Lifelong Learning Framework” (Project No. 2022-0-00113)**, conducted within the *Human-Centered AI Core Technology Development* program, supported by the Ministry of Science and ICT (MSIT) and administered by the Institute of Information & Communications Technology Planning & Evaluation (IITP).
