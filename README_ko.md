# PDM-Framework 한글 안내

[English README](README.md)

PDM-Framework는 **식당 예약을 배운 웹 에이전트가 차량 호출을 추가로 학습하고, 이전 예약 능력을 얼마나 유지하는지** 살펴보는 연속학습 프레임워크입니다. 두 모의 서비스에 자연어 요청을 보내거나, 같은 과제를 여러 체크포인트로 실행해 행동과 예약 내역을 비교할 수 있습니다.

| 서비스 | 요청 예시 | 확인할 항목 |
| --- | --- | --- |
| OpenDining (`opendining`) | 두 사람이 앉을 테이블을 가장 이른 시간으로 예약 | 식당·인원·예약 시간 |
| UDriver (`udriver`) | Local Edition에서 Golden Farmer Market까지 현금 결제로 Comfort 차량 호출 | 출발지·목적지·차종·결제 수단 |

## 에이전트와 연속학습

에이전트는 매번 요청, 화면, 번호가 붙은 페이지 요소와 최근 행동을 읽고 클릭이나 텍스트 입력 같은 명령 하나를 생성합니다. 브라우저가 명령을 실행하면 바뀐 화면을 보고 다음 행동을 고릅니다. [브라우저 실행 코드](pdm/browser.py)

PDM의 **P**는 화면과 텍스트를 읽는 사전학습 모델 Qwen2.5-VL-3B-Instruct입니다. **D**는 예약 완료를 확인한 시연 데이터로, 행동 직전의 화면과 실행한 명령을 짝지어 LoRA 어댑터를 학습합니다. **M**은 식당 행동 일부를 보관하는 메모리입니다. 같은 식당 어댑터에서 출발해 차량 행동만으로 학습하는 방법과 차량·식당 행동을 하나씩 짝지어 학습하는 방법을 비교합니다. [시연 수집](pdm/web_collect.py) · [학습 코드](pdm/web.py)

## 실행 방법

Linux x86-64에서 Python 3.11, uv, BF16 지원 CUDA GPU를 사용합니다. 저장소 루트에서 모델 환경과 Chromium을 설치한 뒤 다음 명령을 실행합니다. [설치 명령](README.md#install)

```bash
.venv/bin/pdm-agent collect --out runs/demos
.venv/bin/pdm-agent train --manifest runs/demos/manifest.jsonl \
  --out runs/agent --passes 2 --capacity 8 --seed 42
```

**직접 요청하기.** `ask`는 저장한 어댑터로 새 브라우저를 엽니다. 사이트와 요청을 바꿔 실행할 때는 새 출력 폴더를 지정합니다.

```bash
.venv/bin/pdm-agent ask --checkpoint runs/agent/replay-udriver.safetensors \
  --site udriver --out runs/my-ride \
  --goal "Book a Comfort ride from Local Edition to Golden Farmer Market and pay with cash."
```

**공통 과제로 비교하기.** `evaluate`는 학습 전후의 체크포인트에 동일한 식당·차량 과제를 주고, 사이트 기록과 에이전트의 답변을 과제별 기준으로 채점합니다.

```bash
.venv/bin/pdm-agent evaluate --run runs/agent --out runs/evaluation \
  --tasks v2.opendining-5 v2.udriver-4 --max-steps 25
```

## 실행 기록 읽기

직접 요청한 차량의 차종과 목적지는 `runs/my-ride/state.json`에서 확인합니다. 선택 과정을 살펴보려면 같은 폴더의 `trace.jsonl`에서 해당 행동을 찾아 그 단계의 화면과 함께 읽으면 됩니다. 공통 과제의 서비스별 성공 횟수와 식당 예약 능력의 변화는 `runs/evaluation/web-summary.json`에 기록됩니다. [과제별 파일 안내](README.md#inspect-an-episode)

## Acknowledgment

This research is associated with the Korean national R&D project **“Developing a Sustainable Collaborative Multi-modal Lifelong Learning Framework” (Project No. 2022-0-00113)**, conducted within the *Human-Centered AI Core Technology Development* program, supported by the Ministry of Science and ICT (MSIT) and administered by the Institute of Information & Communications Technology Planning & Evaluation (IITP).
