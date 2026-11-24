# PDM-Framework 한글 요약

PDM의 대표 응용은 **식당 예약을 학습한 뒤 차량 호출을 배우는 웹 에이전트**입니다. 사용자가 자연어로 요청하면 모델이 브라우저 화면과 접근성 트리를 읽고 다음 행동 하나를 생성합니다. REAL-Eval은 이 행동을 모의 사이트에서 실행하고 예약이 실제로 접수되었는지 공식 REAL v2 기준으로 채점합니다. 기존 영상 질의응답·근거 구간 추정·영상 검색도 별도 응용으로 실행할 수 있습니다. [VideoQA 실행법과 결과](VIDEOQA.md)

| 구성 | 구현 |
| --- | --- |
| **P · 사전학습 모델** | Qwen2.5-VL-3B-Instruct에 rank-8 LoRA를 붙여 클릭·텍스트 입력·응답을 생성하도록 학습합니다. |
| **D · 학습 데이터** | 별도로 작성한 식당 예약 2건과 차량 호출 2건을 브라우저에서 수행하고 완료 화면을 확인했습니다. 각 행동 직전의 화면·지시·이력과 실행한 행동을 짝지어 총 39개 학습 예시를 만들었습니다. |
| **M · 메모리** | 식당 행동 10개에서 8개를 표본으로 보관합니다. 차량 호출 학습 시 메모리 조건은 현재 행동과 과거 식당 행동을 하나씩 사용합니다. 비교 대상인 순차 LoRA는 같은 횟수만큼 차량 행동 두 개로 갱신합니다. |

## 확인한 결과

시드 42에서 두 학습 방식 모두 식당 20회, 차량 58회씩 가중치를 갱신했습니다. 최종 메모리 모델은 학습 때 본 점심 예약 요청을 새 브라우저에서 다시 수행해 **5번의 행동으로 Vintage Bites의 오후 12시 30분 2인 예약을 완료**했습니다. 모의 사이트의 예약 레코드로 확인했습니다.

처음 보는 공식 과제는 생일 저녁 식당 예약과 새 경로의 차량 호출, 각 1건이었습니다. 다섯 평가 조건에서 두 과제를 각각 실행한 결과는 **0/10 성공**이었습니다. 최종 메모리 모델의 시연 행동 손실은 식당 0.075, 차량 0.143이었고, 순차 LoRA는 각각 0.186, 0.111이었습니다. 이 결과는 학습·추론이 연결되고 이전 시연을 재현할 수 있음을 보여줍니다. 새 요청에서의 성공이나 메모리의 성능 우위는 이 실험에서 확인되지 않았습니다. [과제별 점수와 실험 설정](benchmarks/real-v2-web.json)

## 실행

REAL-Eval은 비공개 서브모듈이므로 원본 저장소를 읽을 권한이 필요합니다. [영문 README의 환경 설치 절차](README.md#reproduce-the-web-experiment)를 마친 뒤 저장소 루트에서 실행합니다.

```bash
REAL-Eval/.venv/bin/python -m pdm.web_collect --output runs/web-training-demo

.venv/bin/python -m pdm.web train --manifest runs/web-training-demo/manifest.jsonl \
  --out runs/web-train-42 --passes 2 --capacity 8 --seed 42

.venv/bin/python -m pdm.web evaluate --run runs/web-train-42 --out runs/web-real-42 \
  --real-eval REAL-Eval/.venv/bin/real-eval --max-steps 25
```

평가는 사전학습 모델, 식당 학습 직후 모델, 차량 학습까지 마친 두 모델의 결과를 비교합니다. 식당 학습 직후에는 양쪽 조건이 가중치를 공유합니다. 마지막 메모리 모델의 추론 과정은 `runs/web-real-42/real-eval/replay-udriver/index.html`에서 확인할 수 있습니다. 새 지시로 추론하려면 `pdm.web serve`로 저장한 어댑터를 띄우고 `real-eval ask`로 요청합니다. [추론 명령 예시](README.md#reproduce-the-web-experiment)

## Acknowledgment

This research is associated with the Korean national R&D project **“Developing a Sustainable Collaborative Multi-modal Lifelong Learning Framework” (Project No. 2022-0-00113)**, conducted within the *Human-Centered AI Core Technology Development* program, supported by the Ministry of Science and ICT (MSIT) and administered by the Institute of Information & Communications Technology Planning & Evaluation (IITP).
