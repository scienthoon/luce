# Luce internals (v0 → v2.2 design notes, formerly the jevlocal README)

## v0: 로그확률 엔진

Jev 스타일 System One 결정 엔진의 로컬 클론. 학습 없이, 공개 LLM의 다음 토큰 로그확률만으로
Choice / Score / Noul 세 프리미티브를 지원한다. 텍스트를 생성하지 않으므로 forward 한 번이 응답이다.

## 구조

```
je/
├── jevlocal/           # 패키지
│   ├── __init__.py     # 지연 로딩 (torch 없이 jevlocal.data 만 쓸 수 있게)
│   ├── core.py         # v0 엔진: 프리미티브, 프롬프트 빌더, 로짓 → 확률
│   ├── server.py       # FastAPI 서버 (POST /v1/ask, GET /health). JEVLOCAL_ENGINE=logprob|decision
│   ├── data.py         # v2 데이터 형식 (JSONL → Example), 텍스트 템플릿의 유일한 출처
│   ├── model.py        # v2 결정 모델: AutoModel 백본 + LoRA + PairScorer 헤드
│   ├── decision.py     # v2 추론 엔진 (DecisionEngine, v0 와 같은 ask 인터페이스)
│   ├── metrics.py      # NLL / Brier / ECE / MCE / reliability / temperature fitting
│   ├── train.py        # v2 학습 (python -m jevlocal.train)
│   └── eval.py         # v2 평가 (python -m jevlocal.eval)
├── convert.py          # 학습 JSONL 생성: synthetic | hf 프리셋 | csv
├── demo.py             # v0 동작 확인용 예제 3개
├── pyproject.toml
└── requirements.txt
```

`demo.py`와 `uvicorn` 명령은 이 디렉토리(`je/`)에서 실행한다. 다른 곳에서 쓰려면 `pip install -e .`.

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # 또는 pip install -e .
```

디바이스는 `cuda > mps > cpu` 순으로 자동 선택된다. cuda/mps는 bf16, cpu는 fp32.

- RTX 4080(16GB): `Qwen/Qwen3-1.7B` bf16이 약 3.5GB, `Qwen/Qwen2.5-7B-Instruct` bf16이 약 15GB.
  7B는 빡빡하니 1.7B~3B로 시작 권장.
- Apple Silicon(MPS): 통합 메모리라 32GB 기기에서 7B bf16도 올라간다. 단 CUDA보다 느리다.
- 모델을 바꾸려면 `--model` (demo) 또는 `JEVLOCAL_MODEL` (server).

## 라이브러리로 쓰기

```python
from jevlocal import JevLocal, Choice, Score, Noul

jev = JevLocal("Qwen/Qwen3-1.7B", n_perm=1)

answers = jev.ask(
    "The front door has been unlocked for 40 minutes and nobody is home.",
    {
        "warn":    Noul("Someone should be warned about this."),
        "area":    Choice("Which area is this about?",
                          {"security": "Doors, locks, alarms",
                           "climate": "Heating and ventilation"}),
        "urgency": Score("How urgent is it?", ["Ignore", "Today", "Right now"]),
    },
)

print(answers["warn"].noul)              # 0.0 ~ 1.0
print(answers["area"].choice)            # "security"
print(answers["area"].probabilities)     # {"security": 0.97, "climate": 0.03}
print(answers["urgency"].score)          # 0.0 ~ 2.0, 단계 사이 값 가능
```

## 서버로 쓰기

```bash
JEVLOCAL_MODEL=Qwen/Qwen3-1.7B uvicorn jevlocal.server:app --port 8000
```

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' -d '{
  "state": "The front door has been unlocked for 40 minutes and nobody is home.",
  "questions": {
    "warn":    {"type": "noul",   "prompt": "Someone should be warned about this."},
    "area":    {"type": "choice", "prompt": "Which area is this about?",
                "options": {"security": "Doors, locks, alarms", "climate": "Heating and ventilation"}},
    "urgency": {"type": "score",  "prompt": "How urgent is it?", "levels": ["Ignore", "Today", "Right now"]}
  }
}'
```

## 데모

```bash
python demo.py
python demo.py --n-perm 4      # 선택지 순서 4가지로 섞어 평균 (위치 편향 완화)
```

## 설계 요약

- 프롬프트는 `Answer:`로 끝나고, 그 다음 토큰의 로짓 중 라벨(`A`, `B`, ...) 토큰만 읽어 softmax한다.
- 라벨은 `" A"`와 `"A"` 두 토큰 변형을 logsumexp로 합쳐 토크나이저 차이에 대응한다.
- 한 state의 모든 질문은 하나의 배치로 forward된다. left padding + `position_ids` 직접 계산.
- `confidence = 1 - H(p)/ln(K)` (정규화 엔트로피 기반).
- `n_perm > 1`이면 Choice/Noul의 선택지 순서를 섞어 확률을 평균낸다. Score는 순서가 의미이므로 섞지 않는다.
- `temperature`는 v1 캘리브레이션을 위한 훅. 검증셋으로 맞추기 전까지는 1.0.
  `ask(..., temperature=...)`로 호출별 override 가능하며 엔진 상태는 바꾸지 않는다 (서버 동시 요청 안전).
- forward 시 마지막 위치의 로짓만 계산한다 (`logits_to_keep=1`). Qwen 계열은 vocab이 15만이라
  전체 로짓을 계산하면 batch×seq×vocab로 수 GB가 되기 때문.

## v0 한계 / 다음 단계

| 항목 | v0 | 다음 |
|---|---|---|
| 선택지 수 | 최대 26 (A~Z) | v3: 선택지별 스코어링 헤드로 255 |
| 캘리브레이션 | temperature 훅만 | v1: 검증셋 기반 temperature scaling + ECE 측정 |
| state 인코딩 | 질문마다 반복 | v1: prefix KV 캐시 공유 |
| 품질 | 베이스 LM 그대로 | v2: 소프트 라벨 LoRA 파인튜닝 |
| 추론 깊이 | forward 1회 | 잠재 루프(Huginn/Ouro 베이스) 실험 |

---

# v2: 결정 헤드 (LM 헤드 제거 + LoRA 학습)

v0는 LM 헤드가 `A/B/C` 토큰 확률을 뱉는 것을 읽는 방식이었다. v2는 LM 헤드를 버리고
백본을 인코더로만 쓴다. 모델은 단어를 낼 수 있는 출력층이 없고, 선택지에 점수만 매긴다.

```
query (state+question) ──► backbone ──► h_q ─┐
                                              ├─► PairScorer ──► logit per option ──► softmax
option k ──────────────► backbone ──► h_k ─┘
```

- 선택지 수 제한 없음 (A~Z 26개 제한 사라짐), 라벨 토큰 편향 없음
- 선택지 임베딩은 텍스트 키로 캐시됨 (같은 선택지 집합을 반복하면 두 번째 호출부터 비용 0)
- 손실 = 소프트 타겟 CE + Brier. 학습 끝에 검증셋으로 temperature 를 맞춰 저장
- Choice / Score / Noul 모두 같은 헤드. 차이는 `data.py` 의 텍스트 템플릿과 타겟 구성

## 데이터 형식 (JSONL, 한 줄 = 질문 하나)

```json
{"state": {...}, "type": "choice", "question": "Which queue?", "options": {"billing": "Payments", "shipping": "Delivery"}, "label": "billing"}
{"state": "...",  "type": "score",  "question": "How urgent?",  "levels": ["Low", "Normal", "High"], "label": 2}
{"state": "...",  "type": "noul",   "question": "The customer sounds angry.", "label": true}
```

`label_probs` 를 주면 소프트 라벨 (증류용): choice 는 dict, score 는 list, noul 은 P(true).

## 순서

```bash
# 1. 데이터 (다운로드 없이 먼저 돌려보려면 synthetic)
python convert.py synthetic --out data/synth --n 3000
python convert.py hf --preset klue_ynat --preset nsmc --preset arc_easy --out data/mix --limit 20000

# 2. 학습 (4080: Qwen2.5-3B, bf16, LoRA r=16, grad checkpointing)
python -m jevlocal.train --train data/synth/train.jsonl --val data/synth/val.jsonl \
    --out checkpoints/synth --backbone Qwen/Qwen2.5-3B \
    --epochs 2 --batch-size 8 --grad-accum 2 --grad-checkpointing

# --score-sigma 는 기본 0 (one-hot). 0.5 로 두면 Score 정답 옆 레벨에 질량을 나눠 주는데,
# argmax 확률 기준 ECE 가 0.003 → 0.067 로 나빠진다 (합성 검증셋, 정확도는 동일). 순서 정보를
# 확률에 남기고 싶을 때만 켜고, 그때는 Score 캘리브레이션을 기대값 오차로 따로 재야 한다.

# 3. 평가 (실제 데이터 소량으로 temperature 재보정 가능)
python -m jevlocal.eval --checkpoint checkpoints/synth --data data/synth/val.jsonl
python -m jevlocal.eval --checkpoint checkpoints/synth --data data/real_100.jsonl --fit-temperature --save

# 4. 서빙 (v0 와 같은 API)
JEVLOCAL_ENGINE=decision JEVLOCAL_CHECKPOINT=checkpoints/synth uvicorn jevlocal.server:app --port 8000
```

## 파이썬에서

```python
from jevlocal import Choice, Score, Noul
from jevlocal.decision import DecisionEngine

engine = DecisionEngine("checkpoints/synth")
answers = engine.ask(state, {"queue": Choice(...), "priority": Score(...), "angry": Noul(...)})
```

## 4080 메모리 가이드

| 백본 | bf16 가중치 | 학습 (LoRA r16, ckpt, bs8, q512) |
|---|---|---|
| Qwen2.5-1.5B | 3.1 GB | 여유 |
| Qwen2.5-3B | 6.2 GB | 권장 |
| Qwen2.5-7B | 15.2 GB | 학습 불가 (추론만) |

OOM 이면 `--batch-size 4 --grad-accum 4` 또는 `--max-query-len 384`.

## 실제 데이터 결과 (2026-09-19, Qwen2.5-3B, arc_easy 2,251 + boolq 9,427 train, val 570 + 2,000)

같은 백본, 같은 val, 둘 다 최적 temperature.

| | v0 로그확률 (n_perm 4) | v2 bi | v2.1 cross-isolated | v2.2 cross-label |
|---|---|---|---|---|
| [arc_easy] 정확도 / NLL / ECE | 93.0 / 0.207 / 0.041 | 67.0 / 0.860 / 0.080 | **94.0 / 0.184 / 0.013** | 93.7 / 0.184 / 0.024 |
| [boolq] 정확도 / NLL / ECE | 80.3 / 0.428 / 0.030 | 88.4 / 0.295 / 0.025 | 87.8 / 0.296 / 0.019 | **88.6 / 0.293 / 0.023** |
| 전체 NLL / ECE | 0.379 / 0.026 | 0.420 / 0.030 | 0.271 / **0.014** | **0.269** / 0.019 |

한 줄로: 같은 Qwen2.5-3B, 학습 파라미터 1%, GPU 한 장 1시간. 백본이 아는 것(ARC 93%)은 지키고,
모르던 것(BoolQ 80→88%)은 얻고, NLL 0.379→0.27, ECE 0.026→0.014. 자세한 실험 기록은 `EXPERIMENTS.md`.

- BoolQ(지문을 읽고 명제를 판단): 학습된 헤드가 로그확률 읽기를 +8%p 로 이긴다.
- ARC(선택지 내용 자체를 이해해야 하는 객관식): bi 헤드가 무너진다. 선택지를 질문과 독립적으로
  인코딩하면 "이 선택지가 이 질문의 답인가"를 벡터 곱셈으로만 판단해야 하고, 짧은 선택지 벡터엔
  그 정보가 없다. 백본은 답을 안다(v0 93%). 즉 데이터가 아니라 설계의 한계.

합성 데이터(규칙 기반, val 900)에서는 v2 bi 95.9% vs v0 63.9%, ECE 0.003 vs 0.105 였다.
합성에서의 격차는 헤드가 규칙을 배운 결과이므로 일반성의 증거는 위 실제 데이터 표만 본다.

## v2.1: cross-encoder + LM prior

위 결과에서 규칙이 하나 나온다.

| 선택지 종류 | 예 | 스코어러 |
|---|---|---|
| 닫힌 집합, 고정 텍스트 | 큐, 우선순위, 예/아니오 | `--scorer bi` (선택지 캐시, 두 번째 호출부터 38ms) |
| 내용이 판단 그 자체 | 객관식 정답, 후보 검증 | `--scorer cross` (질문+선택지 한 시퀀스, forward K번) |

`--scorer cross` 는 `…Question:\n…\n\nAnswer: <선택지>` 를 한 시퀀스로 인코딩하고 마지막 토큰 은닉 상태를
MLP 로 점수화한다. 접두부와 선택지를 따로 토크나이즈해 이어붙이므로 토큰 경계 문제가 없고, left padding 으로
선택지 토큰이 항상 끝에 온다.

`--lm-prior` (cross 전용) 는 같은 forward 에서 선택지 토큰들의 LM 로그확률(길이 정규화)을 뽑아 학습 가능한
`lm_scale` 로 점수에 더한다. 헤드 마지막 층은 0 초기화라 **학습 0스텝의 모델이 정확히 lm-eval-harness 식
선택지 우도 채점**이고, 학습은 거기서 올라가기만 한다. `--eval-init` 으로 학습 전 그 출발점을 확인한다.

```bash
# ARC 재대결 (cross 는 예제당 선택지 수만큼 시퀀스가 늘어 배치를 낮춘다)
python -m jevlocal.train --train data/real/train.jsonl --val data/real/val.jsonl \
    --out checkpoints/real_cross --backbone Qwen/Qwen2.5-3B \
    --scorer cross --lm-prior --eval-init \
    --epochs 2 --batch-size 4 --grad-accum 4 --grad-checkpointing
```

기대치: `init (step 0)` 의 [arc_easy] 가 v0 근처(90% 안팎)면 LM prior 와 cross 경로가 맞는 것이고,
25% 근처면 버그다. 추론 시 cross 는 선택지 캐시를 쓰지 않는다 (`DecisionEngine` 이 자동으로 끈다).

## v2.2: 비교 문맥 되돌리기 (`--options-in-prefix`, `--continuation label`)

v2.1 cross + LM prior 의 0스텝 ARC 는 68.1% 였다 (v0 93.0%). 이유는 채점 방식의 차이다. v0 는 선택지 전체를
프롬프트에 보여주고 글자(A/B/C/D)를 읽고, LM prior 는 다른 선택지를 보지 않은 채 선택지 문장 하나의 우도를 잰다.
Qwen2.5 는 시험 형식(선택지 나열 + 글자 답)에 특화되어 이 격차가 크다. 그래서 두 가지를 추가했다.

| 플래그 | 접두부 | continuation | 0스텝 의미 |
|---|---|---|---|
| (v2.1 기본) | `…Question:\n…\n\nAnswer:` | ` <선택지 원문>` | 고립 우도 채점 (lm-eval-harness 식) |
| `--options-in-prefix` | `… \n\nOptions:\nA. …\nB. …\n\nAnswer:` | ` <선택지 원문>` | 비교 문맥 + 원문 우도 |
| `--options-in-prefix --continuation label` | 위와 같음 | ` B` (라벨 글자) | **정확히 v0 라벨 로그확률** |

label 모드는 학습 0스텝이 v0 이고 헤드는 그 위의 잔차만 배운다. 라벨의 위치 편향은 학습 중 예제마다 나열 순서를
섞는 증강(`shuffle_options`, 기본 켜짐, 추론 시엔 꺼짐)으로 불변성을 배우게 해 상쇄한다. 선택지가 26개를 넘으면
자동으로 원문 continuation 으로 떨어진다.

```bash
# label 모드 (ARC 재대결 본선). init (step 0) 의 [arc_easy] 가 v0 n_perm=1 (91.6%) 근처여야 정상.
python -m jevlocal.train --train data/real/train.jsonl --val data/real/val.jsonl \
    --out checkpoints/real_label --backbone Qwen/Qwen2.5-3B \
    --scorer cross --lm-prior --options-in-prefix --continuation label --eval-init \
    --epochs 2 --batch-size 4 --grad-accum 4 --grad-checkpointing

# text+options 모드 0스텝만: "비교 문맥의 값" 과 "라벨의 값" 을 가르는 숫자
python -m jevlocal.train ... --scorer cross --lm-prior --options-in-prefix --continuation text --eval-init --epochs 0
```

## v2 → 다음

- v0 로그확률 vs v2 결정 헤드를 같은 val 로 비교 (`jevlocal.eval` 의 ECE/정확도)
- 프론티어 증류: `label_probs` 로 소프트 라벨 투입
- 백본 축소: 학습된 3B 결정 모델을 스승으로 0.5B~1B 증류
- 잠재 루프 백본 (Huginn/Ouro) 실험
