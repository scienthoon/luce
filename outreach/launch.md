# Launch drafts

No X/Threads account needed. Channels that work without one:
- **GitHub / HF** (lists, tracker, issues on the benchmarks we reused) — already done by the agent, see `EXPERIMENTS.md` E17 and the links below.
- **Show HN** — the one worth doing by hand: an account takes a minute, no followers needed, submissions are ranked by the post not the poster. Title + first comment below.
- **GeekNews (news.hada.io)** — Korean HN, link submission only.
- **Letting others carry it** — list maintainers, the jev-exploration ledger and newsletters scrape this space; being listed with results is what gets picked up.

Done already (agent, no personal account):
[awesome-jev PR #65](https://github.com/yibie/awesome-jev/pull/65) · [awesome-typesafe PR #61](https://github.com/AbdelStark/awesome-typesafe/pull/61) · [jev-exploration #10 comment](https://github.com/SamuelSacco/jev-exploration/issues/10) · [jev-phishing-bench #1](https://github.com/anisselbd/jev-phishing-bench/issues/1) · [system-one-open #1](https://github.com/mithalouni/system-one-open/issues/1) · [NanoJev #9](https://github.com/TianyuCodings/NanoJev/issues/9) · tracker Space PR #5 comment

Canonical link: https://github.com/scienthoon/luce · Demo (no GPU): https://huggingface.co/spaces/noscienthoon/luce-live-triage · Checkpoint: https://huggingface.co/noscienthoon/ouro-2.6b-decision-lora

## Show HN — 15 minutes

1. Account: https://news.ycombinator.com/login (username + password, no e-mail needed).
2. Submit: https://news.ycombinator.com/submit — URL `https://github.com/scienthoon/luce`, title below, text field empty.
3. Immediately post the first comment below as a reply to your own post (HN convention: context goes in a comment, not the title).
4. Best window: Tue–Thu 08:00–10:00 US Eastern = 21:00–23:00 KST. Stay around for an hour or two to answer.

### Title (≤ 80 chars)

**Show HN: Luce – describe a decision task, get a calibrated model; 91% on rule-generated gold vs Jev's 75%**

### First comment (post as a reply to your own submission)

Luce is an open recipe (Apache-2.0) for TypeSafe-Jev-style decision models: a sentence about the task → an LLM writes the
training data → LoRA + a small decision head on Qwen3-4B-Base → typed probabilities (choice / ordered score / boolean),
one forward pass, no generation, with a temperature fit and a review threshold.

What we measured, same test items for us and for Jev (zero-shot via Vercel AI Gateway):
- Support tickets with an organisational rule, training data written by an LLM from one paragraph: 91.1% vs 75.1
- Phishing e-mails (the jev-phishing-bench set): 97.4% vs 62.6, ECE 0.010 vs 0.154
- GitHub issue kind / priority (kubernetes maintainer labels): 86.9 / 41.1 vs 84.7 / 37.5
- Maze: all three questions fail. `risk` and `death` collapse to a constant answer (85.3 / 86.1 = the majority
  baseline to the digit), `safest move` lands below majority. Do not cite the maze numbers as a win — Jev scores below
  a constant predictor there (65.6) and so do we.

The honest summary is in the README: where the label is a function of the input, a few hundred to a few thousand labels
beat the zero-shot model by 20–35 points; where it is not (policy-assigned priority, three-step lookahead), training
adds little. Trains on a 12 GB card in 20–70 minutes. Every number has its experiment log entry.

Demo in the browser, no GPU (replays recorded model outputs): https://huggingface.co/spaces/noscienthoon/luce-live-triage

Likely questions, answered up front: `pip install luce` is not live yet, use `pip install "git+https://github.com/scienthoon/luce"`. The comparison is not apples to apples — Jev is zero-shot, we train — which is the point: the README states it on every row. The GitHub-issue task is where training barely helps, and that is in the table too.

## r/LocalLLaMA

**Luce: open recipe for Jev-style calibrated decision models on a 12 GB card (Qwen3-4B + LoRA), with the numbers vs Jev**

Same body as Show HN, plus: replay demo runs in the browser without a GPU; `luce serve` gives you `/v1/ask` and a review queue.

## GeekNews (news.hada.io)

**Luce — 문장 하나로 Jev 같은 결정 모델 만들기 (오픈소스, 12GB GPU)**
TypeSafe Jev(텍스트 생성 없이 확률로 답하는 모델)를 자기 과제용으로 만드는 레시피. 과제 설명 → LLM이 학습 데이터 생성 → Qwen3-4B에 LoRA 학습 → 보정된 확률 서빙.
규칙 티켓: LLM 생성 3,000개로 학습해 정답 2,964개에서 91.1% (다수 답 45.1, Jev 0샷 75.1). 피싱 97.4 (다수 답 50.0) vs 62.6. 안 되는 경우도 표에 그대로: GitHub priority 는 표본 안 잡음, 미로는 세 질문 모두 상수 예측이라 학습 효과 없음.
