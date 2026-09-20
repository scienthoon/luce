# Launch drafts (paste-ready; the account holder decides whether to post)

Canonical link: https://github.com/scienthoon/luce · Demo (no GPU): https://huggingface.co/spaces/noscienthoon/luce-live-triage · Checkpoint: https://huggingface.co/noscienthoon/ouro-2.6b-decision-lora

## Show HN (title ≤ 80 chars)

**Show HN: Luce – describe a decision task, get a calibrated model; 91% on rule-generated gold vs Jev's 75%**

Luce is an open recipe (Apache-2.0) for TypeSafe-Jev-style decision models: a sentence about the task → an LLM writes the
training data → LoRA + a small decision head on Qwen3-4B-Base → typed probabilities (choice / ordered score / boolean),
one forward pass, no generation, with a temperature fit and a review threshold.

What we measured, same test items for us and for Jev (zero-shot via Vercel AI Gateway):
- Support tickets with an organisational rule, training data written by an LLM from one paragraph: 91.1% vs 75.1
- Phishing e-mails (the jev-phishing-bench set): 97.4% vs 62.6, ECE 0.010 vs 0.154
- GitHub issue kind / priority (kubernetes maintainer labels): 86.9 / 41.1 vs 84.7 / 37.5
- Maze risk level (exact probabilities): 85.3 vs 65.6; the "safest move" question neither of us solves

The honest summary is in the README: where the label is a function of the input, a few hundred to a few thousand labels
beat the zero-shot model by 20–35 points; where it is not (policy-assigned priority, three-step lookahead), training
adds little. Trains on a 12 GB card in 20–70 minutes. Every number has its experiment log entry.

## Threads / X reply (under the Jev use-case thread)

같은 걸 오픈으로: 과제를 문장으로 쓰면 LLM이 데이터를 만들고, 4B 모델에 LoRA+결정 헤드를 얹어 확률로 답하는 레시피(Luce).
규칙 기반 티켓에서 LLM이 쓴 3,000개로 학습 → 정답 2,964개에서 91.1% (Jev 0샷 75.1). 12GB 카드 40분.
데모(GPU 없이 재생): huggingface.co/spaces/noscienthoon/luce-live-triage · 코드: github.com/scienthoon/luce

## r/LocalLLaMA

**Luce: open recipe for Jev-style calibrated decision models on a 12 GB card (Qwen3-4B + LoRA), with the numbers vs Jev**

Same body as Show HN, plus: replay demo runs in the browser without a GPU; `luce serve` gives you `/v1/ask` and a review queue.

## GeekNews (news.hada.io)

**Luce — 문장 하나로 Jev 같은 결정 모델 만들기 (오픈소스, 12GB GPU)**
TypeSafe Jev(텍스트 생성 없이 확률로 답하는 모델)를 자기 과제용으로 만드는 레시피. 과제 설명 → LLM이 학습 데이터 생성 → Qwen3-4B에 LoRA 학습 → 보정된 확률 서빙.
규칙 티켓: LLM 생성 3,000개로 학습해 정답 2,964개에서 91.1% (Jev 0샷 75.1). 피싱 97.4 vs 62.6. 안 되는 경우(정책성 라벨, 3수 미로)도 표에 그대로.
