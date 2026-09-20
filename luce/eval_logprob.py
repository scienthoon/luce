"""
luce.eval_logprob — v0 로그확률 엔진(JevLocal)을 v2 와 같은 JSONL / 같은 지표로 평가.

같은 검증셋에서 두 결과를 나란히 놓기 위한 스크립트. 예:

    luce eval_logprob --model Qwen/Qwen2.5-3B --data data/mix/val.jsonl
    luce eval        --checkpoint checkpoints/mix --data data/mix/val.jsonl

    # 위치 편향 완화 버전 (선택지 순서 4가지 평균)
    luce eval_logprob --model Qwen/Qwen2.5-3B --data data/mix/val.jsonl --n-perm 4

    # v0 도 temperature 를 맞춰서 비교하고 싶으면
    luce eval_logprob --model Qwen/Qwen2.5-3B --data data/mix/val.jsonl --fit-temperature

같은 state 를 공유하는 레코드는 한 번의 ask() 로 묶어 배치 forward 한다.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Dict, List, Tuple

import torch

from .core import Choice, JevLocal, Noul, Question, Score
from .data import Example, describe, example_from_record, iter_jsonl
from .metrics import fit_temperature, format_reliability_table, reliability, summarize
from .train import _parse_overrides


def record_to_question(record: Dict[str, Any]) -> Question:
    qtype = str(record.get("type", "")).lower()
    if qtype == "choice":
        return Choice(record["question"], dict(record["options"]))
    if qtype == "score":
        return Score(record["question"], list(record["levels"]))
    if qtype == "noul":
        return Noul(record["question"])
    raise ValueError(f"unknown record type: {qtype!r}")


def answer_probs_in_key_order(answer: Any, example: Example) -> List[float]:
    """JevLocal 의 Answer 를 example.option_keys 순서의 확률 리스트로 변환."""
    if example.qtype == "choice":
        return [float(answer.probabilities[key]) for key in example.option_keys]
    if example.qtype == "score":
        levels = example.meta["levels"]
        return [float(answer.probabilities[level]) for level in levels]
    if example.qtype == "noul":
        return [float(answer.noul), 1.0 - float(answer.noul)]
    raise ValueError(f"unknown example type: {example.qtype!r}")


def collect(
    engine: JevLocal,
    records: List[Dict[str, Any]],
    examples: List[Example],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    반환: logits (N, Kmax) 는 log(p) (패딩 -inf), target (N, Kmax), mask (N, Kmax), types.
    softmax(log p) = p 이므로 metrics 의 로짓 인터페이스를 그대로 쓸 수 있다.
    """
    k_max = max(example.num_options for example in examples)
    n = len(examples)
    logits = torch.full((n, k_max), float("-inf"))
    target = torch.zeros((n, k_max))
    mask = torch.zeros((n, k_max), dtype=torch.bool)
    types: List[str] = []

    # 같은 state 끼리 묶는다 (순서 보존).
    groups: Dict[str, List[int]] = {}
    order: List[str] = []
    for index, record in enumerate(records):
        key = json.dumps(record.get("state", ""), ensure_ascii=False, sort_keys=True)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(index)

    done = 0
    started = time.time()
    for key in order:
        indices = groups[key]
        state = records[indices[0]].get("state", "")
        questions = {str(i): record_to_question(records[i]) for i in indices}
        answers = engine.ask(state, questions)
        for i in indices:
            example = examples[i]
            probs = answer_probs_in_key_order(answers[str(i)], example)
            k = example.num_options
            logits[i, :k] = torch.log(torch.tensor(probs).clamp(min=eps))
            target[i, :k] = torch.tensor(example.target)
            mask[i, :k] = True
        done += len(indices)
        if done % 200 < len(indices):
            elapsed = time.time() - started
            print(f"  {done}/{n} ({elapsed:.0f}s)")

    types = [example.qtype for example in examples]
    return logits, target, mask, types


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate the v0 logprob engine on a luce JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--backbone-override", action="append", default=None, help="key=value, 반복 가능 (예: total_ut_steps=2)")
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-perm", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-6, help="log(0) 방지용 확률 하한")
    parser.add_argument("--dump", default=None, help="예제별 예측 확률을 JSONL 로 저장")
    args = parser.parse_args()

    records = [record for i, record in enumerate(iter_jsonl(args.data)) if args.limit is None or i < args.limit]
    examples = [example_from_record(record, score_sigma=0.0) for record in records]
    print("data:", json.dumps(describe(examples), ensure_ascii=False))

    engine = JevLocal(
        model_name=args.model,
        device=args.device,
        temperature=args.temperature,
        n_perm=args.n_perm,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        backbone_overrides=_parse_overrides(args.backbone_override),
    )
    print("engine:", args.model, "device:", engine.device, "n_perm:", engine.n_perm, "overrides:", args.backbone_override)

    logits, target, mask, types = collect(engine, records, examples, args.eps)

    overall = summarize(logits, target, mask, temperature=1.0)
    print("overall:", json.dumps(overall))
    for qtype in sorted(set(types)):
        index = torch.tensor([i for i, t in enumerate(types) if t == qtype])
        print(f"  {qtype}: {json.dumps(summarize(logits[index], target[index], mask[index], temperature=1.0))}")
    sources = [str(example.meta.get("source", "")) for example in examples]
    for source in sorted(set(s for s in sources if s)):
        index = torch.tensor([i for i, s in enumerate(sources) if s == source])
        print(f"  [{source}]: {json.dumps(summarize(logits[index], target[index], mask[index], temperature=1.0))}")
    print("reliability:")
    print(format_reliability_table(reliability(logits, target, mask, temperature=1.0)))

    if args.fit_temperature:
        fitted = fit_temperature(logits, target, mask)
        after = summarize(logits, target, mask, temperature=fitted)
        print(f"\nfit temperature on log-probs: T={fitted:.4f}")
        print("after:", json.dumps(after))
        for qtype in sorted(set(types)):
            index = torch.tensor([i for i, t in enumerate(types) if t == qtype])
            print(f"  {qtype}: {json.dumps(summarize(logits[index], target[index], mask[index], temperature=fitted))}")
        for source in sorted(set(s for s in sources if s)):
            index = torch.tensor([i for i, s in enumerate(sources) if s == source])
            print(f"  [{source}]: {json.dumps(summarize(logits[index], target[index], mask[index], temperature=fitted))}")
        print(format_reliability_table(reliability(logits, target, mask, temperature=fitted)))

    if args.dump:
        probs = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
        with open(args.dump, "w", encoding="utf-8") as handle:
            for i, example in enumerate(examples):
                k = example.num_options
                handle.write(json.dumps({
                    "type": example.qtype,
                    "option_keys": example.option_keys,
                    "probs": [float(x) for x in probs[i, :k].tolist()],
                    "target": example.target,
                    "pred": example.option_keys[int(probs[i, :k].argmax().item())],
                    "gold": example.option_keys[example.hard_label_index],
                    "meta": {k: v for k, v in example.meta.items() if k != "levels"},
                }, ensure_ascii=False) + "\n")
        print(f"dumped predictions to {args.dump}")


if __name__ == "__main__":
    main()
