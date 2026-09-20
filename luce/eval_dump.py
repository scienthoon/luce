"""
luce.eval_dump — 예측 덤프 JSONL 에서 지표 계산. 모델과 무관하다.

v0(eval_logprob --dump), 결정 헤드(eval --dump), Jev(scripts/jev_eval.mjs) 의 덤프를
전부 같은 형식으로 받아 같은 지표를 낸다. 그래서 세 모델을 한 표에 놓을 수 있다.

덤프 레코드:
  {"type", "option_keys", "probs", "target", "pred", "gold", "source"?, "confidence"?, "error"?}
  ("error" 가 있는 행은 제외하고 개수만 보고한다)

    luce eval_dump --dump logs/jev_csqa.jsonl
    luce eval_dump --dump logs/jev_csqa.jsonl --fit-temperature      # 과신 정도 진단
    luce eval_dump --dump logs/jev_csqa.jsonl --confidence-reliability  # TypeSafe confidence 필드 자체의 캘리브레이션
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List

import torch

from .metrics import fit_temperature, format_reliability_table, reliability, summarize


def load_dump(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_tensors(rows: List[Dict[str, Any]], eps: float):
    k_max = max(len(row["probs"]) for row in rows)
    n = len(rows)
    logits = torch.full((n, k_max), float("-inf"))
    target = torch.zeros((n, k_max))
    mask = torch.zeros((n, k_max), dtype=torch.bool)
    for i, row in enumerate(rows):
        probs = torch.tensor([float(p) for p in row["probs"]])
        k = probs.numel()
        logits[i, :k] = torch.log(probs.clamp(min=eps))
        target[i, :k] = torch.tensor([float(t) for t in row["target"]])
        mask[i, :k] = True
    return logits, target, mask


def confidence_reliability(rows: List[Dict[str, Any]], n_bins: int = 15) -> Dict[str, Any]:
    """
    TypeSafe 가 따로 주는 confidence 필드가 정답률을 예측하는지 본다.
    confidence 를 bin 으로 나누고 각 bin 의 정답률(pred == gold)을 잰다.
    max-prob 기반 ECE 와는 별개의 질문: "이 confidence 통계를 임계값으로 써도 되는가".
    """
    pairs = []
    for row in rows:
        c = row.get("confidence")
        if c is None:
            continue
        try:
            c = float(c)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(c):
            continue
        pairs.append((c, 1.0 if row["pred"] == row["gold"] else 0.0))
    if not pairs:
        return {"n": 0, "bins": []}
    confidence = torch.tensor([p[0] for p in pairs])
    correct = torch.tensor([p[1] for p in pairs])
    low = float(confidence.min())
    high = float(confidence.max())
    if high <= low:
        high = low + 1e-6
    edges = torch.linspace(low, high, n_bins + 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        lower = edges[b]
        upper = edges[b + 1]
        in_bin = (confidence >= lower) & (confidence <= upper) if b == 0 else (confidence > lower) & (confidence <= upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        avg_conf = float(confidence[in_bin].mean())
        avg_acc = float(correct[in_bin].mean())
        ece += abs(avg_conf - avg_acc) * count / len(pairs)
        bins.append({"lower": float(lower), "upper": float(upper), "count": count, "avg_conf": avg_conf, "avg_acc": avg_acc})
    return {"n": len(pairs), "range": [low, high], "ece_if_probability": ece, "bins": bins}


def main() -> None:
    parser = argparse.ArgumentParser(description="metrics from a prediction dump")
    parser.add_argument("--dump", required=True)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--confidence-reliability", action="store_true", help="덤프의 confidence 필드(예: TypeSafe) 자체의 신뢰도 표")
    args = parser.parse_args()

    all_rows = load_dump(args.dump)
    rows = [row for row in all_rows if "error" not in row and row.get("probs")]
    errors = len(all_rows) - len(rows)
    print(f"dump: {args.dump}  rows: {len(rows)}  errors: {errors}")
    if not rows:
        return

    logits, target, mask = to_tensors(rows, args.eps)
    types = [row["type"] for row in rows]
    sources = [str(row.get("source") or "") for row in rows]

    print("overall:", json.dumps(summarize(logits, target, mask, temperature=1.0)))
    for qtype in sorted(set(types)):
        index = torch.tensor([i for i, t in enumerate(types) if t == qtype])
        print(f"  {qtype}: {json.dumps(summarize(logits[index], target[index], mask[index], temperature=1.0))}")
    for source in sorted(set(s for s in sources if s)):
        index = torch.tensor([i for i, s in enumerate(sources) if s == source])
        print(f"  [{source}]: {json.dumps(summarize(logits[index], target[index], mask[index], temperature=1.0))}")
    print("reliability (max-prob):")
    print(format_reliability_table(reliability(logits, target, mask, temperature=1.0)))

    if args.fit_temperature:
        fitted = fit_temperature(logits, target, mask)
        print(f"\nrefit temperature: T={fitted:.4f}  (1.0 = 이미 캘리브레이션됨, >1 과신, <1 과소확신)")
        print("after:", json.dumps(summarize(logits, target, mask, temperature=fitted)))
        for source in sorted(set(s for s in sources if s)):
            index = torch.tensor([i for i, s in enumerate(sources) if s == source])
            print(f"  [{source}]: {json.dumps(summarize(logits[index], target[index], mask[index], temperature=fitted))}")
        print(format_reliability_table(reliability(logits, target, mask, temperature=fitted)))

    if args.confidence_reliability:
        rel = confidence_reliability(rows)
        print(f"\nconfidence field reliability: n={rel['n']}")
        if rel["n"] > 0:
            print(f"  range: {rel['range'][0]:.3f} .. {rel['range'][1]:.3f}   ECE-if-read-as-probability: {rel['ece_if_probability']:.4f}")
            print("  bin              count   avg_conf   avg_acc")
            for b in rel["bins"]:
                print(f"  {b['lower']:.3f}-{b['upper']:.3f}   {b['count']:6d}   {b['avg_conf']:8.3f}   {b['avg_acc']:7.3f}")


if __name__ == "__main__":
    main()
