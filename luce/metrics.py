"""
luce.metrics — 결정 모델의 정확도와 캘리브레이션 지표.

모든 함수는 (N, K) 로짓 텐서와 (N, K) 타겟 확률 텐서, 그리고 (N, K) 유효 마스크를 받는다.
패딩된 선택지는 mask=False 이고 로짓은 -inf 로 채워져 있다고 가정한다.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Sequence

import torch


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = logits.masked_fill(~mask, float("-inf"))
    return torch.log_softmax(logits, dim=-1)


def probs_from_logits(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    logits = (logits / temperature).masked_fill(~mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def nll(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """소프트 타겟 cross-entropy. 패딩 위치는 target=0 이므로 기여하지 않는다."""
    log_probs = masked_log_softmax(logits / temperature, mask)
    log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    return -(target * log_probs).sum(dim=-1).mean()


def brier(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    probs = probs_from_logits(logits, mask, temperature)
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    return ((probs - target) ** 2).sum(dim=-1).mean()


def accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    pred = masked_logits.argmax(dim=-1)
    gold = target.masked_fill(~mask, -1.0).argmax(dim=-1)
    return (pred == gold).float().mean()


def reliability(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    n_bins: int = 15,
    temperature: float = 1.0,
) -> Dict[str, object]:
    """
    confidence = max prob, correct = (argmax pred == argmax target).
    반환: ece, mce, bins(각 bin의 count / avg_conf / avg_acc).
    """
    probs = probs_from_logits(logits, mask, temperature)
    confidence, pred = probs.max(dim=-1)
    gold = target.masked_fill(~mask, -1.0).argmax(dim=-1)
    correct = (pred == gold).float()

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    total = confidence.numel()
    ece = torch.zeros((), device=logits.device)
    mce = torch.zeros((), device=logits.device)
    bins: List[Dict[str, float]] = []
    for b in range(n_bins):
        lower = edges[b]
        upper = edges[b + 1]
        if b == 0:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence > lower) & (confidence <= upper)
        count = int(in_bin.sum().item())
        if count == 0:
            bins.append({"lower": float(lower), "upper": float(upper), "count": 0, "avg_conf": 0.0, "avg_acc": 0.0})
            continue
        avg_conf = confidence[in_bin].mean()
        avg_acc = correct[in_bin].mean()
        gap = (avg_conf - avg_acc).abs()
        ece = ece + gap * (count / total)
        mce = torch.maximum(mce, gap)
        bins.append({
            "lower": float(lower),
            "upper": float(upper),
            "count": count,
            "avg_conf": float(avg_conf),
            "avg_acc": float(avg_acc),
        })
    return {"ece": float(ece), "mce": float(mce), "bins": bins, "n": int(total)}


def fit_temperature(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    t_min: float = 0.05,
    t_max: float = 20.0,
    grid_size: int = 200,
    refine_iters: int = 60,
) -> float:
    """
    검증셋 NLL 을 최소화하는 스칼라 temperature.

    1차원 문제이므로 그래디언트 옵티마이저 대신 경계가 있는 탐색을 쓴다:
    log 공간 격자 탐색으로 대략의 최소를 찾고, 그 주변을 황금분할 탐색으로 다듬는다.
    (LBFGS + 라인서치는 손실이 평평할 때 스텝이 발산해 overflow 로 죽는다.)
    """
    logits = logits.detach().float()
    target = target.detach().float()
    with torch.no_grad():
        def loss_at(temperature: float) -> float:
            return float(nll(logits, target, mask, temperature=temperature))

        log_min, log_max = math.log(t_min), math.log(t_max)
        grid = [math.exp(log_min + (log_max - log_min) * i / (grid_size - 1)) for i in range(grid_size)]
        losses = [loss_at(t) for t in grid]
        best_index = min(range(grid_size), key=lambda i: losses[i])

        # 격자 최소점 양옆을 브래킷으로 잡고 log 공간에서 황금분할 탐색
        lo = math.log(grid[max(0, best_index - 1)])
        hi = math.log(grid[min(grid_size - 1, best_index + 1)])
        golden = (math.sqrt(5.0) - 1.0) / 2.0
        x1 = hi - golden * (hi - lo)
        x2 = lo + golden * (hi - lo)
        f1 = loss_at(math.exp(x1))
        f2 = loss_at(math.exp(x2))
        for _ in range(refine_iters):
            if f1 < f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - golden * (hi - lo)
                f1 = loss_at(math.exp(x1))
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + golden * (hi - lo)
                f2 = loss_at(math.exp(x2))
        best_log = x1 if f1 < f2 else x2
        best = math.exp(best_log)
        # 다듬은 값이 격자 최소보다 나쁘면(수치 잡음) 격자 값을 쓴다
        if loss_at(best) > losses[best_index]:
            best = grid[best_index]
        best = float(min(max(best, t_min), t_max))
        if best <= t_min * 1.001 or best >= t_max * 0.999:
            # 경계에 붙었다 = NLL 이 T 에 대해 단조. T→∞ 는 "확신도에 정보가 없다(균등 분포가 최선)",
            # T→0 은 "argmax 가 거의 항상 맞고 더 뾰족할수록 좋다". 둘 다 데이터/모델 진단 신호다.
            print(
                f"warning: fitted temperature hit the search bound (T={best:.4f}, range [{t_min}, {t_max}]). "
                "The model's confidences carry little usable information on this validation set; "
                "treat the calibration as unreliable.",
                file=sys.stderr,
            )
        return best


def summarize(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    n_bins: int = 15,
) -> Dict[str, float]:
    with torch.no_grad():
        rel = reliability(logits, target, mask, n_bins=n_bins, temperature=temperature)
        return {
            "n": float(rel["n"]),
            "accuracy": float(accuracy(logits, target, mask)),
            "nll": float(nll(logits, target, mask, temperature=temperature)),
            "brier": float(brier(logits, target, mask, temperature=temperature)),
            "ece": float(rel["ece"]),
            "mce": float(rel["mce"]),
            "temperature": float(temperature),
        }


def format_reliability_table(rel: Dict[str, object]) -> str:
    lines = ["  bin            count   avg_conf   avg_acc    gap"]
    for b in rel["bins"]:  # type: ignore[index]
        if b["count"] == 0:
            continue
        gap = b["avg_conf"] - b["avg_acc"]
        lines.append(f"  {b['lower']:.2f}-{b['upper']:.2f}   {b['count']:6d}   {b['avg_conf']:8.3f}   {b['avg_acc']:7.3f}   {gap:+.3f}")
    lines.append(f"  ECE={rel['ece']:.4f}  MCE={rel['mce']:.4f}  n={rel['n']}")
    return "\n".join(lines)


def selective_risk(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    thresholds: Sequence[float] = (0.8, 0.9, 0.95),
) -> List[Dict[str, float]]:
    """
    보정 후 max-prob 가 임계값 이상인 항목만 자동 처리한다고 할 때의 coverage 와 그 구간 정확도.
    사용자가 "0.9 넘으면 자동, 아니면 사람" 의 임계값을 고르는 데 쓰는 표.
    """
    probs = probs_from_logits(logits, mask, temperature)
    confidence, pred = probs.max(dim=-1)
    gold = target.masked_fill(~mask, -1.0).argmax(dim=-1)
    correct = (pred == gold).float()
    rows: List[Dict[str, float]] = []
    n = confidence.numel()
    for t in thresholds:
        covered = confidence >= t
        c = int(covered.sum())
        rows.append({
            "threshold": float(t),
            "coverage": c / n if n else 0.0,
            "accuracy_covered": float(correct[covered].mean()) if c else float("nan"),
            "accuracy_rest": float(correct[~covered].mean()) if n - c else float("nan"),
            "n_covered": c,
        })
    return rows


def format_selective_risk(rows: Sequence[Dict[str, float]]) -> str:
    lines = ["  threshold   coverage   acc(covered)   acc(rest)   n_covered"]
    for r in rows:
        ac = f"{r['accuracy_covered']:.3f}" if r["accuracy_covered"] == r["accuracy_covered"] else "  n/a"
        ar = f"{r['accuracy_rest']:.3f}" if r["accuracy_rest"] == r["accuracy_rest"] else "  n/a"
        lines.append(f"  {r['threshold']:.2f}        {r['coverage']*100:5.1f}%      {ac:>8s}      {ar:>8s}   {int(r['n_covered']):6d}")
    return "\n".join(lines)
