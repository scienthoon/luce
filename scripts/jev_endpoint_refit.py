"""Jev 확률의 끝점 처리가 refit temperature 에 얼마나 영향을 주는지 재계산한다.

배경 (jev-exploration#10, SamuelSacco). Jev 의 확률은 0.01 격자로 양자화돼 있고, 끝점 사용이 타입마다 다르다:
Choice 는 70.4% 가 정확히 0 / 20.4% 가 정확히 1, Score 는 47.2% / 16.7%, Noul 은 0% / 0% (0.01~0.98 로 clamp).

temperature 는 확률이 (0,1) 안쪽일 때만 제대로 정의된다. 정답에 정확히 0 을 준 항목은 어떤 T 로도 살릴 수 없고,
로그를 바닥 eps 로 자르면 그 한 항목이 −log(eps) 만큼의 손실을 넣는다 (eps=1e-6 이면 13.8). 그런 항목 몇 개가
T 를 위로 끌어올릴 수 있으므로, 우리가 보고한 "Choice/Score T 3.3 (과신) vs Noul T 0.66 (과소확신)" 부호 반전이
실제 과신인지 끝점 처리의 산물인지 구분해야 한다.

    python scripts/jev_endpoint_refit.py --dump logs/jev_synth.jsonl

바닥값을 1e-6(기존), 0.005(격자 절반), 0.01(격자 한 칸)로 바꿔 T 를 다시 맞추고, 타입별 끝점 비율과
"정답에 정확히 0" 비율을 함께 낸다. 정답-0 항목을 뺀 T 도 같이 내서 그 항목들이 T 를 얼마나 끌어올리는지 본다.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from typing import Dict, List, Sequence

import numpy as np


def load(path: str) -> List[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def softmax_t(logp: np.ndarray, t: float) -> np.ndarray:
    z = logp / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logp: np.ndarray, target: np.ndarray, t: float) -> float:
    p = softmax_t(logp, t)
    return float(-(target * np.log(np.clip(p, 1e-300, 1.0))).sum(1).mean())


def fit_t(logp: np.ndarray, target: np.ndarray, t_min: float = 0.05, t_max: float = 20.0, grid: int = 200) -> float:
    """luce.metrics.fit_temperature 와 같은 방식: log 격자 탐색 후 황금분할."""
    lo, hi = math.log(t_min), math.log(t_max)
    ts = [math.exp(lo + (hi - lo) * i / (grid - 1)) for i in range(grid)]
    losses = [nll(logp, target, t) for t in ts]
    b = int(np.argmin(losses))
    a, c = math.log(ts[max(0, b - 1)]), math.log(ts[min(grid - 1, b + 1)])
    phi = (math.sqrt(5) - 1) / 2
    x1, x2 = c - phi * (c - a), a + phi * (c - a)
    f1, f2 = nll(logp, target, math.exp(x1)), nll(logp, target, math.exp(x2))
    for _ in range(60):
        if f1 < f2:
            c, x2, f2 = x2, x1, f1
            x1 = c - phi * (c - a); f1 = nll(logp, target, math.exp(x1))
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (c - a); f2 = nll(logp, target, math.exp(x2))
    return math.exp((a + c) / 2)


def ece(p: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    conf = p.max(1)
    correct = (p.argmax(1) == target.argmax(1)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def analyse(rows: Sequence[dict], floors: Sequence[float]) -> None:
    by_type: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in rows:
        by_type[r.get("type", "?")].append(r)

    print(f"{'type':8s}{'items':>7s}{'values':>8s}{'exact 0':>9s}{'exact 1':>9s}{'0 on correct':>14s}{'min>0':>8s}{'max<1':>8s}")
    print("-" * 71)
    packs = {}
    for qtype, rs in sorted(by_type.items()):
        P = np.array([r["probs"] for r in rs], dtype=float)
        T = np.array([r["target"] for r in rs], dtype=float)
        T = T / np.clip(T.sum(1, keepdims=True), 1e-12, None)
        gold = T.argmax(1)
        zero_on_correct = (P[np.arange(len(P)), gold] == 0.0)
        interior = P[(P > 0) & (P < 1)]
        print(f"{qtype:8s}{len(rs):7d}{P.size:8d}{100*(P==0).mean():8.1f}%{100*(P==1).mean():8.1f}%"
              f"{100*zero_on_correct.mean():13.1f}%{(interior.min() if interior.size else float('nan')):8.2f}"
              f"{(interior.max() if interior.size else float('nan')):8.2f}")
        packs[qtype] = (P, T, zero_on_correct)

    for floor in floors:
        print(f"\n=== floor eps = {floor:g}  ({'current scorer' if floor == 1e-6 else 'half a grid step' if floor == 0.005 else 'one grid step'})")
        print(f"{'type':8s}{'T':>8s}{'NLL@1':>9s}{'NLL@T':>9s}{'ECE@1':>8s}{'ECE@T':>8s}{'acc':>8s}"
              f"{'T w/o 0-on-correct':>21s}{'kept':>7s}")
        print("-" * 88)
        for qtype, (P, T, zoc) in packs.items():
            logp = np.log(np.clip(P, floor, 1.0))
            t = fit_t(logp, T)
            p1, pt = softmax_t(logp, 1.0), softmax_t(logp, t)
            acc = float((P.argmax(1) == T.argmax(1)).mean())
            if (~zoc).sum() >= 10:
                t_sub = fit_t(logp[~zoc], T[~zoc])
                sub = f"{t_sub:21.3f}"
            else:
                sub = f"{'n/a':>21s}"
            print(f"{qtype:8s}{t:8.3f}{nll(logp,T,1.0):9.3f}{nll(logp,T,t):9.3f}{ece(p1,T):8.3f}{ece(pt,T):8.3f}"
                  f"{100*acc:7.1f}%{sub}{100*(~zoc).mean():6.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, nargs="+")
    ap.add_argument("--floors", default="1e-6,0.005,0.01")
    a = ap.parse_args()
    rows: List[dict] = []
    for d in a.dump:
        r = load(d)
        print(f"loaded {len(r):5d} rows from {d}")
        rows.extend(r)
    print()
    analyse(rows, [float(x) for x in a.floors.split(",")])


if __name__ == "__main__":
    main()
