"""E19: Q3 예측 덤프를 상태 단위로 묶어 행동 선택과 생존 regret 을 잰다.

`luce.eval --dump` 이 낸 Q3 행들(상태당 네 방향)을 모아서:
  - 고른 행동 = argmax_a P(yes | s, a)  (방향 간 정규화하지 않는다)
  - 생존 regret = max_a Q3(s,a) − Q3(s, 고른 행동)      ← 주 지표
  - tie-aware 정확도: 최적 수가 여럿이면 그중 아무거나 맞으면 정답 (46% 상태가 동점)
  - 기준선: 항상 같은 방향 / 무작위(동점 인지) / 방향별 Q3 를 학습셋 평균으로 예측
  - 일관성: P(3수 내 사망) = 1 − mean_a Q3(s,a) 를 예측값으로 계산해 정답과 비교
  - 확률 자체의 질: 네 값에 대한 Brier / NLL, 학습셋에서 맞춘 상수와 비교

    python scripts/eval_maze_q3.py --dump logs/q3_test.jsonl --train data/four_tasks/maze_q3/train.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from typing import Dict, List

import numpy as np

DIRS = ["north", "east", "south", "west"]


def load(path: str) -> List[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def group_states(rows: List[dict]) -> Dict[str, dict]:
    """state_id 로 묶고, 각 방향의 예측 P(yes) 와 정답 Q3 를 모은다."""
    by: Dict[str, dict] = collections.defaultdict(lambda: {"pred": {}, "true": {}})
    for r in rows:
        meta = r.get("meta") or {}
        if "first_move" not in meta and isinstance(meta.get("meta"), dict):
            meta = meta["meta"]                      # dump 은 원본 meta 를 한 겹 안에 넣는다
        sid, d = meta.get("state_id"), meta.get("first_move")
        if sid is None or d is None:
            continue
        keys = r["option_keys"]
        p_yes = float(r["probs"][keys.index("yes")])
        by[sid]["pred"][d] = p_yes
        by[sid]["true"][d] = float(meta.get("q3", 0.0))
        by[sid]["death"] = meta.get("exact_death_probability")
    return {k: v for k, v in by.items() if len(v["pred"]) == 4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--train", default=None)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    states = group_states(load(a.dump))
    n = len(states)
    print(f"states with all four directions: {n}")
    if not n:
        return

    P = np.array([[s["pred"][d] for d in DIRS] for s in states.values()])
    T = np.array([[s["true"][d] for d in DIRS] for s in states.values()])
    best = T.max(1)
    chosen = P.argmax(1)
    got = T[np.arange(n), chosen]
    regret = best - got

    # 학습셋에서 맞춘 방향별 상수 (그 방향의 평균 Q3) — "지도를 안 읽는" 최선
    const = None
    if a.train:
        acc = collections.defaultdict(list)
        for r in load(a.train):
            m = r.get("meta") or {}
            if "first_move" not in m and isinstance(m.get("meta"), dict):
                m = m["meta"]
            if m.get("first_move"):
                acc[m["first_move"]].append(float(m.get("q3", 0.0)))
        const = np.array([np.mean(acc[d]) if acc[d] else 0.0 for d in DIRS])

    def regret_of(idx: np.ndarray) -> float:
        return float((best - T[np.arange(n), idx]).mean())

    ties = (T == best[:, None])
    tie_aware = float(ties[np.arange(n), chosen].mean())
    n_tied = int((ties.sum(1) > 1).mean() * n)

    print(f"\n{'selector':34s}{'survival regret':>17s}{'tie-aware acc':>15s}")
    print("-" * 66)
    print(f"{'model (argmax Q3)':34s}{regret.mean():17.4f}{100*tie_aware:14.2f}%")
    for j, d in enumerate(DIRS):
        idx = np.full(n, j)
        print(f"{'always ' + d:34s}{regret_of(idx):17.4f}{100*float(ties[np.arange(n), idx].mean()):14.2f}%")
    rng = np.random.default_rng(0); idx = rng.integers(0, 4, n)
    print(f"{'uniform random':34s}{regret_of(idx):17.4f}{100*float(ties[np.arange(n), idx].mean()):14.2f}%")
    if const is not None:
        idx = np.full(n, int(const.argmax()))
        print(f"{'train-mean Q3 (always ' + DIRS[int(const.argmax())] + ')':34s}{regret_of(idx):17.4f}{100*float(ties[np.arange(n), idx].mean()):14.2f}%")
    print(f"{'oracle':34s}{0.0:17.4f}{100.0:14.2f}%")
    print(f"\ntied-optimal states: {n_tied}/{n} ({100*n_tied/n:.1f}%)   regret quartiles: "
          f"{np.round(np.percentile(regret, [50, 75, 90, 100]), 4).tolist()} (median/75/90/max)")

    # 확률 자체의 질 (네 방향을 독립 이진으로)
    def brier(p): return float(((p - T) ** 2).mean())
    def nll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(T * np.log(p) + (1 - T) * np.log(1 - p)).mean())
    print(f"\n{'probability quality':34s}{'Brier':>10s}{'NLL':>10s}")
    print("-" * 54)
    print(f"{'model':34s}{brier(P):10.4f}{nll(P):10.4f}")
    if const is not None:
        C = np.tile(const, (n, 1))
        print(f"{'train-mean Q3 per direction':34s}{brier(C):10.4f}{nll(C):10.4f}")
    G = np.full_like(P, float(T.mean()))
    print(f"{'global mean Q3':34s}{brier(G):10.4f}{nll(G):10.4f}")

    # 일관성: P(death within 3) = 1 - mean_a Q3
    dt = np.array([s["death"] if s["death"] is not None else np.nan for s in states.values()])
    ok = ~np.isnan(dt)
    if ok.any():
        pred_death = 1 - P.mean(1); true_death = 1 - T.mean(1)
        print(f"\nconsistency  P(death) = 1 - mean_a Q3:")
        print(f"  from targets vs stored exact_death_probability : max|diff| {np.nanmax(np.abs(true_death[ok] - dt[ok])):.6f}")
        print(f"  from model   vs stored exact_death_probability : MAE {np.nanmean(np.abs(pred_death[ok] - dt[ok])):.4f}"
              f"   (always-predicting the train mean would give MAE {np.nanmean(np.abs(np.nanmean(dt[ok]) - dt[ok])):.4f})")

    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as h:
            json.dump({"n": n, "regret_model": float(regret.mean()), "tie_aware_acc": tie_aware,
                       "brier": brier(P), "nll": nll(P),
                       "regret_always": {d: regret_of(np.full(n, j)) for j, d in enumerate(DIRS)}}, h, indent=1)
        print(f"\n-> {a.json_out}")


if __name__ == "__main__":
    main()
