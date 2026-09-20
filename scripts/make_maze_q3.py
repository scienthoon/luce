"""E19: 미로 safe_move 를 "최적 행동 분포" 대신 **방향별 독립 생존확률**로 학습하도록 데이터를 바꾼다.

배경 (NanoJev#9, TianyuCodings 제안). 기존 choice 질문의 타깃은 "최적 수들에 균등" 이라 각 방향이 얼마나 안전한지와
방향 사이 격차가 지워진다. 그래서 모델이 지도를 읽는 대신 주변 분포(north/south)를 외웠다. 대신 방향마다 독립 이진
질문을 만든다.

    Q3(s,a) = P(첫 수를 a 로 둔 뒤 남은 두 수를 균등 무작위로 둘 때 세 수 모두 생존)

- 방향 간 정규화하지 않는다: 여러 방향이 동시에 안전할 수 있다.
- 추론은 네 값 중 최댓값 방향. 지표는 생존 regret = max_a Q3(s,a) − Q3(s, 고른 방향).
- 일관성 검사: P(3수 내 사망) = 1 − mean_a Q3(s,a).
- 값은 이미 각 행 meta 의 `survival_probability_by_first_move` 에 정확히 들어 있다(유한 열거, 몬테카를로 아님).

    python scripts/make_maze_q3.py --in data/four_tasks/maze --out data/four_tasks/maze_q3
    python scripts/make_maze_q3.py --in ... --out ... --augment    # 회전·반사 8배 (방향·좌표 함께 변환)
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

DIRS = ["north", "east", "south", "west"]
# 지도를 시계방향 90도 돌리면 북→동, 동→남, 남→서, 서→북
ROT = {"north": "east", "east": "south", "south": "west", "west": "north"}
# 좌우 반사(열 뒤집기)면 동↔서
FLIP = {"north": "north", "south": "south", "east": "west", "west": "east"}

# noul 은 "명제가 참일 확률" 한 값을 받는다 (options/label 없음, label_probs 는 스칼라).
QUESTION = (
    "The agent's first move is {d}, and every move after that independently chooses north/east/south/west "
    "uniformly. The agent survives all three moves. A wall or boundary collision kills; reaching the goal "
    "stops safely."
)


def rotate_map(grid: List[str]) -> List[str]:
    """문자 격자를 시계방향 90도."""
    n = len(grid)
    return ["".join(grid[n - 1 - c][r] for c in range(n)) for r in range(len(grid[0]))]


def flip_map(grid: List[str]) -> List[str]:
    return [row[::-1] for row in grid]


def transform_state(state: Dict[str, Any], rots: int, flip: bool) -> Dict[str, Any]:
    """local_map 과 방향 표기를 함께 변환. position/goal 은 전역 좌표라 손대지 않고 지운다(로컬 뷰만 쓴다)."""
    grid = state["local_map"].split("\n")
    for _ in range(rots):
        grid = rotate_map(grid)
    if flip:
        grid = flip_map(grid)
    out = dict(state)
    out["local_map"] = "\n".join(grid)
    out.pop("position", None)
    out.pop("goal", None)
    return out


def map_dir(d: str, rots: int, flip: bool) -> str:
    for _ in range(rots):
        d = ROT[d]
    return FLIP[d] if flip else d


def q3_rows(row: Dict[str, Any], rots: int = 0, flip: bool = False, tag: str = "") -> List[Dict[str, Any]]:
    meta = row.get("meta") or {}
    surv = meta.get("survival_probability_by_first_move")
    if not surv:
        return []
    state = transform_state(row["state"], rots, flip) if (rots or flip) else row["state"]
    out = []
    for d in DIRS:
        d_new = map_dir(d, rots, flip)
        q = float(surv[d])
        out.append({
            "state": {**state, "first_move": d_new},
            "type": "noul",
            "question": QUESTION.format(d=d_new),
            "source": "NanoJev_maps_three_step_q3" + (f"_{tag}" if tag else ""),
            "meta": {
                "state_id": meta.get("state_id", "") + (f"|{tag}" if tag else ""),
                "source_group_id": meta.get("source_group_id"),      # 분할은 맵 단위로 유지된다
                "source_map_seed": meta.get("source_map_seed"),
                "question_name": "q3_survival",
                "first_move": d_new,
                "q3": q,
                "exact_death_probability": meta.get("exact_death_probability"),
                "survival_probability_by_first_move": {map_dir(k, rots, flip): v for k, v in surv.items()},
                "target_method": meta.get("target_method"),
                "augmentation": tag or "identity",
            },
            "label_probs": q,        # noul soft target: 명제("세 수 모두 생존")가 참일 확률
        })
    return out


def convert(path_in: str, path_out: str, augment: str, limit_states: int) -> None:
    rows = [json.loads(l) for l in open(path_in, encoding="utf-8") if l.strip()]
    states = [r for r in rows if r["type"] == "choice"]
    if limit_states:
        states = states[:limit_states]
    variants = [(0, False, "")]
    if augment == "rot":        # 회전만 4배 — 이것만으로 네 방향이 정답이 되는 빈도가 균등해진다
        variants = [(r, False, ("" if r == 0 else f"rot{r}")) for r in range(4)]
    elif augment == "all":      # 회전 x 반사 8배
        variants = [(r, f, ("" if (r == 0 and not f) else f"rot{r}{'f' if f else ''}")) for r in range(4) for f in (False, True)]
    out = []
    for r in states:
        for rots, flip, tag in variants:
            out.extend(q3_rows(r, rots, flip, tag))
    with open(path_out, "w", encoding="utf-8") as h:
        for r in out:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    qs = [r["label_probs"] for r in out]
    mean_q = sum(qs) / max(len(qs), 1)
    zero = sum(1 for q in qs if q == 0.0)
    print(f"{os.path.basename(path_in):18s} {len(states):5d} states x {len(variants)} variants x 4 dirs = {len(out):6d} rows"
          f"  | mean Q3 {mean_q:.4f}  P(true)>0.5 {100*sum(1 for q in qs if q > 0.5)/max(len(qs),1):5.1f}%"
          f"  dead-on-arrival {100*zero/max(len(qs),1):5.1f}%  -> {path_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True); ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--augment", choices=["none", "rot", "all"], default="none",
                    help="학습 분할에만: rot = 회전 4배, all = 회전x반사 8배 (방향·지도 함께 변환)")
    ap.add_argument("--limit-train-states", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    for split in ("train", "val", "calibration", "test", "ood"):
        p = os.path.join(a.src, f"{split}.jsonl")
        if not os.path.exists(p):
            continue
        convert(p, os.path.join(a.dst, f"{split}.jsonl"),
                augment=(a.augment if split == "train" else "none"),
                limit_states=(a.limit_train_states if split == "train" else 0))


if __name__ == "__main__":
    main()
