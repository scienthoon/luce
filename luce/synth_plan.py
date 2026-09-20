"""Deterministic allocation for shared-state synthesis; no model dependencies."""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping


def allocate(weights: Mapping[str, float], total: int) -> Dict[str, int]:
    """Largest-remainder allocation: preserve the requested total, including zero weights."""
    scale = sum(weights.values())
    if total < 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("allocation needs a nonnegative count and positive finite weights")
    exact = {key: total * weight / scale for key, weight in weights.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    order = sorted(weights, key=lambda key: exact[key] - result[key], reverse=True)
    for key in order[:total - sum(result.values())]:
        result[key] += 1
    return result


def grid_cells(grid: Mapping[str, Any], total: int, rng: random.Random) -> List[Dict[str, str]]:
    cells: List[Dict[str, str]] = [{} for _ in range(total)]
    for axis, values in grid.items():
        weights = values if isinstance(values, dict) else dict.fromkeys(values, 1.0)
        counts = allocate(weights, total)
        column = [value for value, count in counts.items() for _ in range(count)]
        rng.shuffle(column)
        for cell, value in zip(cells, column):
            cell[axis] = value
    return cells


def new_slots(counts: Dict[str, int], grid: Mapping[str, Any], ratios: Mapping[str, Any],
              val_fraction: float, rng: random.Random) -> List[Dict[str, Any]]:
    """All questions share one pool; smaller quotas choose a subset of the same states.

    Holdout membership is chosen before generation. Only training slots receive desired
    answers, so answer-ratio sampling cannot rebalance the validation set.
    """
    val_counts = {name: int(round(count * val_fraction)) for name, count in counts.items()}
    train_counts = {name: count - val_counts[name] for name, count in counts.items()}
    n_val = max(val_counts.values(), default=0)
    n_train = max(train_counts.values(), default=0)
    slots = [{"id": i, "split": "val" if i < n_val else "train", "wanted": [], "intents": {}}
             for i in range(n_val + n_train)]
    for name in counts:
        for split, available, count in (("val", list(range(n_val)), val_counts[name]),
                                         ("train", list(range(n_val, len(slots))), train_counts[name])):
            rng.shuffle(available)
            chosen = available[:count]
            for index in chosen:
                slots[index]["wanted"].append(name)
            if split == "train" and name in ratios:
                label_counts = allocate(ratios[name], count)
                labels = [label for label, n in label_counts.items() for _ in range(n)]
                rng.shuffle(labels)
                for index, label in zip(chosen, labels):
                    slots[index]["intents"][name] = label
    for slot, cell in zip(slots, grid_cells(grid, len(slots), rng)):
        slot["grid"] = cell
    rng.shuffle(slots)
    return slots


def label_key(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def select_rows(rows: List[dict], total: int | None, ratios: Mapping[str, float] | None,
                rng: random.Random) -> List[dict]:
    """Sample imported, already-labeled rows without changing Teacher answers.

    An explicit total is strict. Without one, a forced ratio downsamples to the
    largest feasible total; neither mode duplicates rows to invent a base rate.
    """
    pool = list(rows)
    rng.shuffle(pool)
    if ratios is None:
        wanted = len(pool) if total is None else total
        if wanted > len(pool):
            raise ValueError(f"requested {wanted} labels, but only {len(pool)} are available")
        return pool[:wanted]
    buckets = {key: [] for key in ratios}
    for row in pool:
        key = label_key(row["label"])
        if key in buckets:
            buckets[key].append(row)
    wanted = len(pool) if total is None else total
    if total is None:
        scale = sum(ratios.values())
        # A quota allocated by largest remainder is at least floor(n * weight / scale).
        # Hence these bounds include the largest feasible n without assuming monotonic
        # allocation (largest-remainder allocation can have the Alabama paradox).
        wanted = min([wanted] + [max(0, math.ceil((len(buckets[key]) + 1) * scale / weight) - 1)
                                 for key, weight in ratios.items() if weight > 0])
    while True:
        quotas = allocate(ratios, wanted)
        missing = {key: count - len(buckets[key]) for key, count in quotas.items() if count > len(buckets[key])}
        if not missing:
            chosen = [row for key, count in quotas.items() for row in buckets[key][:count]]
            rng.shuffle(chosen)
            return chosen
        if total is not None:
            raise ValueError(f"not enough Teacher labels for requested answer ratios: {missing}")
        wanted -= 1
