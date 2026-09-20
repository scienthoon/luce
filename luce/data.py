"""
luce.data — 결정 모델 학습/평가용 데이터 형식.

JSONL 한 줄 = 질문 하나. 세 가지 타입.

  {"state": "...", "type": "choice", "question": "Which queue?",
   "options": {"billing": "Payments, refunds", "shipping": "Delivery"},
   "label": "billing"}

  {"state": "...", "type": "score", "question": "How urgent?",
   "levels": ["Low", "Normal", "High"],
   "label": 2}                              # 0-based 레벨 인덱스

  {"state": "...", "type": "noul", "question": "The customer sounds angry.",
   "label": true}

선택: "label_probs" 로 소프트 라벨을 줄 수 있다 (증류용).
  choice: {"billing": 0.9, "shipping": 0.1}
  score : [0.1, 0.2, 0.7]
  noul  : 0.8                              # P(true)

state는 문자열 또는 JSON 객체(자동 직렬화).
"""

from __future__ import annotations

import json
import math
import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 텍스트 템플릿 (학습과 추론이 반드시 같은 템플릿을 써야 한다)
# ---------------------------------------------------------------------------

NOUL_QUESTION_PREFIX: str = "Is the following statement true given the state?"
NOUL_OPTION_TRUE: str = "Yes, the statement is true."
NOUL_OPTION_FALSE: str = "No, the statement is false."
SCORE_QUESTION_SUFFIX: str = "(Answer on an ordered scale from lowest to highest.)"
# cross-encoder: query_text + CROSS_ANSWER_PREFIX 가 접두부, " " + description 이 이어붙는 선택지 부분.
# LM prior 는 이 선택지 부분 토큰들의 로그확률을 쓴다.
CROSS_ANSWER_PREFIX: str = "\n\nAnswer:"


def state_to_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def build_query_text(state_text: str, question_text: str) -> str:
    """백본에 넣을 쿼리 텍스트. state와 질문을 함께 인코딩한다."""
    return f"State:\n{state_text}\n\nQuestion:\n{question_text}"


def build_option_text(description: str) -> str:
    """bi-encoder 용 선택지 텍스트. 질문과 독립적으로 인코딩되므로 캐시 가능하다."""
    return f"Option: {description}"


def build_cross_prefix(query_text: str) -> str:
    """cross-encoder 용 접두부 (선택지 나열 없음). 이 뒤에 ' ' + description 이 붙는다."""
    return f"{query_text}{CROSS_ANSWER_PREFIX}"


OPTION_LABELS: List[str] = list(string.ascii_uppercase)  # 라벨 continuation 모드의 최대 선택지 수 = 26


def build_cross_prefix_with_options(query_text: str, labeled_options: Sequence[Tuple[str, str]]) -> str:
    """
    cross-encoder 용 접두부 (선택지 나열 있음). 모델이 후보들을 비교하며 볼 수 있다.
    labeled_options: [(label, description), ...]  라벨은 OPTION_LABELS 순서.
    이 뒤에 ' ' + label (label 모드) 또는 ' ' + description (text 모드) 이 붙는다.
    """
    lines = [query_text, "", "Options:"]
    for label, description in labeled_options:
        lines.append(f"{label}. {description}")
    return "\n".join(lines) + CROSS_ANSWER_PREFIX


def build_cross_continuation(description: str) -> str:
    """cross-encoder 용 선택지 부분. 접두부 뒤에 이어붙는 텍스트 (앞 공백 포함)."""
    return f" {description}"


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

@dataclass
class Example:
    """모델이 직접 소비하는 형태. option_keys[i] 의 목표 확률이 target[i]."""
    query_text: str
    option_texts: List[str]
    option_keys: List[str]
    target: List[float]
    qtype: str
    meta: Dict[str, Any] = field(default_factory=dict)
    option_descriptions: List[str] = field(default_factory=list)  # 접두어 없는 원문. cross 모드가 쓴다.

    def __post_init__(self) -> None:
        if not self.option_descriptions:
            # 하위 호환: option_texts 가 "Option: X" 형식이면 X 를 복원한다.
            self.option_descriptions = [
                text[len("Option: "):] if text.startswith("Option: ") else text
                for text in self.option_texts
            ]

    @property
    def num_options(self) -> int:
        return len(self.option_texts)

    @property
    def hard_label_index(self) -> int:
        best = 0
        best_value = -1.0
        for i, value in enumerate(self.target):
            if value > best_value:
                best = i
                best_value = value
        return best


def _normalize(values: Sequence[float]) -> List[float]:
    total = float(sum(values))
    if total <= 0.0:
        raise ValueError("target distribution must have positive mass")
    return [float(v) / total for v in values]


def _score_soft_target(label_index: int, num_levels: int, sigma: float) -> List[float]:
    """
    Score 타입의 거리 인지 소프트 타겟. sigma <= 0 이면 one-hot.
    레벨이 순서를 가지므로 정답 옆 레벨에 약간의 질량을 준다.
    """
    if sigma <= 0.0:
        return [1.0 if i == label_index else 0.0 for i in range(num_levels)]
    weights = [math.exp(-0.5 * ((i - label_index) / sigma) ** 2) for i in range(num_levels)]
    return _normalize(weights)


def example_from_record(record: Dict[str, Any], score_sigma: float = 0.0) -> Example:
    """JSONL 레코드 하나를 Example로 변환한다. 학습/평가/추론이 공유하는 유일한 진입점."""
    qtype = str(record.get("type", "")).lower()
    state_text = state_to_text(record.get("state", ""))
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("record needs a non-empty 'question'")

    label_probs = record.get("label_probs")
    label = record.get("label")
    meta: Dict[str, Any] = {k: v for k, v in record.items() if k not in ("state", "question", "options", "levels", "label", "label_probs")}

    if qtype == "choice":
        options = record.get("options")
        if not isinstance(options, dict) or len(options) < 2:
            raise ValueError("choice record needs 'options' dict with >= 2 entries")
        keys = list(options.keys())
        if label_probs is not None:
            if not isinstance(label_probs, dict):
                raise ValueError("choice label_probs must be a dict")
            target = _normalize([float(label_probs.get(k, 0.0)) for k in keys])
        else:
            if label not in options:
                raise ValueError(f"choice label {label!r} not in options {keys}")
            target = [1.0 if k == label else 0.0 for k in keys]
        return Example(
            query_text=build_query_text(state_text, question),
            option_texts=[build_option_text(options[k]) for k in keys],
            option_keys=keys,
            target=target,
            qtype="choice",
            meta=meta,
            option_descriptions=[str(options[k]) for k in keys],
        )

    if qtype == "score":
        levels = record.get("levels")
        if not isinstance(levels, (list, tuple)) or len(levels) < 2:
            raise ValueError("score record needs 'levels' list with >= 2 entries")
        levels = [str(x) for x in levels]
        if label_probs is not None:
            if not isinstance(label_probs, (list, tuple)) or len(label_probs) != len(levels):
                raise ValueError("score label_probs must be a list matching levels")
            target = _normalize([float(x) for x in label_probs])
        else:
            if not isinstance(label, int) or not (0 <= label < len(levels)):
                raise ValueError(f"score label must be an int in [0, {len(levels)})")
            target = _score_soft_target(label, len(levels), score_sigma)
        question_text = f"{question}\n{SCORE_QUESTION_SUFFIX}"
        return Example(
            query_text=build_query_text(state_text, question_text),
            option_texts=[build_option_text(level) for level in levels],
            option_keys=[str(i) for i in range(len(levels))],
            target=target,
            qtype="score",
            meta={**meta, "levels": levels},
            option_descriptions=list(levels),
        )

    if qtype == "noul":
        if label_probs is not None:
            p_true = float(label_probs)
            if not (0.0 <= p_true <= 1.0):
                raise ValueError("noul label_probs must be a probability in [0, 1]")
        else:
            if isinstance(label, bool):
                p_true = 1.0 if label else 0.0
            elif isinstance(label, (int, float)) and label in (0, 1):
                p_true = float(label)
            else:
                raise ValueError("noul label must be a bool")
        question_text = f"{NOUL_QUESTION_PREFIX}\n{question}"
        return Example(
            query_text=build_query_text(state_text, question_text),
            option_texts=[build_option_text(NOUL_OPTION_TRUE), build_option_text(NOUL_OPTION_FALSE)],
            option_keys=["yes", "no"],
            target=[p_true, 1.0 - p_true],
            qtype="noul",
            meta=meta,
            option_descriptions=[NOUL_OPTION_TRUE, NOUL_OPTION_FALSE],
        )

    raise ValueError(f"unknown record type: {qtype!r} (expected choice | score | noul)")


# ---------------------------------------------------------------------------
# 입출력
# ---------------------------------------------------------------------------

def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({error})")


def load_examples(path: str, score_sigma: float = 0.0, limit: Optional[int] = None) -> List[Example]:
    examples: List[Example] = []
    for index, record in enumerate(iter_jsonl(path)):
        if limit is not None and index >= limit:
            break
        examples.append(example_from_record(record, score_sigma=score_sigma))
    if not examples:
        raise ValueError(f"no examples loaded from {path}")
    return examples


def write_jsonl(path: str, records: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_records(
    records: Sequence[Dict[str, Any]],
    val_fraction: float,
    seed: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def batches(examples: Sequence[Example], batch_size: int, shuffle: bool, seed: int) -> Iterator[List[Example]]:
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[i] for i in order[start:start + batch_size]]


def permute_example(example: Example, rng: random.Random) -> Example:
    """
    선택지 순서를 무작위로 바꾼 새 Example. keys / texts / descriptions / target 을 함께 움직인다.
    위치 편향 측정용: 고정 순서 평가와 순열 평가의 정확도 차이가 편향의 크기다.
    Score 는 순서가 의미라 그대로 둔다.
    """
    if example.qtype == "score" or example.num_options < 2:
        return example
    order = list(range(example.num_options))
    rng.shuffle(order)
    return Example(
        query_text=example.query_text,
        option_texts=[example.option_texts[i] for i in order],
        option_keys=[example.option_keys[i] for i in order],
        target=[example.target[i] for i in order],
        qtype=example.qtype,
        meta=dict(example.meta),
        option_descriptions=[example.option_descriptions[i] for i in order],
    )


def describe(examples: Sequence[Example]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    option_counts: List[int] = []
    for example in examples:
        counts[example.qtype] = counts.get(example.qtype, 0) + 1
        option_counts.append(example.num_options)
    return {
        "n": len(examples),
        "by_type": counts,
        "options_min": min(option_counts),
        "options_max": max(option_counts),
        "options_mean": sum(option_counts) / len(option_counts),
    }


def subsample_options(example: "Example", max_options: int, rng: random.Random) -> "Example":
    """학습용 음성 표본추출: choice 예제의 선택지가 max_options 보다 많으면 정답(목표 확률 최대) + 무작위 음성
    (max_options - 1) 개만 남긴다. 평가는 전체 선택지로 하므로 호출자가 학습 배치에만 적용할 것.
    소프트 타겟은 남은 선택지 위에서 다시 정규화."""
    k = example.num_options
    if example.qtype != "choice" or max_options <= 0 or k <= max_options:
        return example
    gold = max(range(k), key=lambda i: example.target[i])
    negatives = [i for i in range(k) if i != gold]
    keep = sorted([gold] + rng.sample(negatives, max_options - 1))
    target = [example.target[i] for i in keep]
    total = sum(target)
    target = [t / total for t in target] if total > 0 else [1.0 / len(keep)] * len(keep)
    return Example(
        query_text=example.query_text,
        option_texts=[example.option_texts[i] for i in keep],
        option_keys=[example.option_keys[i] for i in keep],
        target=target,
        qtype=example.qtype,
        meta=dict(example.meta, subsampled_from=k),
        option_descriptions=[example.option_descriptions[i] for i in keep],
    )


def sequence_cost(example: Example) -> int:
    """한 예제가 forward 에 넣는 시퀀스 수 (cross: 선택지 수, bi: 질문 1 + 선택지 수 와 같은 자릿수). 배치 예산의 단위."""
    return max(1, example.num_options)


def batches_by_budget(
    examples: Sequence[Example],
    budget: int,
    shuffle: bool,
    seed: int,
    bucket: int = 256,
) -> Iterator[List[Example]]:
    """
    시퀀스 수 합이 budget 을 넘지 않게 예제를 묶는다 (최소 1개). 선택지 77개짜리와 2개짜리가 섞인 데이터에서
    "가장 큰 예제에 맞춘 고정 배치 1" 대신 작은 예제는 수십 개씩, 큰 예제는 한두 개씩 들어간다.
    shuffle 이면 전체를 섞은 뒤 bucket 크기 구간 안에서만 비용순 정렬해(패딩·메모리 편차 감소) 묶는다.
    """
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), bucket):
        chunk = order[start:start + bucket]
        chunk.sort(key=lambda i: sequence_cost(examples[i]), reverse=True)
        batch: List[Example] = []
        used = 0
        for i in chunk:
            cost = sequence_cost(examples[i])
            if batch and used + cost > budget:
                yield batch
                batch, used = [], 0
            batch.append(examples[i]); used += cost
        if batch:
            yield batch
