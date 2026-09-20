"""
luce.decision — 학습된 결정 모델로 추론. v0 JevLocal 과 같은 ask() 인터페이스.

    from luce.decision import DecisionEngine
    from luce import Choice, Score, Noul

    engine = DecisionEngine("checkpoints/v2")
    answers = engine.ask(state, {"queue": Choice(...), "priority": Score(...), "angry": Noul(...)})

선택지 임베딩은 텍스트를 키로 캐시된다. 같은 선택지 집합을 반복해서 쓰면 두 번째 호출부터
선택지 인코딩 비용이 0 이다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .core import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    normalized_entropy_confidence,
)
from .data import Example, example_from_record, state_to_text
from .metrics import probs_from_logits
from .model import DecisionModel


def example_from_question(state_text: str, question: Question) -> Example:
    """
    추론용 Example. 레코드를 만들어 data.example_from_record 에 그대로 넣는다.
    그래서 학습과 추론의 텍스트 템플릿(bi 의 option_texts, cross 의 option_descriptions)이
    한 곳(data.py)에만 존재한다. 타겟은 균등 분포 자리표시자이며 추론에서 쓰이지 않는다.
    """
    if isinstance(question, Choice):
        record = {
            "state": state_text, "type": "choice", "question": question.prompt,
            "options": dict(question.options),
            "label_probs": {key: 1.0 for key in question.options},
        }
    elif isinstance(question, Score):
        record = {
            "state": state_text, "type": "score", "question": question.prompt,
            "levels": list(question.levels),
            "label_probs": [1.0] * len(question.levels),
        }
    elif isinstance(question, Noul):
        record = {"state": state_text, "type": "noul", "question": question.prompt, "label_probs": 0.5}
    else:
        raise TypeError(f"Unknown question type: {type(question).__name__}")
    return example_from_record(record)


class DecisionEngine:
    def __init__(
        self,
        checkpoint_dir: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        batch_size: int = 16,
        temperature: Optional[float] = None,
    ) -> None:
        self.model = DecisionModel.load(checkpoint_dir, device=device, dtype=dtype)
        self.model_name = f"{self.model.config.backbone} + decision head ({checkpoint_dir})"
        self.device = self.model.device_name
        self.batch_size = max(1, int(batch_size))
        self.temperature = float(temperature) if temperature is not None else float(self.model.config.temperature)
        self.temperatures: Dict[str, float] = dict(getattr(self.model.config, "temperatures", {}) or {})  # 타입별 T (있으면 우선)
        self.n_perm = 1  # 서버 응답 호환용. 결정 헤드는 라벨 위치가 없어 순열 평균이 필요 없다.
        # 선택지 캐시는 bi 모드에서만 의미가 있다. cross 는 선택지가 질문에 묶여 매번 인코딩한다.
        self._option_cache: Optional[Dict[str, torch.Tensor]] = {} if self.model.config.scorer == "bi" else None
        self._autocast_enabled = self.device == "cuda"
        self._autocast_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def clear_option_cache(self) -> None:
        if self._option_cache is not None:
            self._option_cache.clear()

    @torch.inference_mode()
    def ask(
        self,
        state: Any,
        questions: Dict[str, Question],
        temperature: Optional[float] = None,
    ) -> Dict[str, Answer]:
        """temperature: 이번 호출에만 쓸 값. None 이면 체크포인트에 저장된 값. 엔진 상태는 바꾸지 않는다."""
        if not questions:
            return {}
        if temperature is None:
            temperature = self.temperature
        elif temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        state_text = state_to_text(state)
        names = list(questions.keys())
        examples = [example_from_question(state_text, questions[name]) for name in names]

        probs_per_example: List[List[float]] = []
        for start in range(0, len(examples), self.batch_size):
            chunk = examples[start:start + self.batch_size]
            with torch.autocast(
                device_type="cuda" if self.device == "cuda" else "cpu",
                dtype=self._autocast_dtype,
                enabled=self._autocast_enabled,
            ):
                logits, _, mask = self.model.forward_examples(chunk, option_cache=self._option_cache)
            for i, example in enumerate(chunk):
                t_i = float(temperature) if temperature != self.temperature else float(self.temperatures.get(example.qtype, self.temperature))
                probs_i = probs_from_logits(logits[i:i + 1].float(), mask[i:i + 1], temperature=t_i)
                probs_per_example.append(probs_i[0, :example.num_options].tolist())

        answers: Dict[str, Answer] = {}
        for name, example, probs in zip(names, examples, probs_per_example):
            answers[name] = self._to_answer(questions[name], example, probs)
        return answers

    def ask_batch(
        self,
        states: List[Any],
        questions: Dict[str, Question],
        temperature: Optional[float] = None,
    ) -> List[Dict[str, Answer]]:
        return [self.ask(state, questions, temperature=temperature) for state in states]

    @staticmethod
    def _to_answer(question: Question, example: Example, probs: List[float]) -> Answer:
        if isinstance(question, Choice):
            ordered = {key: float(p) for key, p in zip(example.option_keys, probs)}
            best = max(ordered.items(), key=lambda kv: kv[1])[0]
            return ChoiceAnswer(
                choice=best,
                probabilities=ordered,
                confidence=normalized_entropy_confidence(list(ordered.values())),
            )
        if isinstance(question, Score):
            levels = example.meta["levels"]
            expected = sum(i * p for i, p in enumerate(probs))
            return ScoreAnswer(
                score=float(expected),
                probabilities={level: float(p) for level, p in zip(levels, probs)},
                confidence=normalized_entropy_confidence(probs),
                legend=list(levels),
            )
        if isinstance(question, Noul):
            p_yes = float(probs[0])
            p_no = float(probs[1])
            return NoulAnswer(
                noul=p_yes,
                confidence=normalized_entropy_confidence([p_yes, p_no]),
            )
        raise TypeError(f"Unknown question type: {type(question).__name__}")
