"""luce — train a 1%-parameter decision head on an open LLM and serve typed Choice / Score / Noul probabilities.

Package import stays torch-free (core symbols are lazy) so `luce.data` and `luce.config` load without a GPU stack.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.2.0"

__all__ = [
    "Answer", "Choice", "ChoiceAnswer", "JevLocal", "Noul", "NoulAnswer", "Question", "Score", "ScoreAnswer",
    "answers_to_dict", "question_from_dict", "DecisionEngine",
]

_CORE_SYMBOLS = {
    "Answer", "Choice", "ChoiceAnswer", "JevLocal", "Noul", "NoulAnswer", "Question", "Score", "ScoreAnswer",
    "answers_to_dict", "question_from_dict",
}


def __getattr__(name: str) -> Any:
    if name in _CORE_SYMBOLS:
        from . import core
        return getattr(core, name)
    if name == "DecisionEngine":
        from .decision import DecisionEngine
        return DecisionEngine
    raise AttributeError(f"module 'luce' has no attribute {name!r}")
