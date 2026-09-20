"""
luce.core — Jev 스타일 System One 결정 엔진의 v0 구현.

핵심 아이디어
------------
텍스트를 생성하지 않는다. state + 질문 + 선택지를 프롬프트로 만들고,
"Answer:" 다음 위치의 다음 토큰 로짓 중 선택지 라벨(A, B, C, ...)에
해당하는 것만 읽어 softmax한다. forward 한 번이 곧 응답이다.

지원 프리미티브 (Jev와 동일한 셋)
- Choice : 선택지 중 하나 고르기        -> choice, probabilities, confidence
- Score  : 순서가 있는 단계에 점수 매기기 -> score, probabilities, confidence, legend
- Noul   : 명제가 참일 확률              -> noul, confidence

v0 한계
- 선택지는 최대 26개(A~Z). 255개 지원은 v3(선택지별 스코어링 헤드)에서.
- 캘리브레이션은 temperature 하나뿐. 검증셋 기반 보정은 v1(calibrate.py)에서.
- state는 질문마다 다시 인코딩된다. prefix 캐시 공유는 v1 이후.
"""

from __future__ import annotations

import sys
import time
import inspect
import json
import math
import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version


# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

LABELS: List[str] = list(string.ascii_uppercase)  # "A" .. "Z"
MAX_OPTIONS: int = len(LABELS)

SYSTEM_HEADER: str = (
    "You are a decision engine. Read the state carefully, then answer the "
    "question by choosing exactly one option. Reply with the option letter only."
)

NOUL_YES: str = "Yes, the statement is true."
NOUL_NO: str = "No, the statement is false."


# ---------------------------------------------------------------------------
# 질문 프리미티브
# ---------------------------------------------------------------------------

@dataclass
class Choice:
    """목록에서 하나 고르기. options는 {option_id: description}."""
    prompt: str
    options: Dict[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.options, dict):
            raise TypeError("Choice.options must be a dict of {option_id: description}")
        if len(self.options) < 2:
            raise ValueError("Choice needs at least 2 options")
        if len(self.options) > MAX_OPTIONS:
            raise ValueError(f"Choice supports at most {MAX_OPTIONS} options in v0")
        for key in self.options:
            if not isinstance(key, str) or not key:
                raise ValueError("Choice option ids must be non-empty strings")


@dataclass
class Score:
    """순서가 있는 단계에 점수 매기기. levels는 낮은 것 -> 높은 것 순서."""
    prompt: str
    levels: List[str]

    def __post_init__(self) -> None:
        if not isinstance(self.levels, (list, tuple)):
            raise TypeError("Score.levels must be an ordered list of level descriptions")
        if len(self.levels) < 2:
            raise ValueError("Score needs at least 2 levels")
        if len(self.levels) > 10:
            raise ValueError("Score supports at most 10 levels (same limit as Jev)")
        self.levels = list(self.levels)


@dataclass
class Noul:
    """이 명제가 참인가? 0~1 사이 확률을 돌려준다."""
    prompt: str


Question = Union[Choice, Score, Noul]


# ---------------------------------------------------------------------------
# 답 타입
# ---------------------------------------------------------------------------

@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: Dict[str, float]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "choice",
            "choice": self.choice,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
        }


@dataclass
class ScoreAnswer:
    score: float
    probabilities: Dict[str, float]
    confidence: float
    legend: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "score",
            "score": self.score,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "legend": self.legend,
        }


@dataclass
class NoulAnswer:
    noul: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "noul",
            "noul": self.noul,
            "confidence": self.confidence,
        }


Answer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


# ---------------------------------------------------------------------------
# 프롬프트 구성 (모델 독립, 단독 테스트 가능)
# ---------------------------------------------------------------------------

def state_to_text(state: Any) -> str:
    """문자열은 그대로, dict/list 등 구조화 데이터는 JSON으로 직렬화."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def build_prompt(
    state_text: str,
    question_text: str,
    labeled_options: Sequence[Tuple[str, str]],
    extra_instruction: Optional[str] = None,
) -> str:
    """
    labeled_options: [(label, description), ...] 예: [("A", "Doors, locks"), ("B", "Heating")]
    프롬프트는 반드시 "Answer:"로 끝나야 한다. 다음 토큰 위치의 로짓을 읽기 때문.
    """
    lines: List[str] = [SYSTEM_HEADER, ""]
    if extra_instruction:
        lines.append(extra_instruction)
        lines.append("")
    lines.append("State:")
    lines.append(state_text)
    lines.append("")
    lines.append("Question:")
    lines.append(question_text)
    lines.append("")
    lines.append("Options:")
    for label, description in labeled_options:
        lines.append(f"{label}. {description}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


@dataclass
class _Instance:
    """한 번의 forward에 들어갈 프롬프트 하나와, 라벨 -> 원래 키 매핑."""
    question_name: str
    prompt: str
    label_to_key: Dict[str, str]


def _plan_instances(
    question_name: str,
    question: Question,
    state_text: str,
    n_perm: int,
    rng: random.Random,
) -> List[_Instance]:
    """
    질문 하나를 n_perm개의 프롬프트 인스턴스로 펼친다.
    - Choice, Noul: 선택지 순서를 섞어 위치 편향을 평균으로 상쇄한다.
    - Score: 순서가 의미를 가지므로 섞지 않고 인스턴스 1개만 만든다.
    """
    instances: List[_Instance] = []

    if isinstance(question, Choice):
        keys: List[str] = list(question.options.keys())
        for perm_index in range(n_perm):
            order = list(keys)
            if perm_index > 0:
                rng.shuffle(order)
            labeled = [(LABELS[i], question.options[k]) for i, k in enumerate(order)]
            label_to_key = {LABELS[i]: k for i, k in enumerate(order)}
            instances.append(_Instance(
                question_name=question_name,
                prompt=build_prompt(state_text, question.prompt, labeled),
                label_to_key=label_to_key,
            ))
        return instances

    if isinstance(question, Score):
        labeled = [(LABELS[i], level) for i, level in enumerate(question.levels)]
        label_to_key = {LABELS[i]: str(i) for i in range(len(question.levels))}
        instruction = (
            "The options form an ordered scale from lowest (first) to highest (last). "
            "Pick the level that best matches the state."
        )
        instances.append(_Instance(
            question_name=question_name,
            prompt=build_prompt(state_text, question.prompt, labeled, extra_instruction=instruction),
            label_to_key=label_to_key,
        ))
        return instances

    if isinstance(question, Noul):
        for perm_index in range(n_perm):
            if perm_index % 2 == 0:
                order = [("yes", NOUL_YES), ("no", NOUL_NO)]
            else:
                order = [("no", NOUL_NO), ("yes", NOUL_YES)]
            labeled = [(LABELS[i], desc) for i, (_, desc) in enumerate(order)]
            label_to_key = {LABELS[i]: key for i, (key, _) in enumerate(order)}
            question_text = f"Is the following statement true given the state?\n{question.prompt}"
            instances.append(_Instance(
                question_name=question_name,
                prompt=build_prompt(state_text, question_text, labeled),
                label_to_key=label_to_key,
            ))
        return instances

    raise TypeError(f"Unknown question type: {type(question).__name__}")


# ---------------------------------------------------------------------------
# 확률 계산 유틸 (모델 독립, 단독 테스트 가능)
# ---------------------------------------------------------------------------

def normalized_entropy_confidence(probs: Sequence[float]) -> float:
    """confidence = 1 - H(p) / ln(K). K=1이면 1.0."""
    k = len(probs)
    if k <= 1:
        return 1.0
    entropy = 0.0
    for p in probs:
        if p > 0.0:
            entropy -= p * math.log(p)
    return float(max(0.0, min(1.0, 1.0 - entropy / math.log(k))))


def label_logits_from_logprobs(
    logprobs_row: torch.Tensor,
    label_variant_ids: List[List[int]],
) -> torch.Tensor:
    """
    logprobs_row: (vocab,) 다음 토큰 log-softmax.
    label_variant_ids: 라벨별 토큰 id 변형 목록. 예: [[id(" A"), id("A")], [id(" B"), id("B")], ...]
    각 라벨의 로그확률은 변형들의 logsumexp. 반환 shape: (num_labels,)
    """
    out: List[torch.Tensor] = []
    for ids in label_variant_ids:
        gathered = logprobs_row[ids]
        out.append(torch.logsumexp(gathered, dim=0))
    return torch.stack(out, dim=0)


def probs_over_labels(label_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """라벨 집합 안에서만 재정규화. temperature는 캘리브레이션 훅."""
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    return torch.softmax(label_logits / temperature, dim=0)


# ---------------------------------------------------------------------------
# 디바이스 / 모델 유틸
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    """cuda > mps(Apple Silicon) > cpu 순으로 고른다."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _default_dtype(device: str) -> torch.dtype:
    """cuda/mps는 bf16, cpu는 fp32."""
    return torch.bfloat16 if device.split(":")[0] in ("cuda", "mps") else torch.float32


def _transformers_major() -> int:
    try:
        return int(transformers_version.split(".")[0])
    except (ValueError, IndexError):
        return 0


def _detect_logits_kwarg(model: Any) -> Optional[str]:
    """
    마지막 위치의 로짓만 계산하게 하는 forward 인자 이름을 찾는다.
    transformers 버전에 따라 `logits_to_keep` 또는 `num_logits_to_keep`.
    없으면 None (전체 로짓을 계산한 뒤 마지막만 쓴다).
    """
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return None
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None


# ---------------------------------------------------------------------------
# 엔진
# ---------------------------------------------------------------------------

class JevLocal:
    """
    사용 예:
        jev = JevLocal("Qwen/Qwen3-1.7B")
        answers = jev.ask(state, {"warn": Noul("..."), "area": Choice(...), "urgency": Score(...)})
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Base",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        temperature: float = 1.0,
        n_perm: int = 1,
        batch_size: int = 16,
        max_length: int = 2048,
        seed: int = 0,
        tokenizer: Any = None,
        model: Any = None,
        trust_remote_code: bool = False,
        backbone_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        if device is None:
            device = _auto_device()
        if dtype is None:
            dtype = _default_dtype(device)

        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.temperature = float(temperature)
        self.n_perm = max(1, int(n_perm))
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length)
        self._rng = random.Random(seed)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if model is None:
            # transformers 5.x는 `dtype`, 4.x는 `torch_dtype`.
            dtype_kwarg = "dtype" if _transformers_major() >= 5 else "torch_dtype"
            load_kwargs: Dict[str, Any] = {dtype_kwarg: dtype}
            if trust_remote_code:
                load_kwargs["trust_remote_code"] = True
            if backbone_overrides:
                from transformers import AutoConfig
                hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
                for key, value in backbone_overrides.items():
                    setattr(hf_config, key, value)
                load_kwargs["config"] = hf_config
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            model = model.to(device)
        self.model = model
        self.model.eval()
        # 로짓만 읽으므로 KV 캐시 불필요. (Ouro 같은 custom cache 백본은 use_cache=True 에서 깨지기도 한다)
        try:
            self.model.config.use_cache = False
        except Exception:  # noqa: BLE001
            pass

        self._logits_kwarg: Optional[str] = _detect_logits_kwarg(self.model)
        self._label_variant_ids: List[List[int]] = self._build_label_variant_ids()

    # -- 준비 ---------------------------------------------------------------

    def _build_label_variant_ids(self) -> List[List[int]]:
        """
        각 라벨에 대해 단일 토큰으로 인코딩되는 표기 변형(" A", "A")의 id를 모은다.
        토크나이저마다 "Answer:" 뒤에 공백 붙은 토큰을 예측하는지가 달라서
        두 변형을 모두 보고 logsumexp로 합친다.
        """
        variants: List[List[int]] = []
        for label in LABELS:
            ids_for_label: List[int] = []
            for text in (" " + label, label):
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if len(ids) == 1 and ids[0] not in ids_for_label:
                    ids_for_label.append(ids[0])
            if not ids_for_label:
                raise RuntimeError(f"Tokenizer cannot encode label {label!r} as a single token")
            variants.append(ids_for_label)
        return variants

    # -- 공개 API -----------------------------------------------------------

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def ask(
        self,
        state: Any,
        questions: Dict[str, Question],
        temperature: Optional[float] = None,
    ) -> Dict[str, Answer]:
        """
        state: str 또는 JSON 직렬화 가능한 객체.
        questions: {이름: Choice | Score | Noul}
        temperature: 이번 호출에만 쓸 값. None이면 엔진 기본값. 엔진 상태는 바꾸지 않는다.
        모든 질문을 한 배치로 묶어 forward하고, 질문별 답을 돌려준다.
        """
        if not questions:
            return {}
        if temperature is None:
            temperature = self.temperature
        elif temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        state_text = state_to_text(state)

        instances: List[_Instance] = []
        for name, question in questions.items():
            instances.extend(_plan_instances(name, question, state_text, self.n_perm, self._rng))

        per_instance_probs: List[Dict[str, float]] = self._score_instances(instances, float(temperature))

        # 같은 질문의 인스턴스(순서 섞은 것들)를 원래 키 기준으로 평균낸다.
        accumulated: Dict[str, Dict[str, float]] = {}
        counts: Dict[str, int] = {}
        for instance, probs_by_key in zip(instances, per_instance_probs):
            bucket = accumulated.setdefault(instance.question_name, {})
            for key, p in probs_by_key.items():
                bucket[key] = bucket.get(key, 0.0) + p
            counts[instance.question_name] = counts.get(instance.question_name, 0) + 1

        answers: Dict[str, Answer] = {}
        for name, question in questions.items():
            probs_by_key = {k: v / counts[name] for k, v in accumulated[name].items()}
            answers[name] = self._to_answer(question, probs_by_key)
        return answers

    def ask_batch(
        self,
        states: Sequence[Any],
        questions: Dict[str, Question],
        temperature: Optional[float] = None,
    ) -> List[Dict[str, Answer]]:
        """여러 state에 같은 질문 집합을 던진다. 데이터셋 평가/증류용."""
        return [self.ask(state, questions, temperature=temperature) for state in states]

    # -- 내부 ---------------------------------------------------------------

    def _score_instances(self, instances: List[_Instance], temperature: float) -> List[Dict[str, float]]:
        """인스턴스 목록을 batch_size 단위로 forward해서 키별 확률 dict 목록을 만든다."""
        results: List[Dict[str, float]] = []
        started = last_report = time.time()
        for start in range(0, len(instances), self.batch_size):
            chunk = instances[start:start + self.batch_size]
            results.extend(self._score_chunk(chunk, temperature))
            now = time.time()
            if now - last_report >= 30 and len(results) < len(instances):
                rate = len(results) / max(now - started, 1e-6)
                print(f"v0 progress: {len(results)}/{len(instances)} ({100 * len(results) / len(instances):.0f}%), {now - started:.0f}s elapsed, ETA {(len(instances) - len(results)) / rate:.0f}s", file=sys.stderr, flush=True)
                last_report = now
        return results

    @torch.inference_mode()
    def _score_chunk(self, chunk: List[_Instance], temperature: float) -> List[Dict[str, float]]:
        prompts = [inst.prompt for inst in chunk]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        # left padding이므로 position_ids를 attention_mask로부터 직접 계산한다.
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

        # 마지막 위치의 로짓만 필요하다. 지원되는 모델이면 나머지는 계산하지 않는다.
        # (Qwen 계열은 vocab이 15만이라 전체 로짓은 batch*seq*vocab로 수 GB가 된다.)
        forward_kwargs: Dict[str, Any] = {}
        if self._logits_kwarg is not None:
            forward_kwargs[self._logits_kwarg] = 1
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **forward_kwargs,
        )
        last_logits = outputs.logits[:, -1, :].float()
        logprobs = torch.log_softmax(last_logits, dim=-1)

        out: List[Dict[str, float]] = []
        for row_index, inst in enumerate(chunk):
            num_labels = len(inst.label_to_key)
            variant_ids = self._label_variant_ids[:num_labels]
            label_logits = label_logits_from_logprobs(logprobs[row_index], variant_ids)
            probs = probs_over_labels(label_logits, temperature).tolist()
            probs_by_key: Dict[str, float] = {}
            for i in range(num_labels):
                key = inst.label_to_key[LABELS[i]]
                probs_by_key[key] = float(probs[i])
            out.append(probs_by_key)
        return out

    def _to_answer(self, question: Question, probs_by_key: Dict[str, float]) -> Answer:
        if isinstance(question, Choice):
            ordered = {k: probs_by_key[k] for k in question.options.keys()}
            best = max(ordered.items(), key=lambda kv: kv[1])[0]
            return ChoiceAnswer(
                choice=best,
                probabilities=ordered,
                confidence=normalized_entropy_confidence(list(ordered.values())),
            )

        if isinstance(question, Score):
            level_probs: List[float] = [probs_by_key[str(i)] for i in range(len(question.levels))]
            expected = sum(i * p for i, p in enumerate(level_probs))
            return ScoreAnswer(
                score=float(expected),
                probabilities={level: float(p) for level, p in zip(question.levels, level_probs)},
                confidence=normalized_entropy_confidence(level_probs),
                legend=list(question.levels),
            )

        if isinstance(question, Noul):
            p_yes = probs_by_key["yes"]
            p_no = probs_by_key["no"]
            return NoulAnswer(
                noul=float(p_yes),
                confidence=normalized_entropy_confidence([p_yes, p_no]),
            )

        raise TypeError(f"Unknown question type: {type(question).__name__}")


# ---------------------------------------------------------------------------
# 직렬화 헬퍼
# ---------------------------------------------------------------------------

def answers_to_dict(answers: Dict[str, Answer]) -> Dict[str, Dict[str, Any]]:
    return {name: answer.to_dict() for name, answer in answers.items()}


def question_from_dict(spec: Dict[str, Any]) -> Question:
    """
    서버/JSON 입력용.
    {"type": "choice", "prompt": "...", "options": {"id": "desc", ...}}
    {"type": "score",  "prompt": "...", "levels": ["low", "mid", "high"]}
    {"type": "noul",   "prompt": "..."}
    """
    qtype = str(spec.get("type", "")).lower()
    prompt = spec.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("question needs a non-empty 'prompt'")
    if qtype == "choice":
        return Choice(prompt=prompt, options=spec.get("options", {}))
    if qtype == "score":
        return Score(prompt=prompt, levels=spec.get("levels", []))
    if qtype == "noul":
        return Noul(prompt=prompt)
    raise ValueError(f"unknown question type: {qtype!r} (expected choice | score | noul)")
