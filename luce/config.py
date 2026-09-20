"""
luce.config — `luce.yaml` schema, loader, teacher spec parsing, and the `mode: auto` rule.

Everything here is torch-free so `luce init` / `luce synth --dry-run` work on a laptop without the GPU stack.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

DEFAULT_CONFIG_PATH = "luce.yaml"
MAX_LABEL_OPTIONS = 26  # A..Z; beyond this the label continuation cannot be used

TEACHER_EXAMPLES = """teacher is required. Pass --teacher "URL|MODEL" or set synth.teacher in luce.yaml. Examples:

  # Vercel AI Gateway  (key: AI_GATEWAY_API_KEY)
  --teacher "https://ai-gateway.vercel.sh/v1|anthropic/claude-sonnet-5"
  # OpenAI             (key: OPENAI_API_KEY)
  --teacher "https://api.openai.com/v1|gpt-5"
  # Ollama, local      (no key)
  --teacher "http://localhost:11434/v1|qwen2.5:7b"

The key is read from LUCE_TEACHER_API_KEY, then OPENAI_API_KEY, then AI_GATEWAY_API_KEY. Luce never bundles a default so no
call is made to a paid API you did not name."""


# ---------------------------------------------------------------------------
# teacher / writer endpoints
# ---------------------------------------------------------------------------

@dataclass
class Endpoint:
    """An OpenAI-compatible chat endpoint: base URL + model id."""
    url: str
    model: str
    votes: int = 1
    min_agreement: int = 1

    @staticmethod
    def parse(spec: Any, votes: int = 1, min_agreement: int = 1) -> "Endpoint":
        if isinstance(spec, Endpoint):
            return spec
        if isinstance(spec, dict):
            if "url" not in spec or "model" not in spec:
                raise ValueError("endpoint mapping needs 'url' and 'model'")
            return Endpoint(str(spec["url"]).rstrip("/"), str(spec["model"]), int(spec.get("votes", votes)), int(spec.get("min_agreement", min_agreement)))
        if isinstance(spec, str) and "|" in spec:
            url, model = spec.split("|", 1)
            return Endpoint(url.strip().rstrip("/"), model.strip(), votes, min_agreement)
        raise ValueError(f"endpoint must be 'URL|MODEL' or a mapping with url/model, got {spec!r}")

    def api_key(self) -> Optional[str]:
        for name in ("LUCE_TEACHER_API_KEY", "OPENAI_API_KEY", "AI_GATEWAY_API_KEY"):
            value = os.environ.get(name)
            if value:
                return value
        return None

    def describe(self) -> str:
        return f"{self.model} @ {self.url}"


# ---------------------------------------------------------------------------
# luce.yaml sections
# ---------------------------------------------------------------------------

@dataclass
class QuestionSpec:
    name: str
    type: str                      # choice | score | noul
    prompt: str
    options: Dict[str, str] = field(default_factory=dict)   # choice
    levels: List[str] = field(default_factory=list)         # score
    dynamic_options: bool = False                          # choice options vary by state
    options_prompt: str = ""                              # writer instructions for dynamic options

    def validate(self) -> None:
        if self.type not in ("choice", "score", "noul"):
            raise ValueError(f"question {self.name!r}: type must be choice|score|noul, got {self.type!r}")
        if not self.prompt.strip():
            raise ValueError(f"question {self.name!r}: prompt is empty")
        if not isinstance(self.dynamic_options, bool):
            raise ValueError(f"question {self.name!r}: dynamic_options must be true or false")
        if self.type != "choice" and (self.dynamic_options or self.options_prompt):
            raise ValueError(f"question {self.name!r}: dynamic_options and options_prompt are only supported for choice")
        if self.type == "choice" and (not self.dynamic_options or self.options) and len(self.options) < 2:
            raise ValueError(f"question {self.name!r}: choice needs >= 2 options")
        if self.type == "score" and not (2 <= len(self.levels) <= 10):
            raise ValueError(f"question {self.name!r}: score needs 2..10 levels")

    def num_options(self) -> int:
        return {"choice": len(self.options), "score": len(self.levels), "noul": 2}[self.type]

    def to_record_fields(self) -> Dict[str, Any]:
        """Fields that go into a JSONL record for this question (label added by the caller)."""
        out: Dict[str, Any] = {"type": self.type, "question": self.prompt}
        if self.type == "choice":
            out["options"] = dict(self.options)
        if self.type == "score":
            out["levels"] = list(self.levels)
        return out

    def to_server_spec(self) -> Dict[str, Any]:
        """Shape accepted by luce.core.question_from_dict / the HTTP API."""
        out: Dict[str, Any] = {"type": self.type, "prompt": self.prompt}
        if self.type == "choice":
            out["options"] = dict(self.options)
        if self.type == "score":
            out["levels"] = list(self.levels)
        return out


@dataclass
class TaskSpec:
    name: str = "task"
    description: str = ""
    state_fields: List[str] = field(default_factory=list)   # empty = free text state


@dataclass
class SynthSpec:
    n: int = 3000
    teacher: Optional[Endpoint] = None
    writer: Optional[Endpoint] = None            # None -> teacher writes states too
    grid: Dict[str, Union[List[str], Dict[str, float]]] = field(default_factory=dict)
    personas: int = 20
    distractors: str = "near_miss"               # near_miss | none
    distractor_share: float = 0.1
    on_disagreement: str = "mark_hard"           # drop | mark_hard
    val_fraction: float = 0.1
    seed: int = 0
    mode: str = "new"                            # new | label | relabel | append
    input: Optional[str] = None                   # existing state/record JSONL
    questions: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)  # final record count per question
    answer_ratios: Dict[str, Dict[str, float]] = field(default_factory=dict)
    max_rounds: int = 3                           # bounded replenishment of generated records


def _weight_mapping(value: Any, field_name: str) -> Dict[str, float]:
    """Parse YAML/JSON weights while preserving their unnormalized proportions."""
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field_name} must be a nonempty mapping of values to weights")
    out: Dict[str, float] = {}
    for key, weight in value.items():
        name = str(key).lower() if isinstance(key, bool) else str(key)
        if name in out:
            raise ValueError(f"{field_name} has duplicate value {name!r}")
        try:
            number = float(weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field_name}.{name} must be a finite nonnegative weight") from exc
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{field_name}.{name} must be a finite nonnegative weight")
        out[name] = number
    if not any(weight > 0 for weight in out.values()):
        raise ValueError(f"{field_name} weights must have a positive sum")
    return out


def _string_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    result = [str(item) for item in value]
    if any(not item.strip() for item in result):
        raise ValueError(f"{field_name} cannot contain empty values")
    return result


def _synth_grid(value: Any) -> Dict[str, Union[List[str], Dict[str, float]]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("synth.grid must be a mapping of axes to value lists or weight mappings")
    result: Dict[str, Union[List[str], Dict[str, float]]] = {}
    for axis, values in value.items():
        name = str(axis)
        if isinstance(values, dict):
            result[name] = _weight_mapping(values, f"synth.grid.{name}")
        else:
            result[name] = _string_list(values, f"synth.grid.{name}")
            if not result[name]:
                raise ValueError(f"synth.grid.{name} needs at least one value")
    return result


def _question_counts(value: Any) -> Dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("synth.counts must be a mapping of question names to nonnegative integer counts")
    out: Dict[str, int] = {}
    for name, count in value.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"synth.counts.{name} must be a nonnegative integer")
        out[str(name)] = count
    return out


def _answer_ratios(value: Any) -> Dict[str, Dict[str, float]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("synth.answer_ratios must be a mapping of question names to weight mappings")
    return {str(name): _weight_mapping(weights, f"synth.answer_ratios.{name}") for name, weights in value.items()}


@dataclass
class ModelSpec:
    backbone: str = "auto"                       # auto -> BACKBONE_LADDER by labelled-data size; or an HF id
    mode: str = "auto"                           # auto | bi | isolated | label
    lora_r: int = 16
    lora_alpha: int = 32
    select_by: str = "calibrated_nll"
    score_sigma: float = 0.0
    epochs: int = 2
    lr: Optional[float] = None                   # None -> per-mode default (2e-4 bi/isolated, 5e-5 label)
    batch_size: Optional[int] = None             # None -> per-mode default
    trust_remote_code: bool = False              # custom_code backbones (e.g. ByteDance/Ouro-2.6B)
    backbone_overrides: Dict[str, Any] = field(default_factory=dict)  # HF config attributes set before loading (e.g. total_ut_steps: 4)
    label_overflow: str = "isolated"             # label mode, records with > 26 options: isolated | text


@dataclass
class EvalSpec:
    real: Optional[str] = None
    calibration: str = "per_type"                # per_type | global


@dataclass
class ServeSpec:
    port: int = 8000
    review_path: str = "review.jsonl"
    review_threshold: float = 0.9                # calibrated max-prob below this -> logged for review


@dataclass
class LuceConfig:
    task: TaskSpec = field(default_factory=TaskSpec)
    questions: List[QuestionSpec] = field(default_factory=list)
    examples: Optional[str] = None               # seeds.jsonl — synth reference only, never evaluated on
    synth: SynthSpec = field(default_factory=SynthSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)
    serve: ServeSpec = field(default_factory=ServeSpec)
    path: Optional[str] = None

    def question(self, name: str) -> QuestionSpec:
        for q in self.questions:
            if q.name == name:
                return q
        raise KeyError(name)

    def max_options(self) -> int:
        return max((q.num_options() for q in self.questions), default=0)

    def validate(self) -> None:
        if not self.questions:
            raise ValueError("luce.yaml needs at least one question")
        names = [q.name for q in self.questions]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate question names: {names}")
        for q in self.questions:
            q.validate()
        if self.model.label_overflow not in ("isolated", "text"):
            raise ValueError(f"model.label_overflow must be isolated or text, got {self.model.label_overflow!r}")
        if not isinstance(self.model.backbone_overrides, dict):
            raise ValueError("model.backbone_overrides must be a mapping (e.g. {total_ut_steps: 4})")
        if self.model.mode not in ("auto", "bi", "isolated", "label"):
            raise ValueError(f"model.mode must be auto|bi|isolated|label, got {self.model.mode!r}")
        if self.synth.on_disagreement not in ("drop", "mark_hard"):
            raise ValueError("synth.on_disagreement must be drop|mark_hard")
        if self.synth.mode not in ("new", "label", "relabel", "append"):
            raise ValueError("synth.mode must be new|label|relabel|append")
        if isinstance(self.synth.max_rounds, bool) or not isinstance(self.synth.max_rounds, int) or self.synth.max_rounds < 1:
            raise ValueError("synth.max_rounds must be a positive integer")
        selected = _string_list(self.synth.questions, "synth.questions")
        unknown = set(selected) - set(names)
        if unknown:
            raise ValueError(f"synth.questions contains unknown questions: {', '.join(sorted(unknown))}")
        types = _string_list(self.synth.types, "synth.types")
        if set(types) - {"choice", "score", "noul"}:
            raise ValueError("synth.types can only contain choice, score, noul")
        counts = _question_counts(self.synth.counts)
        unknown = set(counts) - set(names)
        if unknown:
            raise ValueError(f"synth.counts contains unknown questions: {', '.join(sorted(unknown))}")
        _synth_grid(self.synth.grid)
        ratios = _answer_ratios(self.synth.answer_ratios)
        for name, weights in ratios.items():
            if name not in names:
                raise ValueError(f"synth.answer_ratios contains unknown question {name!r}")
            question = self.question(name)
            if question.type == "choice":
                if question.dynamic_options:
                    raise ValueError(f"synth.answer_ratios.{name} cannot be used with dynamic choice options")
                expected = set(question.options)
            elif question.type == "score":
                expected = {str(i) for i in range(len(question.levels))}
            else:
                expected = {"true", "false"}
            if set(weights) != expected:
                raise ValueError(f"synth.answer_ratios.{name} must contain exactly: {', '.join(sorted(expected))}")

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        def ep(e: Optional[Endpoint]) -> Any:
            return None if e is None else {"url": e.url, "model": e.model, "votes": e.votes, "min_agreement": e.min_agreement}
        questions: Dict[str, Any] = {}
        for q in self.questions:
            entry: Dict[str, Any] = {"type": q.type, "prompt": q.prompt}
            if q.type == "choice":
                entry["options"] = dict(q.options)
                if q.dynamic_options:
                    entry["dynamic_options"] = True
                if q.options_prompt:
                    entry["options_prompt"] = q.options_prompt
            if q.type == "score":
                entry["levels"] = list(q.levels)
            questions[q.name] = entry
        out: Dict[str, Any] = {
            "task": {"name": self.task.name, "description": self.task.description, "state": {"fields": list(self.task.state_fields)}},
            "questions": questions,
            "examples": self.examples,
            "synth": {
                "n": self.synth.n, "teacher": ep(self.synth.teacher), "writer": ep(self.synth.writer),
                "grid": {k: dict(v) if isinstance(v, dict) else list(v) for k, v in self.synth.grid.items()}, "personas": self.synth.personas,
                "distractors": self.synth.distractors, "distractor_share": self.synth.distractor_share,
                "on_disagreement": self.synth.on_disagreement, "val_fraction": self.synth.val_fraction, "seed": self.synth.seed,
                "mode": self.synth.mode, "input": self.synth.input,
                "questions": list(self.synth.questions), "types": list(self.synth.types),
                "counts": dict(self.synth.counts),
                "answer_ratios": {name: dict(weights) for name, weights in self.synth.answer_ratios.items()},
                "max_rounds": self.synth.max_rounds,
            },
            "model": {"backbone": self.model.backbone, "mode": self.model.mode, "lora": {"r": self.model.lora_r, "alpha": self.model.lora_alpha},
                      "select_by": self.model.select_by, "score_sigma": self.model.score_sigma, "epochs": self.model.epochs,
                      "lr": self.model.lr, "batch_size": self.model.batch_size,
                      "trust_remote_code": self.model.trust_remote_code, "backbone_overrides": dict(self.model.backbone_overrides),
                      "label_overflow": self.model.label_overflow},
            "eval": {"real": self.eval.real, "calibration": self.eval.calibration},
            "serve": {"port": self.serve.port, "review": {"path": self.serve.review_path, "threshold": self.serve.review_threshold}},
        }
        return out

    def save(self, path: str) -> None:
        import yaml
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, allow_unicode=True, sort_keys=False)
        self.path = path

    @staticmethod
    def from_dict(data: Dict[str, Any], path: Optional[str] = None) -> "LuceConfig":
        data = dict(data or {})
        task_d = data.get("task") or {}
        state_d = task_d.get("state") or {}
        task = TaskSpec(name=str(task_d.get("name", "task")), description=str(task_d.get("description", "")),
                        state_fields=list(state_d.get("fields") or []))
        questions: List[QuestionSpec] = []
        q_d = data.get("questions") or {}
        if isinstance(q_d, list):
            q_d = {str(q.get("name", i)): q for i, q in enumerate(q_d)}
        for name, q in q_d.items():
            questions.append(QuestionSpec(name=str(name), type=str(q.get("type", "")).lower(), prompt=str(q.get("prompt", "")),
                                          options={str(k): str(v) for k, v in (q.get("options") or {}).items()},
                                          levels=[str(x) for x in (q.get("levels") or [])],
                                          dynamic_options=q.get("dynamic_options", False),
                                          options_prompt=str(q.get("options_prompt") or "")))
        s_d = data.get("synth") or {}
        synth = SynthSpec(
            n=int(s_d.get("n", 3000)),
            teacher=Endpoint.parse(s_d["teacher"], votes=3, min_agreement=2) if s_d.get("teacher") else None,
            writer=Endpoint.parse(s_d["writer"]) if s_d.get("writer") else None,
            grid=_synth_grid(s_d.get("grid")),
            personas=int(s_d.get("personas", 20)), distractors=str(s_d.get("distractors", "near_miss")),
            distractor_share=float(s_d.get("distractor_share", 0.1)),
            on_disagreement=str(s_d.get("on_disagreement", "mark_hard")), val_fraction=float(s_d.get("val_fraction", 0.1)),
            seed=int(s_d.get("seed", 0)),
            mode=str(s_d.get("mode", "new")), input=s_d.get("input"),
            questions=_string_list(s_d.get("questions"), "synth.questions"),
            types=_string_list(s_d.get("types"), "synth.types"),
            counts=_question_counts(s_d.get("counts")), answer_ratios=_answer_ratios(s_d.get("answer_ratios")),
            max_rounds=s_d.get("max_rounds", 3),
        )
        m_d = data.get("model") or {}
        lora_d = m_d.get("lora") or {}
        model = ModelSpec(backbone=str(m_d.get("backbone", "auto")), mode=str(m_d.get("mode", "auto")),
                          lora_r=int(lora_d.get("r", 16)), lora_alpha=int(lora_d.get("alpha", 32)),
                          select_by=str(m_d.get("select_by", "calibrated_nll")), score_sigma=float(m_d.get("score_sigma", 0.0)),
                          epochs=int(m_d.get("epochs", 2)), lr=m_d.get("lr"), batch_size=m_d.get("batch_size"),
                          trust_remote_code=bool(m_d.get("trust_remote_code", False)),
                          backbone_overrides=dict(m_d.get("backbone_overrides") or {}),
                          label_overflow=str(m_d.get("label_overflow", "isolated")))
        e_d = data.get("eval") or {}
        ev = EvalSpec(real=e_d.get("real"), calibration=str(e_d.get("calibration", "per_type")))
        sv_d = data.get("serve") or {}
        rv = sv_d.get("review") or {}
        serve = ServeSpec(port=int(sv_d.get("port", 8000)), review_path=str(rv.get("path", "review.jsonl")),
                          review_threshold=float(rv.get("threshold", 0.9)))
        cfg = LuceConfig(task=task, questions=questions, examples=data.get("examples"), synth=synth, model=model, eval=ev, serve=serve, path=path)
        return cfg

    @staticmethod
    def load(path: str = DEFAULT_CONFIG_PATH) -> "LuceConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        cfg = LuceConfig.from_dict(data, path=path)
        cfg.validate()
        return cfg


def load_config_if_present(path: Optional[str]) -> Optional[LuceConfig]:
    candidate = path or DEFAULT_CONFIG_PATH
    if os.path.exists(candidate):
        return LuceConfig.load(candidate)
    if path:  # explicitly given but missing
        raise FileNotFoundError(candidate)
    return None


# ---------------------------------------------------------------------------
# mode: auto
# ---------------------------------------------------------------------------

@dataclass
class ModeDecision:
    mode: str                    # bi | isolated | label
    reason: str
    closed_set: bool
    max_options: int
    train_records: int
    train_flags: List[str]

    def describe(self) -> str:
        return f"mode=auto -> {self.mode}: {self.reason}"


def inspect_records(paths: Iterable[str]) -> Tuple[int, bool, int]:
    """Return (n_records, closed_set, max_options) over the given JSONL files.
    closed_set = every distinct question prompt maps to exactly one option set (choice) — score/noul are closed by construction."""
    n = 0
    option_sets: Dict[str, set] = {}
    max_options = 0
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                n += 1
                t = str(r.get("type", "")).lower()
                if t == "choice":
                    key = str(r.get("question", ""))
                    opts = r.get("options") or {}
                    option_sets.setdefault(key, set()).add(json.dumps(opts, sort_keys=True, ensure_ascii=False))
                    max_options = max(max_options, len(opts))
                elif t == "score":
                    max_options = max(max_options, len(r.get("levels") or []))
                else:
                    max_options = max(max_options, 2)
    closed = all(len(v) == 1 for v in option_sets.values()) if option_sets else True
    return n, closed, max_options


def decide_mode(requested: str, train_paths: Iterable[str], config_max_options: int = 0) -> ModeDecision:
    n, closed, max_opts = inspect_records(train_paths)
    max_opts = max(max_opts, config_max_options)
    if requested != "auto":
        mode = requested
        reason = f"mode set explicitly (train records={n}, closed_set={closed}, max_options={max_opts})"
    elif n == 0:
        if max_opts <= MAX_LABEL_OPTIONS:
            mode, reason = "label", f"no training data; max_options={max_opts} <= {MAX_LABEL_OPTIONS} so the label readout starts exactly at the prompting baseline"
        else:
            mode, reason = "isolated", f"no training data; max_options={max_opts} > {MAX_LABEL_OPTIONS} rules out letter labels, isolated scoring with LM prior"
    elif closed:
        mode, reason = "bi", f"train records={n}; every question has one fixed option set (closed set) -> bi-encoder with option cache"
    else:
        mode, reason = "isolated", f"train records={n}; option sets vary per record (open candidates) -> isolated cross scoring with LM prior"
    flags = mode_to_train_flags(mode)
    return ModeDecision(mode=mode, reason=reason, closed_set=closed, max_options=max_opts, train_records=n, train_flags=flags)


DEFAULT_BACKBONE = "Qwen/Qwen3-4B-Base"   # largest rung: prompting-grade floor with little or no data

# backbone: auto — E15 사다리 실험(합성 티켓, 라벨 0~2,000, Qwen3 0.6B/1.7B/4B-Base) 결과 모든 라벨 수에서
# 4B > 1.7B > 0.6B (라벨 1,000: 88.8 / 84.3 / 82.1). 작은 백본은 "공짜" 가 아니라 4~7 점을 내는 비용 선택이므로
# auto 는 4B 로 고정하고, 작은 백본은 model.backbone 에 명시해서 고른다. (banking77 같은 다른 과제에서 재측정 예정.)
BACKBONE_LADDER: List[Tuple[int, str]] = [
    (0, "Qwen/Qwen3-4B-Base"),
]
SMALLEST_RUNG_NEEDS_CLOSED_SET = True
SMALLER_RUNGS_NOTE = "measured on the synthetic-ticket task (E15): Qwen3-1.7B-Base costs ~4 points and Qwen3-0.6B-Base ~7 points at 1,000 labels; set model.backbone explicitly to trade accuracy for size"


@dataclass
class BackboneDecision:
    backbone: str
    reason: str
    train_records: int
    closed_set: bool

    def describe(self) -> str:
        return f"backbone=auto -> {self.backbone}: {self.reason}"


def decide_backbone(requested: str, train_paths: Iterable[str]) -> BackboneDecision:
    """backbone: auto 를 데이터 양(라벨 수)과 선택지 집합의 폐쇄성으로 사다리에서 고른다."""
    n, closed, _ = inspect_records(train_paths)
    if requested and requested != "auto":
        return BackboneDecision(backbone=requested, reason=f"backbone set explicitly (train records={n})", train_records=n, closed_set=closed)
    chosen = BACKBONE_LADDER[0][1]
    lo = 0
    for threshold, name in BACKBONE_LADDER:
        if n >= threshold:
            chosen, lo = name, threshold
    if len(BACKBONE_LADDER) > 1 and lo == BACKBONE_LADDER[-1][0] and SMALLEST_RUNG_NEEDS_CLOSED_SET and not closed:
        chosen, lo = BACKBONE_LADDER[-2][1], BACKBONE_LADDER[-2][0]
        reason = f"train records={n} >= {BACKBONE_LADDER[-1][0]} but option sets vary per record (open candidates); the smallest rung needs a closed set -> {chosen}"
    elif n == 0:
        reason = f"no training data; the prompting floor carries everything -> {chosen}"
    elif len(BACKBONE_LADDER) == 1:
        reason = f"train records={n} (closed_set={closed}) -> {chosen}; {SMALLER_RUNGS_NOTE}"
    else:
        hi = next((t for t, _ in BACKBONE_LADDER if t > lo), None)
        span = f"{lo}..{hi - 1}" if hi else f"{lo}+"
        reason = f"train records={n} in {span} (closed_set={closed}) -> {chosen}"
    return BackboneDecision(backbone=chosen, reason=reason, train_records=n, closed_set=closed)


def backbone_flags(model: "ModelSpec", include_label_overflow: bool = True) -> List[str]:
    """luce.yaml model: 의 백본 관련 항목을 train / eval_logprob CLI 플래그로."""
    flags: List[str] = []
    if model.trust_remote_code:
        flags.append("--trust-remote-code")
    for key, value in (model.backbone_overrides or {}).items():
        flags += ["--backbone-override", f"{key}={value}"]
    if include_label_overflow and model.label_overflow != "isolated":
        flags += ["--label-overflow", model.label_overflow]
    return flags


def mode_to_train_flags(mode: str) -> List[str]:
    if mode == "bi":
        return ["--scorer", "bi"]
    if mode == "isolated":
        return ["--scorer", "cross", "--lm-prior"]
    if mode == "label":
        return ["--scorer", "cross", "--lm-prior", "--options-in-prefix", "--continuation", "label"]
    raise ValueError(mode)


MODE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "bi": {"lr": 2e-4, "batch_size": 8, "grad_accum": 2},
    "isolated": {"lr": 5e-5, "batch_size": 4, "grad_accum": 4},
    "label": {"lr": 5e-5, "batch_size": 4, "grad_accum": 4},
}
