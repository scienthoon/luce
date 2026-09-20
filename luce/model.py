"""
luce.model — 결정 모델 (v2 / v2.1).

두 가지 스코어러
---------------
scorer="bi"  (v2)   질문과 선택지를 따로 인코딩하고 벡터 상호작용으로 점수.
                    선택지 임베딩 캐시 가능. 닫힌 선택지 집합(큐, 우선순위, 예/아니오)에 적합.
                    내용이 있는 선택지(객관식 정답 후보)에서는 무너진다 (ARC 37%).

scorer="cross" (v2.1)  질문 + 선택지를 한 시퀀스로 인코딩. 백본 어텐션이 상호작용을 하고
                    마지막 토큰 은닉 상태를 MLP 로 점수화. 선택지 K 개면 forward K 번.
                    객관식, 후보 검증처럼 선택지 내용이 판단 그 자체인 과제에 적합.

  lm_prior=True (cross 전용)  같은 forward 에서 선택지 토큰들의 LM 로그확률(길이 정규화)을
                    뽑아 학습 가능한 스케일로 점수에 더한다. 헤드 마지막 층을 0 으로 초기화하므로
                    학습 시작 시점의 모델은 "선택지 텍스트 우도로 채점"(lm-eval-harness 방식)과
                    같다. 백본이 이미 아는 것에서 출발해 학습이 위로만 쌓인다.

                    ┌──────────────┐        h_last ──► CrossHead(MLP, 마지막층 0-init) ─┐
  query + " ans_k" ─►│  backbone    │──►                                                ├─► s_k
                    │ (Qwen, LoRA) │        logprob(ans_k tokens) ──► × lm_scale ──────┘
                    └──────────────┘

- 두 모드 모두 LM 헤드로 텍스트를 생성하지 않는다. lm_prior 는 선택지 토큰 위치의 우도만 읽는다.
- 선택지 수 제한이 없다. 라벨 글자가 없으므로 위치 편향도 없다.
- Choice / Score / Noul 모두 같은 헤드. 차이는 data.py 의 텍스트 템플릿과 타겟 구성에 있다.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version

import random

from .data import (
    OPTION_LABELS,
    Example,
    build_cross_continuation,
    build_cross_prefix,
    build_cross_prefix_with_options,
)


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

@dataclass
class DecisionConfig:
    backbone: str = "Qwen/Qwen3-4B-Base"
    scorer: str = "bi"               # "bi" | "cross"
    lm_prior: bool = False           # cross 전용: 선택지 토큰 LM 로그확률을 점수에 더한다
    options_in_prefix: bool = False  # cross 전용: 접두부에 선택지 전체를 나열 (비교 문맥)
    continuation: str = "text"       # cross 전용: 이어붙는 부분. "text"(선택지 원문) | "label"(A/B/C, options_in_prefix 필요)
    label_overflow: str = "isolated" # label 모드에서 선택지가 26개를 넘는 예제 처리: "isolated"(프리픽스에서 나열 제거, 원문 이어붙임) | "text"(나열 유지, 원문 이어붙임; 느림)
    shuffle_options: bool = True     # cross+options_in_prefix 전용: 학습 중 선택지 순서를 섞어 위치 불변성을 학습
    pooling: str = "last"            # bi 전용: "last" | "mean"
    proj_dim: int = 512
    head_dropout: float = 0.1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    max_query_len: int = 512
    max_option_len: int = 64
    temperature: float = 1.0         # 학습 후 검증셋으로 맞춘 값이 저장된다
    trust_remote_code: bool = False  # Ouro 같은 custom_code 백본
    prefix_cache: bool = True        # cross 추론/평가 시 접두부(state+질문)를 한 번만 인코딩하고 KV 를 선택지 수만큼 복제
    backbone_overrides: Dict[str, Any] = field(default_factory=dict)  # 백본 config 덮어쓰기 (예: {"total_ut_steps": 2})
    temperatures: Dict[str, float] = field(default_factory=dict)  # 타입별 T (choice/score/noul). 있으면 우선.

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["lora_targets"] = list(self.lora_targets)
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DecisionConfig":
        data = dict(data)
        if "lora_targets" in data:
            data["lora_targets"] = tuple(data["lora_targets"])
        return DecisionConfig(**data)

    def validate(self) -> None:
        if self.scorer not in ("bi", "cross"):
            raise ValueError(f"scorer must be 'bi' or 'cross', got {self.scorer!r}")
        if self.lm_prior and self.scorer != "cross":
            raise ValueError("lm_prior requires scorer='cross'")
        if self.continuation not in ("text", "label"):
            raise ValueError(f"continuation must be 'text' or 'label', got {self.continuation!r}")
        if self.continuation == "label" and not self.options_in_prefix:
            raise ValueError("continuation='label' requires options_in_prefix=True")
        if self.label_overflow not in ("isolated", "text"):
            raise ValueError(f"label_overflow must be 'isolated' or 'text', got {self.label_overflow!r}")
        if self.options_in_prefix and self.scorer != "cross":
            raise ValueError("options_in_prefix requires scorer='cross'")
        if self.pooling not in ("last", "mean"):
            raise ValueError(f"pooling must be 'last' or 'mean', got {self.pooling!r}")


# ---------------------------------------------------------------------------
# 헤드
# ---------------------------------------------------------------------------

class PairScorer(nn.Module):
    """bi: h_q 와 h_k 를 받아 스칼라 점수. [q, k, q*k, |q-k|] 상호작용 특징 위의 MLP."""

    def __init__(self, hidden_size: int, proj_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, proj_dim)
        self.option_proj = nn.Linear(hidden_size, proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(4 * proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )

    def forward(self, h_query: torch.Tensor, h_option: torch.Tensor) -> torch.Tensor:
        """h_query: (N, H) — 각 선택지에 맞춰 이미 반복된 상태. h_option: (N, H). 반환 (N,)"""
        q = self.query_proj(h_query)
        o = self.option_proj(h_option)
        features = torch.cat([q, o, q * o, (q - o).abs()], dim=-1)
        return self.mlp(features).squeeze(-1)


class CrossHead(nn.Module):
    """cross: (질문+선택지) 시퀀스의 마지막 토큰 은닉 상태 -> 스칼라. lm_prior 가 있으면 우도 항을 더한다."""

    def __init__(self, hidden_size: int, proj_dim: int, dropout: float, lm_prior: bool) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )
        self.lm_prior = lm_prior
        if lm_prior:
            # 학습 시작 시 점수 = lm_scale * mean_logprob 이 되도록 마지막 층을 0 으로.
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
            self.lm_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("lm_scale", None)

    def forward(self, h_last: torch.Tensor, lm_logprob: Optional[torch.Tensor] = None) -> torch.Tensor:
        """h_last: (N, H). lm_logprob: (N,) 선택지 토큰 평균 로그확률 (lm_prior 일 때). 반환 (N,)"""
        score = self.mlp(h_last).squeeze(-1)
        if self.lm_prior:
            if lm_logprob is None:
                raise ValueError("lm_prior head needs lm_logprob")
            score = score + self.lm_scale * lm_logprob
        return score


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------

class _CacheUnsupported(RuntimeError):
    pass


def _expand_cache(past: Any, idx: torch.Tensor) -> Any:
    """KV 캐시의 배치 행을 idx 로 재배열/복제한다 (transformers 5.x layers, 4.x key_cache, legacy tuple 지원)."""
    if hasattr(past, "layers"):
        for layer in past.layers:
            keys = getattr(layer, "keys", None); values = getattr(layer, "values", None)
            if keys is None or values is None:
                raise _CacheUnsupported(f"cache layer {type(layer).__name__} has no keys/values")
            layer.keys = keys.index_select(0, idx); layer.values = values.index_select(0, idx)
        return past
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        for layer_index in range(len(past.key_cache)):
            past.key_cache[layer_index] = past.key_cache[layer_index].index_select(0, idx)
            past.value_cache[layer_index] = past.value_cache[layer_index].index_select(0, idx)
        return past
    if isinstance(past, tuple):
        return tuple(tuple(t.index_select(0, idx) for t in layer) for layer in past)
    raise _CacheUnsupported(f"unknown cache type {type(past).__name__}")


def _transformers_major() -> int:
    try:
        return int(transformers_version.split(".")[0])
    except (ValueError, IndexError):
        return 0


def _last_hidden(outputs: Any) -> torch.Tensor:
    """백본 출력에서 last_hidden_state 를 꺼낸다. 표준 ModelOutput 외에 Ouro 처럼
    (ModelOutput, hidden_states_list, gate_list) 튜플을 돌려주는 커스텀 바디도 처리."""
    seen = 0
    while seen < 4:
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (tuple, list)) and outputs:
            outputs = outputs[0]; seen += 1; continue
        break
    raise TypeError(f"cannot find last_hidden_state in backbone output of type {type(outputs).__name__}")


def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """right padding 가정."""
    if pooling == "last":
        lengths = attention_mask.sum(dim=1) - 1
        index = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, index).squeeze(1)
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    raise ValueError(f"unknown pooling: {pooling}")


class DecisionModel(nn.Module):
    def __init__(
        self,
        config: DecisionConfig,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        use_lora: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        config.validate()
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if device.split(":")[0] in ("cuda", "mps") else torch.float32

        self.config = config
        self.device_name = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone, trust_remote_code=config.trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = int(self.tokenizer.pad_token_id)

        # 백본 로드. lm_prior 면 CausalLM 을 받아 몸통(.model)과 lm_head 를 분리한다.
        self.lm_head: Optional[nn.Module] = None
        dtype_kwarg = "dtype" if _transformers_major() >= 5 else "torch_dtype"  # transformers 5.x: dtype / 4.x: torch_dtype
        load_kwargs: Dict[str, Any] = {dtype_kwarg: dtype}
        if config.trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        if config.backbone_overrides:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(config.backbone, trust_remote_code=config.trust_remote_code)
            for key, value in config.backbone_overrides.items():
                setattr(hf_config, key, value)
            load_kwargs["config"] = hf_config
            print(f"backbone config overrides: {config.backbone_overrides}")
        if config.lm_prior:
            causal = AutoModelForCausalLM.from_pretrained(config.backbone, **load_kwargs)
            backbone = getattr(causal, "model", None) or getattr(causal, "transformer", None) or getattr(causal, "base_model", None)
            if backbone is None:
                raise RuntimeError(f"cannot find the decoder body inside {type(causal).__name__} (tried .model/.transformer/.base_model)")
            lm_head = getattr(causal, "lm_head", None) or causal.get_output_embeddings()
            for parameter in lm_head.parameters():
                parameter.requires_grad_(False)
            self.lm_head = lm_head
            del causal
        else:
            backbone = AutoModel.from_pretrained(config.backbone, **load_kwargs)
        backbone.config.use_cache = False
        hidden_size = int(backbone.config.hidden_size)
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.lora_targets),
                bias="none",
            )
            backbone = get_peft_model(backbone, lora_config)
        else:
            # LoRA 없이 쓰면 백본은 동결. (--no-lora 헤드만 학습, 또는 load() 경로)
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)
        self.backbone = backbone
        self.use_lora = use_lora

        # 헤드는 fp32. 백본 은닉 상태를 float으로 올려 넣는다.
        if config.scorer == "bi":
            self.head: nn.Module = PairScorer(hidden_size, config.proj_dim, config.head_dropout)
        else:
            self.head = CrossHead(hidden_size, config.proj_dim, config.head_dropout, config.lm_prior)
        self.head.to(device=device, dtype=torch.float32)
        self.backbone.to(device)
        self._prefix_cache_broken = False
        if self.lm_head is not None:
            self.lm_head.to(device)

    # -- 공통 ---------------------------------------------------------------

    def _autocast_off(self) -> torch.autocast:
        return torch.autocast(device_type="cuda" if self.device_name == "cuda" else "cpu", enabled=False)

    # -- bi: 인코딩 ---------------------------------------------------------

    def encode(self, texts: Sequence[str], max_length: int, truncation_side: str = "right") -> torch.Tensor:
        """
        텍스트 목록 -> (N, H) float32 풀링 벡터 (right padding).
        truncation_side="left" 면 앞부분을 자른다. 쿼리는 "State ... Question ..." 순서이고
        마지막 토큰을 풀링하므로, 길이 초과 시 state 앞부분을 버리고 질문을 남겨야 한다.
        """
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = truncation_side
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(self.device_name)
        attention_mask = encoded["attention_mask"].to(self.device_name)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hidden = _last_hidden(outputs)
        pooled = _pool(hidden, attention_mask, self.config.pooling)
        return pooled.float()

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        return self.encode(texts, self.config.max_query_len, truncation_side="left")

    def encode_options(self, texts: Sequence[str]) -> torch.Tensor:
        return self.encode(texts, self.config.max_option_len, truncation_side="right")

    # -- cross: 인코딩 ------------------------------------------------------

    def _cross_ids(self, prefix_text: str, continuation_text: str) -> Tuple[List[int], List[int]]:
        """접두부와 선택지 부분을 따로 토크나이즈해서 경계 문제 없이 이어붙일 id 를 만든다."""
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=True)
        continuation_ids = self.tokenizer.encode(continuation_text, add_special_tokens=False)
        if len(continuation_ids) > self.config.max_option_len:
            continuation_ids = continuation_ids[:self.config.max_option_len]
        if not continuation_ids:
            continuation_ids = [self.pad_id]
        room = self.config.max_query_len - len(continuation_ids)
        if room < 8:
            room = 8
        if len(prefix_ids) > room:
            # state 의 앞부분을 잘라내고 질문 쪽(끝)을 남긴다.
            prefix_ids = prefix_ids[-room:]
        return prefix_ids, continuation_ids

    def encode_cross(
        self,
        pairs: Sequence[Tuple[str, str]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        pairs: [(prefix_text, continuation_text), ...]
        반환:
          h_last     (N, H) float32  각 시퀀스 마지막 토큰(= 선택지 마지막 토큰)의 은닉 상태
          lm_logprob (N,)   float32  선택지 토큰들의 평균 로그확률 (lm_prior 일 때, 아니면 None)
        left padding 으로 배치해서 선택지 토큰이 항상 시퀀스 끝에 오게 한다.
        """
        id_pairs = [self._cross_ids(prefix, continuation) for prefix, continuation in pairs]
        if self.config.prefix_cache and not torch.is_grad_enabled() and not self._prefix_cache_broken:
            try:
                return self._encode_cross_shared(id_pairs)
            except Exception as error:  # noqa: BLE001  (custom cache 구현 등 어떤 실패든 per-option 경로로)
                self._prefix_cache_broken = True
                print(f"prefix KV cache unsupported for this backbone ({type(error).__name__}: {str(error)[:120]}); falling back to per-option encoding", file=sys.stderr)
        lengths = [len(p) + len(c) for p, c in id_pairs]
        max_len = max(lengths)
        n = len(id_pairs)

        input_ids = torch.full((n, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        for i, (prefix_ids, continuation_ids) in enumerate(id_pairs):
            ids = prefix_ids + continuation_ids
            input_ids[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, max_len - len(ids):] = 1
        input_ids = input_ids.to(self.device_name)
        attention_mask = attention_mask.to(self.device_name)
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
        hidden = _last_hidden(outputs)  # (N, T, H)
        h_last = hidden[:, -1, :].float()

        if not self.config.lm_prior:
            return h_last, None

        # 선택지 토큰 우도: 마지막 (max_cont + 1) 위치의 은닉 상태에만 lm_head 를 적용한다.
        max_cont = max(len(c) for _, c in id_pairs)
        tail = max_cont + 1
        tail_hidden = hidden[:, -tail:, :]                       # (N, tail, H)
        with self._autocast_off():
            lm_dtype = self.lm_head.weight.dtype
            tail_logits = self.lm_head(tail_hidden.to(lm_dtype)).float()   # (N, tail, V)  -> fp32
            tail_logprobs = torch.log_softmax(tail_logits, dim=-1)
        lm_logprob = torch.zeros((n,), device=self.device_name, dtype=torch.float32)
        for i, (_, continuation_ids) in enumerate(id_pairs):
            c = len(continuation_ids)
            # 위치 [tail-c-1, tail-1) 의 로짓이 위치 [tail-c, tail) 의 토큰을 예측한다.
            pred_positions = torch.arange(tail - c - 1, tail - 1, device=self.device_name)
            targets = torch.tensor(continuation_ids, device=self.device_name, dtype=torch.long)
            token_logprobs = tail_logprobs[i, pred_positions, targets]
            lm_logprob[i] = token_logprobs.mean()
        return h_last, lm_logprob

    # -- cross: 접두부 KV 공유 (no-grad) ------------------------------------

    def _encode_cross_shared(self, id_pairs: List[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        같은 접두부를 가진 pair 들(한 예제의 선택지들)은 접두부를 한 번만 forward 하고, KV 캐시를 선택지 수만큼 복제한 뒤
        선택지 토큰만 이어 돌린다. 선택지 K 개면 접두부 계산이 K배 → 1배. 수치적으로 per-option 경로와 동일(패딩 잡음 제외).
        """
        device = self.device_name
        # 1) 연속된 동일 접두부로 그룹화
        groups: List[Tuple[List[int], List[List[int]]]] = []
        for prefix_ids, cont_ids in id_pairs:
            if groups and groups[-1][0] == prefix_ids:
                groups[-1][1].append(cont_ids)
            else:
                groups.append((prefix_ids, [cont_ids]))
        # 2) 접두부 배치 (left padding) → KV 캐시 + 마지막 위치 로짓
        P = max(len(g[0]) for g in groups)
        B = len(groups)
        prefix_input = torch.full((B, P), self.pad_id, dtype=torch.long)
        prefix_mask = torch.zeros((B, P), dtype=torch.long)
        for b, (prefix_ids, _) in enumerate(groups):
            prefix_input[b, P - len(prefix_ids):] = torch.tensor(prefix_ids, dtype=torch.long)
            prefix_mask[b, P - len(prefix_ids):] = 1
        prefix_input = prefix_input.to(device); prefix_mask = prefix_mask.to(device)
        prefix_pos = (prefix_mask.cumsum(dim=-1) - 1).clamp(min=0)
        out = self.backbone(input_ids=prefix_input, attention_mask=prefix_mask, position_ids=prefix_pos, use_cache=True)
        past = getattr(out, "past_key_values", None)
        if past is None:
            raise _CacheUnsupported("backbone returned no past_key_values")
        prefix_last_hidden = _last_hidden(out)[:, -1, :]                    # (B, H)
        prefix_len = prefix_mask.sum(dim=1)                                      # (B,)
        # 3) 캐시를 pair 단위로 복제
        idx = [b for b, (_, conts) in enumerate(groups) for _ in conts]
        idx_t = torch.tensor(idx, device=device, dtype=torch.long)
        past = _expand_cache(past, idx_t)
        # 4) 선택지 토큰 (right padding) 이어 돌리기
        conts = [c for _, cs in groups for c in cs]
        N = len(conts); C = max(len(c) for c in conts)
        cont_input = torch.full((N, C), self.pad_id, dtype=torch.long)
        cont_mask = torch.zeros((N, C), dtype=torch.long)
        for i, c in enumerate(conts):
            cont_input[i, :len(c)] = torch.tensor(c, dtype=torch.long); cont_mask[i, :len(c)] = 1
        cont_input = cont_input.to(device); cont_mask = cont_mask.to(device)
        full_mask = torch.cat([prefix_mask.index_select(0, idx_t), cont_mask], dim=1)   # (N, P+C)
        cont_pos = prefix_len.index_select(0, idx_t).unsqueeze(1) + torch.arange(C, device=device).unsqueeze(0)
        out2 = self.backbone(input_ids=cont_input, attention_mask=full_mask, position_ids=cont_pos, past_key_values=past, use_cache=True)
        hidden = _last_hidden(out2)                                          # (N, C, H)
        last_index = (cont_mask.sum(dim=1) - 1).clamp(min=0)
        h_last = hidden[torch.arange(N, device=device), last_index, :].float()
        if not self.config.lm_prior:
            return h_last, None
        # 5) LM prior: 첫 선택지 토큰은 접두부 마지막 위치 로짓이, 나머지는 선택지 위치 로짓이 예측한다
        with self._autocast_off():
            lm_dtype = self.lm_head.weight.dtype
            first_logprobs = torch.log_softmax(self.lm_head(prefix_last_hidden.to(lm_dtype)).float(), dim=-1)  # (B, V)
            rest_logprobs = torch.log_softmax(self.lm_head(hidden[:, : max(C - 1, 1), :].to(lm_dtype)).float(), dim=-1) if C > 1 else None  # (N, C-1, V)
        lm_logprob = torch.zeros((N,), device=device, dtype=torch.float32)
        for i, c in enumerate(conts):
            total = first_logprobs[idx[i], c[0]]
            if len(c) > 1:
                pos = torch.arange(len(c) - 1, device=device)
                tgt = torch.tensor(c[1:], device=device, dtype=torch.long)
                total = total + rest_logprobs[i, pos, tgt].sum()
            lm_logprob[i] = total / len(c)
        return h_last, lm_logprob

    # -- 배치 forward ------------------------------------------------------

    def forward_examples(
        self,
        examples: Sequence[Example],
        option_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        반환:
          logits (B, Kmax)  패딩 위치는 -inf
          target (B, Kmax)  패딩 위치는 0
          mask   (B, Kmax)  유효 선택지 True
        option_cache 는 bi 모드에서만 쓰인다 (cross 는 선택지가 질문에 묶여 캐시할 수 없다).
        """
        counts = [example.num_options for example in examples]

        if self.config.scorer == "bi":
            flat_scores = self._scores_bi(examples, option_cache)
        else:
            flat_scores = self._scores_cross(examples)

        k_max = max(counts)
        batch_size = len(examples)
        logits = torch.full((batch_size, k_max), float("-inf"), device=flat_scores.device, dtype=flat_scores.dtype)
        target = torch.zeros((batch_size, k_max), device=flat_scores.device, dtype=torch.float32)
        mask = torch.zeros((batch_size, k_max), device=flat_scores.device, dtype=torch.bool)

        offset = 0
        for i, example in enumerate(examples):
            k = example.num_options
            logits[i, :k] = flat_scores[offset:offset + k]
            target[i, :k] = torch.tensor(example.target, device=flat_scores.device, dtype=torch.float32)
            mask[i, :k] = True
            offset += k
        return logits, target, mask

    def _scores_bi(self, examples: Sequence[Example], option_cache: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        query_texts = [example.query_text for example in examples]
        h_query = self.encode_queries(query_texts)  # (B, H)

        counts = [example.num_options for example in examples]
        flat_option_texts: List[str] = []
        for example in examples:
            flat_option_texts.extend(example.option_texts)

        if option_cache is None:
            h_option = self.encode_options(flat_option_texts)  # (sum K, H)
        else:
            missing = [text for text in dict.fromkeys(flat_option_texts) if text not in option_cache]
            if missing:
                encoded_missing = self.encode_options(missing)
                for text, vector in zip(missing, encoded_missing):
                    option_cache[text] = vector
            h_option = torch.stack([option_cache[text] for text in flat_option_texts], dim=0)

        counts_tensor = torch.tensor(counts, device=h_query.device)
        h_query_repeated = torch.repeat_interleave(h_query, counts_tensor, dim=0)  # (sum K, H)
        with self._autocast_off():
            return self.head(h_query_repeated.float(), h_option.float())  # (sum K,)

    def _scores_cross(self, examples: Sequence[Example]) -> torch.Tensor:
        """
        options_in_prefix=False : 접두부 = query.  continuation = ' ' + 선택지 원문.  (고립 검증)
        options_in_prefix=True  : 접두부 = query + 선택지 나열.  continuation = ' ' + 라벨 또는 ' ' + 원문. (비교 선택)
        학습 중(self.training) 이고 shuffle_options 면 나열 순서를 예제마다 섞는다. 점수는 원래 순서로 되돌린다.
        """
        pairs: List[Tuple[str, str]] = []
        # 각 예제의 선택지 k 가 pairs 의 어느 위치에 들어갔는지 기록 (섞었을 때 되돌리기 위해)
        slots: List[List[int]] = []
        for example in examples:
            k = example.num_options
            overflow = (self.config.continuation == "label" and k > len(OPTION_LABELS)
                        and self.config.label_overflow == "isolated")
            if not self.config.options_in_prefix or overflow:
                # 선택지를 나열하지 않는 isolated 방식 (label 모드에서 26개 초과 예제도 여기로: 77개 나열 x 77 시퀀스는 너무 비쌈)
                prefix = build_cross_prefix(example.query_text)
                slots.append(list(range(len(pairs), len(pairs) + k)))
                for description in example.option_descriptions:
                    pairs.append((prefix, build_cross_continuation(description)))
                continue

            order = list(range(k))
            if self.training and self.config.shuffle_options:
                random.shuffle(order)
            use_label = self.config.continuation == "label" and k <= len(OPTION_LABELS)
            labeled = [(OPTION_LABELS[pos] if pos < len(OPTION_LABELS) else str(pos + 1), example.option_descriptions[orig])
                       for pos, orig in enumerate(order)]
            prefix = build_cross_prefix_with_options(example.query_text, labeled)
            example_slots = [0] * k
            for pos, orig in enumerate(order):
                example_slots[orig] = len(pairs)
                if use_label:
                    pairs.append((prefix, build_cross_continuation(labeled[pos][0])))
                else:
                    pairs.append((prefix, build_cross_continuation(example.option_descriptions[orig])))
            slots.append(example_slots)

        h_last, lm_logprob = self.encode_cross(pairs)
        with self._autocast_off():
            scores = self.head(h_last, lm_logprob)  # (sum K,) pairs 순서
        # 원래 선택지 순서로 재배열
        index = torch.tensor([slot for example_slots in slots for slot in example_slots], device=scores.device)
        return scores[index]

    # -- 파라미터 ---------------------------------------------------------

    def trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = [p for p in self.backbone.parameters() if p.requires_grad]
        params.extend(self.head.parameters())
        return params

    def count_parameters(self) -> Dict[str, int]:
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_total = sum(p.numel() for p in self.head.parameters())
        return {
            "backbone_total": backbone_total,
            "backbone_trainable": backbone_trainable,
            "head": head_total,
            "trainable_total": backbone_trainable + head_total,
        }

    # -- 저장/로드 ----------------------------------------------------------

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "decision_config.json"), "w", encoding="utf-8") as handle:
            json.dump(self.config.to_dict(), handle, ensure_ascii=False, indent=2)
        torch.save(self.head.state_dict(), os.path.join(out_dir, "head.pt"))
        if self.use_lora:
            self.backbone.save_pretrained(os.path.join(out_dir, "adapter"))

    @staticmethod
    def load(
        checkpoint_dir: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "DecisionModel":
        with open(os.path.join(checkpoint_dir, "decision_config.json"), "r", encoding="utf-8") as handle:
            config = DecisionConfig.from_dict(json.load(handle))
        adapter_dir = os.path.join(checkpoint_dir, "adapter")
        has_adapter = os.path.isdir(adapter_dir)

        model = DecisionModel(config, device=device, dtype=dtype, use_lora=False, gradient_checkpointing=False)
        if has_adapter:
            from peft import PeftModel
            model.backbone = PeftModel.from_pretrained(model.backbone, adapter_dir)
            model.backbone.to(model.device_name)
            model.use_lora = True
        head_state = torch.load(os.path.join(checkpoint_dir, "head.pt"), map_location=model.device_name)
        model.head.load_state_dict(head_state)
        model.eval()
        return model
