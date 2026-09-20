"""
luce.server — Jev API 모양을 흉내낸 로컬 HTTP 서버.

실행:
    LUCE_MODEL=Qwen/Qwen3-4B-Base uvicorn luce.server:app --host 0.0.0.0 --port 8000

요청 예:
    POST /v1/ask
    {
      "state": "The front door has been unlocked for 40 minutes and nobody is home.",
      "questions": {
        "warn":    {"type": "noul",   "prompt": "Should someone be warned about this?"},
        "area":    {"type": "choice", "prompt": "Which area is this about?",
                    "options": {"security": "Doors, locks, alarms", "climate": "Heating and ventilation"}},
        "urgency": {"type": "score",  "prompt": "How urgent is it?",
                    "levels": ["Ignore", "Today", "Right now"]}
      }
    }

응답 예:
    {
      "answers": {
        "warn":    {"type": "noul",   "noul": 0.94, "confidence": 0.67},
        "area":    {"type": "choice", "choice": "security",
                    "probabilities": {"security": 0.97, "climate": 0.03}, "confidence": 0.81},
        "urgency": {"type": "score",  "score": 1.8,
                    "probabilities": {"Ignore": 0.02, "Today": 0.16, "Right now": 0.82},
                    "confidence": 0.55, "legend": ["Ignore", "Today", "Right now"]}
      },
      "model": "Qwen/Qwen3-1.7B",
      "temperature": 1.0,
      "n_perm": 1
    }
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .core import JevLocal, answers_to_dict, question_from_dict


# ---------------------------------------------------------------------------
# 설정 (환경변수)
# ---------------------------------------------------------------------------

MODEL_NAME: str = os.environ.get("LUCE_MODEL", "Qwen/Qwen3-4B-Base")
TEMPERATURE: float = float(os.environ.get("LUCE_TEMPERATURE", "1.0"))
N_PERM: int = int(os.environ.get("LUCE_N_PERM", "1"))
BATCH_SIZE: int = int(os.environ.get("LUCE_BATCH_SIZE", "16"))
MAX_LENGTH: int = int(os.environ.get("LUCE_MAX_LENGTH", "2048"))
DEVICE: Optional[str] = os.environ.get("LUCE_DEVICE") or None
# 엔진 선택: "logprob" (v0, 베이스 LM 로그확률) | "decision" (v2, 학습된 결정 헤드 체크포인트)
ENGINE: str = os.environ.get("LUCE_ENGINE", "logprob").lower()
CHECKPOINT: Optional[str] = os.environ.get("LUCE_CHECKPOINT") or None
# 리뷰 큐: 보정 후 max-prob 가 임계값 미만인 답을 JSONL 로 쌓는다 (`luce train --append review.jsonl` 로 재학습).
REVIEW_PATH: Optional[str] = os.environ.get("LUCE_REVIEW_PATH") or None
REVIEW_THRESHOLD: float = float(os.environ.get("LUCE_REVIEW_THRESHOLD", "0.9"))
TRUST_REMOTE_CODE: bool = os.environ.get("LUCE_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes")  # logprob 엔진, custom_code 백본
BACKBONE_OVERRIDES: str = os.environ.get("LUCE_BACKBONE_OVERRIDES", "")  # logprob 엔진, "total_ut_steps=4,foo=bar"


# ---------------------------------------------------------------------------
# 요청/응답 스키마
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    state: Any = Field(..., description="문자열 또는 JSON 객체")
    questions: Dict[str, Dict[str, Any]] = Field(..., description="이름 -> 질문 스펙")
    temperature: Optional[float] = Field(None, description="이번 요청에만 적용할 temperature")


class AskResponse(BaseModel):
    answers: Dict[str, Dict[str, Any]]
    model: str
    temperature: float
    n_perm: int
    review: Dict[str, bool] = {}   # 질문별로 리뷰 큐에 들어갔는지 (max_prob < threshold)


_review_lock = threading.Lock()


def _max_prob(answer: Dict[str, Any]) -> float:
    if answer.get("type") == "noul":
        p = float(answer["noul"]); return max(p, 1.0 - p)
    return max(float(v) for v in answer["probabilities"].values())


def _maybe_review(state: Any, question_specs: Dict[str, Dict[str, Any]], answers: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
    """임계값 미만 답을 review.jsonl 에 기록. 레코드는 학습 JSONL 스키마 + 예측 (label 은 비워 두어 사람이 채움)."""
    flags: Dict[str, bool] = {}
    if not REVIEW_PATH:
        return flags
    rows = []
    for name, answer in answers.items():
        mp = _max_prob(answer)
        low = mp < REVIEW_THRESHOLD
        flags[name] = low
        if low:
            spec = question_specs[name]
            row: Dict[str, Any] = {"state": state, "type": spec["type"], "question": spec["prompt"], "label": None,
                                   "source": "review", "meta": {"question_name": name, "max_prob": mp, "predicted": answer, "ts": time.time()}}
            if spec["type"] == "choice":
                row["options"] = spec["options"]
            if spec["type"] == "score":
                row["levels"] = spec["levels"]
            rows.append(row)
    if rows:
        with _review_lock:
            with open(REVIEW_PATH, "a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return flags


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


# ---------------------------------------------------------------------------
# 앱
# ---------------------------------------------------------------------------

_engine: Optional[Any] = None  # JevLocal | DecisionEngine (같은 ask/model_name/device/temperature/n_perm 인터페이스)


def _build_engine() -> Any:
    if ENGINE == "decision":
        if not CHECKPOINT:
            raise RuntimeError("LUCE_ENGINE=decision 이면 LUCE_CHECKPOINT 가 필요합니다")
        from .decision import DecisionEngine
        return DecisionEngine(checkpoint_dir=CHECKPOINT, device=DEVICE, batch_size=BATCH_SIZE)
    if ENGINE != "logprob":
        raise RuntimeError(f"unknown LUCE_ENGINE={ENGINE!r} (expected logprob | decision)")
    from .train import _parse_overrides
    return JevLocal(
        model_name=MODEL_NAME,
        device=DEVICE,
        temperature=TEMPERATURE,
        n_perm=N_PERM,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        trust_remote_code=TRUST_REMOTE_CODE,
        backbone_overrides=_parse_overrides([x for x in BACKBONE_OVERRIDES.split(",") if x.strip()]) or None,
    )


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """서버 시작 시 모델을 한 번 올리고, 종료 시 내린다."""
    global _engine
    _engine = _build_engine()
    try:
        yield
    finally:
        _engine = None


app = FastAPI(title="luce", version="0.2.0", lifespan=_lifespan)


def _get_engine() -> Any:
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not loaded yet")
    return _engine


@app.get("/demo", response_class=HTMLResponse)
def demo_page() -> str:
    """브라우저 데모: 티켓 피드 + 확률 막대 + 검토 큐 + 타이핑 실시간 채점. 같은 origin 이라 CORS 불필요."""
    path = os.path.join(os.path.dirname(__file__), "demo.html")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


@app.get("/demo/tickets")
def demo_tickets() -> JSONResponse:
    """데모 피드용 표본 티켓. LUCE_DEMO_TICKETS=<json> 이 있으면 그 파일, 없으면 패키지의 demo_tickets.json."""
    path = os.environ.get("LUCE_DEMO_TICKETS") or os.path.join(os.path.dirname(__file__), "demo_tickets.json")
    with open(path, "r", encoding="utf-8") as handle:
        return JSONResponse(content=json.load(handle))


@app.get("/demo/replay")
def demo_replay() -> JSONResponse:
    """데모 재생용: 실제 서버가 낸 답과 지연을 저장한 파일 (LUCE_DEMO_REPLAY 또는 패키지의 demo_replay.json). GPU 없이 /demo?replay=1."""
    path = os.environ.get("LUCE_DEMO_REPLAY") or os.path.join(os.path.dirname(__file__), "demo_replay.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="no replay file; run scripts/make_replay.py against a live server")
    with open(path, "r", encoding="utf-8") as handle:
        return JSONResponse(content=json.load(handle))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    engine = _get_engine()
    return HealthResponse(status="ok", model=engine.model_name, device=engine.device)


@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    engine = _get_engine()

    if not request.questions:
        raise HTTPException(status_code=422, detail="questions must not be empty")

    try:
        questions = {name: question_from_dict(spec) for name, spec in request.questions.items()}
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error))

    if request.temperature is not None and request.temperature <= 0.0:
        raise HTTPException(status_code=422, detail="temperature must be > 0")

    # 요청별 temperature는 인자로만 넘긴다. 엔진 상태를 바꾸지 않으므로 동시 요청에 안전하다.
    answers = engine.ask(request.state, questions, temperature=request.temperature)
    answers_d = answers_to_dict(answers)
    review = _maybe_review(request.state, request.questions, answers_d)

    return AskResponse(
        answers=answers_d,
        model=engine.model_name,
        temperature=request.temperature if request.temperature is not None else engine.temperature,
        n_perm=engine.n_perm,
        review=review,
    )
