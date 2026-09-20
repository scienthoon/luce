"""
luce.llm — minimal OpenAI-compatible chat client (Gateway / OpenAI / Ollama / vLLM). No SDK dependency.
"""

from __future__ import annotations

import threading
import os
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from .config import Endpoint


class ChatError(RuntimeError):
    pass


_RAW_LOCK = threading.Lock()


def _log_raw(endpoint: Endpoint, messages: List[Dict[str, str]], content: str, payload: Dict[str, Any]) -> None:
    """LUCE_LLM_LOG=<path> 가 설정돼 있으면 모든 teacher/writer 호출의 요청과 원본 응답을 JSONL 로 남긴다 (재현성·감사용)."""
    path = os.environ.get("LUCE_LLM_LOG")
    if not path:
        return
    row = {"ts": time.time(), "url": endpoint.url, "model": endpoint.model, "messages": messages, "content": content,
           "usage": payload.get("usage"), "id": payload.get("id")}
    with _RAW_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def chat(endpoint: Endpoint, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 1024,
         retries: int = 4, timeout: int = 120) -> str:
    """One chat completion; returns the assistant text. Retries on 429/5xx with backoff."""
    body = {"model": endpoint.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    key = endpoint.api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(f"{endpoint.url}/chat/completions", data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    delay = 2.0
    last: Optional[str] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"] or ""
            _log_raw(endpoint, messages, content, payload)
            return content
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8", errors="replace")[:300]
            last = f"HTTP {error.code}: {text}"
            if error.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay); delay *= 2; continue
            raise ChatError(last)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last = str(error)
            if attempt < retries - 1:
                time.sleep(delay); delay *= 2; continue
            raise ChatError(last)
    raise ChatError(last or "unknown error")


def chat_many(endpoint: Endpoint, prompts: Sequence[List[Dict[str, str]]], temperature: float = 0.2, max_tokens: int = 1024,
              concurrency: int = 8, on_progress=None) -> List[Optional[str]]:
    """Run many chats concurrently; failures become None (caller decides)."""
    results: List[Optional[str]] = [None] * len(prompts)

    def run(i: int) -> None:
        try:
            results[i] = chat(endpoint, prompts[i], temperature=temperature, max_tokens=max_tokens)
        except ChatError as error:
            results[i] = None
            if on_progress:
                on_progress(i, str(error))

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(run, range(len(prompts))))
    return results


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in a model reply (tolerates code fences and prose around it)."""
    if text is None:
        raise ValueError("empty reply")
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        for opener, closer in (("{", "}"), ("[", "]")):
            i = cand.find(opener); j = cand.rfind(closer)
            if i != -1 and j > i:
                try:
                    return json.loads(cand[i:j + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"no JSON found in reply: {text[:200]!r}")


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token for English, ~2 for CJK)."""
    cjk = sum(1 for ch in text if "぀" <= ch <= "鿿" or "가" <= ch <= "힯")
    return int(cjk / 1.5 + (len(text) - cjk) / 4) + 1
