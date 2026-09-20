"""Run the existing Luce synth pipeline with a logged OpenRouter transport.

The key is read silently from the terminal and exists only in memory. The adapter
disables optional reasoning so Luce's existing JSON response budgets are used for
answers. It records requests/responses/usage, never authorization headers.
"""
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import sys
from threading import Lock
import time

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from luce import llm, synth
from luce.config import LuceConfig

ROOT = Path(__file__).resolve().parents[1] / "data/four_tasks/rule_tickets"
lock = Lock()


def main():
    secret = getpass.getpass("OpenRouter key (not saved): ")
    cfg = LuceConfig.load(str(ROOT / "luce.yaml"))
    journal = ROOT / "raw/teacher_calls.jsonl"

    def chat(endpoint, messages, temperature=0.2, max_tokens=1024, retries=4, timeout=120):
        body = {"model": endpoint.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "reasoning": {"enabled": False}}
        for attempt in range(retries):
            try:
                response = requests.post(endpoint.url + "/chat/completions", json=body,
                    headers={"Authorization": "Bearer " + secret}, timeout=(15, timeout))
                if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                    time.sleep(2 ** (attempt + 1))
                    continue
                if not response.ok:
                    # Do not log HTTP headers or credential-bearing request objects.
                    raise llm.ChatError(f"OpenRouter HTTP {response.status_code}: {response.text[:250]}")
                value = response.json()
                text = value["choices"][0]["message"].get("content") or ""
                entry = {"utc": datetime.now(timezone.utc).isoformat(), "request": body,
                         "response_id": value.get("id"), "model": value.get("model"),
                         "provider": value.get("provider"), "usage": value.get("usage"),
                         "finish_reason": value["choices"][0].get("finish_reason"), "content": text}
                with lock:
                    with journal.open("a") as handle:
                        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                return text
            except (requests.RequestException, KeyError, ValueError) as error:
                if attempt + 1 == retries:
                    raise llm.ChatError(type(error).__name__) from None
                time.sleep(2 ** (attempt + 1))
        raise llm.ChatError("OpenRouter retries exhausted")

    # Adapt only the provider transport; Luce's generation/annotation/planning is unchanged.
    llm.chat = chat
    synth.chat = chat
    synth.run_synth(cfg, cfg.synth.teacher, None, None, str(ROOT / "generated"), votes=1, concurrency=12)
    calls = [json.loads(line) for line in journal.read_text().splitlines()]
    usage = {"calls": len(calls), "model": cfg.synth.teacher.model, "reasoning": False,
             "prompt_tokens": sum((row.get("usage") or {}).get("prompt_tokens", 0) for row in calls),
             "completion_tokens": sum((row.get("usage") or {}).get("completion_tokens", 0) for row in calls),
             "reported_cost_usd": sum((row.get("usage") or {}).get("cost", 0) or 0 for row in calls)}
    (ROOT / "generation_usage.json").write_text(json.dumps(usage, indent=2) + "\n")
    print(json.dumps(usage), flush=True)


if __name__ == "__main__":
    main()
