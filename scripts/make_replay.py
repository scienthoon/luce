"""Run the demo tickets (and the typing script) through a live `luce serve` once and save every answer with its
latency, so the demo page can replay real model outputs without a GPU (`/demo?replay=1`).

    python scripts/make_replay.py --url http://localhost:8000 --out luce/demo_replay.json
"""
import argparse
import json
import time
import urllib.request

Q = {
    "queue": {"type": "choice", "prompt": "Which support queue should handle this ticket?",
              "options": {"billing": "Payments, refunds, duplicate charges", "shipping": "Delivery status, lost or damaged packages",
                          "technical": "App or website bugs, login problems", "general": "Questions, feedback, anything else"}},
    "priority": {"type": "score", "prompt": "How should this ticket be prioritized?", "levels": ["Low", "Normal", "High", "Critical"]},
    "angry": {"type": "noul", "prompt": "The customer sounds angry."},
}
TYPED = [
    ("Charged twice for order #4471", "Two charges for one order. This is unacceptable, fix it today or I dispute the card."),
    ("Charged twice for order #4471", "Hi, I noticed two charges for one order. Could you take a look when you have a moment? Thanks."),
]


def ask(url, state):
    body = json.dumps({"state": state, "questions": Q}).encode("utf-8")
    req = urllib.request.Request(url + "/v1/ask", data=body, headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        j = json.load(r)
    j["ms"] = (time.time() - t0) * 1000
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--tickets", default="luce/demo_tickets.json")
    ap.add_argument("--out", default="luce/demo_replay.json")
    a = ap.parse_args()
    tickets = json.load(open(a.tickets, encoding="utf-8"))
    feed = []
    for t in tickets:
        j = ask(a.url, t["state"])
        feed.append({"state": t["state"], "answers": j["answers"], "review": j.get("review", {}), "ms": j["ms"]})
    typed = []
    for subject, body in TYPED:
        prefixes = []
        for n in range(8, len(body) + 1, 6):
            state = {"channel": "chat", "customer_tier": "gold", "subject": subject, "body": body[:n]}
            j = ask(a.url, state)
            prefixes.append({"text": body[:n], "answers": j["answers"], "review": j.get("review", {}), "ms": j["ms"]})
        typed.append({"subject": subject, "body": body, "prefixes": prefixes})
    json.dump({"feed": feed, "typed": typed, "recorded_with": "luce serve, Qwen3-4B-Base + LoRA (rule tickets), RTX 4070 SUPER"},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"replay: {len(feed)} tickets, {sum(len(t['prefixes']) for t in typed)} typing prefixes, mean {sum(f['ms'] for f in feed)/len(feed):.0f} ms -> {a.out}")


if __name__ == "__main__":
    main()
