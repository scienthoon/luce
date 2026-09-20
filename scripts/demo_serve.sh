#!/bin/bash
# Demo: serve a trained Luce checkpoint and ask it typed questions.
#   1. a billing ticket -> queue / priority / angry with probabilities
#   2. the same ticket, calm wording -> priority and anger move
#   3. an ambiguous ticket -> low confidence lands in the review queue
# Recorded with: asciinema rec -i 2 -c "bash scripts/demo_serve.sh" media/serve_demo.cast
set -u
CKPT=${CKPT:-checkpoints/four/rule_tickets}
PORT=${PORT:-8000}
REVIEW=/tmp/luce_review_demo.jsonl
rm -f "$REVIEW"
PY=${PY:-./.venv/bin/python}
say() { printf '\n\033[1;36m$ %s\033[0m\n' "$*"; }

say "luce serve --checkpoint $CKPT --port $PORT --review review.jsonl --review-threshold 0.9"
$PY -m luce.cli serve --checkpoint "$CKPT" --port "$PORT" --review "$REVIEW" --review-threshold 0.9 > /tmp/luce_serve_demo.log 2>&1 &
SRV=$!
printf 'loading Qwen3-4B-Base + LoRA adapter'
for i in $(seq 1 300); do
  if curl -s -o /dev/null "http://localhost:$PORT/health"; then echo " ready"; break; fi
  printf '.'; sleep 2
done

QUESTIONS='{
  "queue":    {"type": "choice", "prompt": "Which support queue should handle this ticket?",
               "options": {"billing": "Payments, refunds, duplicate charges", "shipping": "Delivery status, lost or damaged packages", "technical": "App or website bugs, login problems", "general": "Questions, feedback, anything else"}},
  "priority": {"type": "score", "prompt": "How should this ticket be prioritized?", "levels": ["Low", "Normal", "High", "Critical"]},
  "angry":    {"type": "noul", "prompt": "The customer sounds angry."}
}'

ask() {  # $1 = state json
  curl -s "http://localhost:$PORT/v1/ask" -H 'content-type: application/json' \
    -d "{\"state\": $1, \"questions\": $QUESTIONS}" | $PY scripts/demo_format.py
}

say 'ticket 1: gold customer, charged twice, angry'
ask '{"channel": "email", "customer_tier": "gold", "subject": "Charged twice", "body": "Two charges for one order. This is unacceptable, fix it today or I dispute the card."}'

say 'ticket 2: same problem, calm wording, free tier'
ask '{"channel": "email", "customer_tier": "free", "subject": "Charged twice", "body": "Hi, I noticed two charges for one order. Could you take a look when you have a moment? Thanks."}'

say 'ticket 3: order never came AND charged -> which answers fall under the 0.9 review threshold?'
ask '{"channel": "chat", "customer_tier": "standard", "subject": "Order issue", "body": "My order never came and I was still charged for it."}'

say "cat review.jsonl   # what a human sees"
if [ -f "$REVIEW" ]; then $PY scripts/demo_format.py --review "$REVIEW"; else echo "  (empty)"; fi

say 'done; stopping the server'
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
