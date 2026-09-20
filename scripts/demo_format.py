"""Pretty-print a /v1/ask response (stdin) or a review.jsonl (--review PATH) for the demo recording."""
import json
import sys


def show_answer(r):
    a = r["answers"]
    q, p, n = a["queue"], a["priority"], a["angry"]
    top = sorted(q["probabilities"].items(), key=lambda kv: -kv[1])
    print("  queue    : %-9s  %s" % (q["choice"], "  ".join("%s %.2f" % kv for kv in top)))
    lvl = p["legend"][int(round(p["score"]))]
    print("  priority : %-9s  expected %.2f  %s" % (lvl, p["score"], "  ".join("%s %.2f" % kv for kv in p["probabilities"].items())))
    print("  angry    : %.2f" % n["noul"])
    flagged = [k for k, v in r.get("review", {}).items() if v]
    if flagged:
        print("  review   : %s  (calibrated max-prob < threshold -> appended to review.jsonl)" % ", ".join(flagged))


def show_review(path):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        m = r.get("meta", {})
        pred = m.get("predicted", {})
        short = pred.get("choice") or ("%.2f" % pred["score"] if "score" in pred else "%.2f" % pred.get("noul", 0.0))
        print("   %-9s max_prob %.3f  predicted %-6s label %s  subject %r" % (
            m.get("question_name"), m.get("max_prob", 0.0), short, r.get("label"), r["state"].get("subject")))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--review":
        show_review(sys.argv[2])
    else:
        show_answer(json.load(sys.stdin))
