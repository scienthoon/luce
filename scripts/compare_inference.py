"""Compare the standalone inference.py predictions with a luce.eval --dump on the same records.
    python scripts/compare_inference.py --luce logs/x_preds.jsonl --standalone preds.jsonl"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--luce", required=True)
ap.add_argument("--standalone", required=True)
args = ap.parse_args()
a = [json.loads(l) for l in open(args.luce, encoding="utf-8") if l.strip()]
b = [json.loads(l) for l in open(args.standalone, encoding="utf-8") if l.strip()]
n = min(len(a), len(b))
worst = 0.0; agree = 0; per_type = {}
for x, y in zip(a[:n], b[:n]):
    assert x["option_keys"] == y["option_keys"], (x["option_keys"], y["option_keys"])
    d = max(abs(p - q) for p, q in zip(x["probs"], y["probs"]))
    worst = max(worst, d); agree += x["pred"] == y["pred"]
    t = per_type.setdefault(x["type"], [0, 0.0]); t[0] += 1; t[1] = max(t[1], d)
print(f"records compared: {n} | argmax agreement: {agree}/{n} | max |dp|: {worst:.5f} | per type: " + ", ".join(f"{k}: n={v[0]} max|dp|={v[1]:.5f}" for k, v in per_type.items()))
