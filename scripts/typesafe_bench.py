"""TypeSafe 공개 eval (evals.typesafe.ai, 20 cases / 372 reference pairs) 을 Luce 형식으로 만들고 채점한다.

시험지 복원 로직은 system-one-open (MIT, mithalouni) 의 typesafe_eval.py 를 그대로 쓴다 (build_from_viewer).
strict common subset 정의도 그쪽 evaluate.py 와 같다: 공개된 세 모델(opus, sol, typesafe=Jev) 이 모두 답한 쌍만.

  python scripts/typesafe_bench.py rebuild --raw data/typesafe/raw --repo <system-one-open clone> --out data/typesafe/typesafe_full.json
  python scripts/typesafe_bench.py convert --full data/typesafe/typesafe_full.json --out data/typesafe/val.jsonl
  python -m luce.eval --checkpoint <ckpt> --data data/typesafe/val.jsonl --dump logs/typesafe_preds.jsonl
  python scripts/typesafe_bench.py score --dump logs/typesafe_preds.jsonl [--full data/typesafe/typesafe_full.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

WORKFLOWS = ["security_incidents", "agent_trace_observability", "invoice_processing", "customer_service"]
MODELS = ("opus", "sol", "typesafe")


# ---------------------------------------------------------------------------
# rebuild: viewer js -> typesafe_full.json (system-one-open 의 build_from_viewer 재사용)
# ---------------------------------------------------------------------------

def cmd_rebuild(args: argparse.Namespace) -> None:
    src = pathlib.Path(args.repo, "typesafe_eval.py").read_text(encoding="utf-8")
    pure = src.split("@app.function")[0].replace("import modal, json, re, os", "import json, re, os")
    pure = "\n".join(l for l in pure.splitlines() if not l.startswith(("app = ", "vol = ", "image = ")))
    ns: Dict[str, Any] = {}
    exec(pure, ns)  # noqa: S102  (MIT 코드, 순수 함수만)
    raw = {wf: pathlib.Path(args.raw, f"{wf}-cases.js").read_text(encoding="utf-8") for wf in WORKFLOWS}
    out = ns["build_from_viewer"](raw)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    total = 0
    for wf, d in out["workflows"].items():
        pairs = sum(len(c["questions"]) for c in d["cases"]); total += pairs
        print(f"{wf:26s} cases {len(d['cases'])} pairs {pairs}")
    print("TOTAL pairs", total, "->", args.out)


# ---------------------------------------------------------------------------
# convert: typesafe_full.json -> Luce JSONL (한 쌍 = 한 레코드)
# ---------------------------------------------------------------------------

def _noul_p_yes(probs: Dict[str, float]) -> float | None:
    if not probs:
        return None
    yes = sum(v for k, v in probs.items() if str(k).lower() in ("true", "yes", "1"))
    no = sum(v for k, v in probs.items() if str(k).lower() in ("false", "no", "0"))
    if yes + no <= 0:
        return None
    return yes / (yes + no)


def cmd_convert(args: argparse.Namespace) -> None:
    full = json.load(open(args.full, encoding="utf-8"))
    records: List[Dict[str, Any]] = []
    for wf in WORKFLOWS:
        for case in full["workflows"][wf]["cases"]:
            for q in case["questions"]:
                published = {m: case["published"].get(m, {}).get(q["qid"]) for m in MODELS}
                base = {"state": case["state"], "question": q["text"], "source": wf,
                        "wf": wf, "case_id": case["case_id"], "qid": q["qid"], "published": published}
                kind, opts, descs, ref, probs = q["kind"], q["options"], q.get("descs"), q["ref_value"], q.get("ref_probs") or {}
                if kind == "noul":
                    rec = {**base, "type": "noul", "label": ref == "yes"}
                    p = _noul_p_yes(probs)
                    if p is not None and args.soft:
                        rec["label_probs"] = p
                elif kind == "score":
                    levels = [d if d else o for o, d in zip(opts, descs or [None] * len(opts))]
                    rec = {**base, "type": "score", "levels": levels, "label": int(ref)}
                    if probs and args.soft:
                        vec = [float(probs.get(str(i), probs.get(i, 0.0))) for i in range(len(levels))]
                        if sum(vec) > 0:
                            rec["label_probs"] = vec
                else:
                    options = {o: (d if d else o) for o, d in zip(opts, descs or [None] * len(opts))}
                    rec = {**base, "type": "choice", "options": options, "label": ref}
                    if probs and args.soft:
                        lp = {k: float(v) for k, v in probs.items() if k in options}
                        if sum(lp.values()) > 0:
                            rec["label_probs"] = lp
                records.append(rec)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = Counter(r["type"] for r in records)
    chars = [len(r["state"]) for r in records]
    print(f"wrote {len(records)} -> {args.out}; kinds {dict(kinds)}; state chars median {sorted(chars)[len(chars)//2]} max {max(chars)}")


# ---------------------------------------------------------------------------
# score: luce.eval --dump 결과 -> 표
# ---------------------------------------------------------------------------

def _canon_pred(rec: Dict[str, Any]) -> str:
    """dump 의 pred 를 typesafe 의 canonical 값으로: noul 'yes'/'no', score '0'.., choice 키."""
    return str(rec["pred"])


def _acc(rows: List[Dict[str, Any]], key) -> float:
    return sum(1 for r in rows if key(r)) / max(1, len(rows))


def _ece(rows: List[Dict[str, Any]], bins: int = 15) -> float:
    total = len(rows); if_empty = 0.0
    if not total:
        return if_empty
    buckets = defaultdict(list)
    for r in rows:
        buckets[min(bins - 1, int(r["conf"] * bins))].append(r)
    return sum(len(b) / total * abs(sum(x["ok"] for x in b) / len(b) - sum(x["conf"] for x in b) / len(b)) for b in buckets.values())


def cmd_score(args: argparse.Namespace) -> None:
    dump = [json.loads(l) for l in open(args.dump, encoding="utf-8") if l.strip()]
    rows: List[Dict[str, Any]] = []
    for d in dump:
        meta = d.get("meta") or {}
        if "published" not in meta:
            sys.exit("dump has no meta.published; re-run luce.eval with a build that dumps example.meta")
        pred = _canon_pred(d)
        ref = d["gold"]
        probs = d["probs"]; keys = d["option_keys"]
        conf = max(probs)
        p_ref = probs[keys.index(ref)] if ref in keys else 0.0
        rows.append({"wf": meta["wf"], "case": meta["case_id"], "qid": meta["qid"], "kind": d["type"], "pred": pred, "ref": ref,
                     "ok": pred == ref, "conf": conf, "nll": -math.log(max(p_ref, 1e-12)), "published": meta["published"]})
    common = [r for r in rows if all(r["published"].get(m) is not None for m in MODELS)]
    print(f"pairs {len(rows)} | strict common subset {len(common)}")
    print("\n== strict common subset (same pairs for every column) ==")
    print(f"  ours      {100 * _acc(common, lambda r: r['ok']):.1f}%   (ECE {_ece(common):.3f}, NLL {sum(r['nll'] for r in common) / max(1, len(common)):.3f})")
    for m in MODELS:
        print(f"  {m:9s} {100 * _acc(common, lambda r, m=m: r['published'][m] == r['ref']):.1f}%")
    maj_type = {k: Counter(r["ref"] for r in rows if r["kind"] == k).most_common(1)[0][0] for k in {r["kind"] for r in rows}}
    maj_q: Dict[str, str] = {}
    for qid in {r["qid"] for r in rows}:
        maj_q[qid] = Counter(r["ref"] for r in rows if r["qid"] == qid).most_common(1)[0][0]
    print(f"  baseline per-type majority {100 * _acc(common, lambda r: maj_type[r['kind']] == r['ref']):.1f}%, per-question majority {100 * _acc(common, lambda r: maj_q[r['qid']] == r['ref']):.1f}% (질문별 최빈 답: 시험지를 아는 상한)")
    print("\n== all 372 pairs, ours ==")
    print(f"  overall {100 * _acc(rows, lambda r: r['ok']):.1f}%  ECE {_ece(rows):.3f}  NLL {sum(r['nll'] for r in rows) / len(rows):.3f}")
    for wf in WORKFLOWS:
        sub = [r for r in rows if r["wf"] == wf]
        print(f"  {wf:26s} {100 * _acc(sub, lambda r: r['ok']):5.1f}%  (n={len(sub)})")
    for kind in ("choice", "score", "noul"):
        sub = [r for r in rows if r["kind"] == kind]
        if sub:
            print(f"  {kind:26s} {100 * _acc(sub, lambda r: r['ok']):5.1f}%  (n={len(sub)})")
    print("\n== published models on the pairs they answered (not comparable across columns) ==")
    for m in MODELS:
        answered = [r for r in rows if r["published"].get(m) is not None]
        print(f"  {m:9s} {100 * _acc(answered, lambda r, m=m: r['published'][m] == r['ref']):.1f}%  (n={len(answered)})")
    if args.out:
        json.dump(rows, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("rows ->", args.out)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("rebuild"); a.add_argument("--raw", required=True); a.add_argument("--repo", required=True); a.add_argument("--out", required=True); a.set_defaults(fn=cmd_rebuild)
    b = sub.add_parser("convert"); b.add_argument("--full", required=True); b.add_argument("--out", required=True); b.add_argument("--soft", action="store_true", help="ref_probs 를 label_probs 로"); b.set_defaults(fn=cmd_convert)
    c = sub.add_parser("score"); c.add_argument("--dump", required=True); c.add_argument("--out", default=None); c.set_defaults(fn=cmd_score)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
