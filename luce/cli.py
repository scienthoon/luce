"""
luce — command line.

    luce init "<task description>" [--examples seeds.jsonl] [--teacher URL|MODEL] [--out luce.yaml]
    luce synth [--questions queue,angry] [--counts queue=100,angry=200] [--mode new|label|relabel|append] [--input records.jsonl]
    luce baseline --real real.jsonl [--n-perm 4]
    luce train [--data train.jsonl] [--append review.jsonl] [--mode auto|bi|isolated|label] [--real real.jsonl] [--out DIR]
    luce eval --checkpoint DIR --real real.jsonl [--synth val.jsonl] [--permute-seed N] [--dump preds.jsonl]
    luce serve --checkpoint DIR [--port 8000] [--review review.jsonl --review-threshold 0.9]

Every command reads luce.yaml (or --config) for defaults; flags override. Extra flags after the known ones are passed
through to the underlying train / eval parsers, so every advanced option stays reachable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from .config import DEFAULT_BACKBONE, DEFAULT_CONFIG_PATH, MODE_DEFAULTS, Endpoint, LuceConfig, TEACHER_EXAMPLES, _weight_mapping, backbone_flags, decide_backbone, decide_mode, load_config_if_present


def _die(message: str, code: int = 2) -> None:
    print(f"luce: {message}", file=sys.stderr)
    sys.exit(code)


def _resolve_teacher(flag: Optional[str], cfg: Optional[LuceConfig]) -> Endpoint:
    if flag:
        return Endpoint.parse(flag, votes=3, min_agreement=2)
    if cfg is not None and cfg.synth.teacher is not None:
        return cfg.synth.teacher
    _die(TEACHER_EXAMPLES)
    raise SystemExit  # unreachable


def _csv_names(value: str) -> List[str]:
    values = [part.strip() for part in value.split(",")]
    if any(not part for part in values):
        raise argparse.ArgumentTypeError("use a comma-separated list without empty entries")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("list entries must be unique")
    return values


def _csv_types(value: str) -> List[str]:
    values = _csv_names(value)
    if set(values) - {"choice", "score", "noul"}:
        raise argparse.ArgumentTypeError("types must be choice, score, or noul")
    return values


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _positive_int(value: str) -> int:
    number = _nonnegative_int(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _count_pairs(value: str) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for pair in _csv_names(value):
        if "=" not in pair:
            raise argparse.ArgumentTypeError("use question=count pairs, e.g. queue=100,angry=200")
        name, count = (part.strip() for part in pair.split("=", 1))
        if not name or name in result:
            raise argparse.ArgumentTypeError("count entries need unique, nonempty question names")
        result[name] = _nonnegative_int(count)
    return result


def _json_weights(value: str) -> Dict[str, Dict[str, float]]:
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object containing weight objects")
        return {str(name): _weight_mapping(weights, f"ratios.{name}") for name, weights in parsed.items()}
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace, passthrough: List[str]) -> None:
    from .scaffold import run_init
    cfg = load_config_if_present(args.config) if os.path.exists(args.config or DEFAULT_CONFIG_PATH) else None
    teacher = Endpoint.parse(args.teacher, votes=3, min_agreement=2) if args.teacher else (cfg.synth.teacher if cfg else None)
    run_init(description=args.description, examples=args.examples, teacher=teacher, out=args.out, name=args.name)


def cmd_synth(args: argparse.Namespace, passthrough: List[str]) -> None:
    from .synth import run_synth
    if passthrough:
        _die(f"unrecognized synth arguments: {' '.join(passthrough)}")
    try:
        cfg = LuceConfig.load(args.config or DEFAULT_CONFIG_PATH)
        teacher = _resolve_teacher(args.teacher, cfg)
        writer = Endpoint.parse(args.writer) if args.writer else cfg.synth.writer
        run_synth(cfg, teacher=teacher, writer=writer, n=args.n, out=args.out, dry_run=args.dry_run, votes=args.votes, seed=args.seed,
                  mode=args.mode, input_path=args.input, questions=args.questions, types=args.types, counts=args.counts,
                  grid_ratios=args.grid_ratios, answer_ratios=args.answer_ratios, max_rounds=args.max_rounds)
    except (ValueError, FileNotFoundError) as exc:
        _die(str(exc))


def cmd_baseline(args: argparse.Namespace, passthrough: List[str]) -> None:
    from . import eval_logprob
    cfg = load_config_if_present(args.config)
    backbone = args.backbone or (cfg.model.backbone if cfg else "auto")
    if backbone == "auto":
        backbone = DEFAULT_BACKBONE
        print(f"backbone=auto -> {backbone}: the prompting baseline uses the largest rung (no training)")
    argv = ["--model", backbone, "--data", args.real, "--n-perm", str(args.n_perm), "--fit-temperature"]
    if cfg:
        argv += backbone_flags(cfg.model, include_label_overflow=False)
    argv += passthrough
    sys.argv = ["luce baseline"] + argv
    eval_logprob.main()


def cmd_train(args: argparse.Namespace, passthrough: List[str]) -> None:
    from . import train as train_module
    cfg = load_config_if_present(args.config)
    data = args.data or "data/synth/train.jsonl"
    train_paths = [data] + list(args.append or [])
    if not os.path.exists(data):
        _die(f"training data not found: {data} (run `luce synth` first or pass --data)")
    requested = args.mode or (cfg.model.mode if cfg else "auto")
    decision = decide_mode(requested, train_paths, cfg.max_options() if cfg else 0)
    print(decision.describe())
    train_file = data
    if args.append:
        # merge appended review data into one file next to the main one
        merged = os.path.join(os.path.dirname(data) or ".", "train+review.jsonl")
        with open(merged, "w", encoding="utf-8") as out:
            for path in train_paths:
                with open(path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            out.write(line if line.endswith("\n") else line + "\n")
        train_file = merged
        print(f"merged {len(train_paths)} files -> {merged}")
    val = args.val or (os.path.join(os.path.dirname(data) or ".", "val.jsonl") if os.path.exists(os.path.join(os.path.dirname(data) or ".", "val.jsonl")) else None)
    real = args.real or (cfg.eval.real if cfg else None)
    if real and os.path.exists(real):
        print(f"real validation set: {real} (best checkpoint and temperature are chosen on it)")
        val = real
    elif val:
        print(f"[in-synth] no --real set; best checkpoint and temperature are chosen on synthetic val {val}. Numbers are in-synth.")
    else:
        print("[in-synth] no validation set at all; the last epoch is saved without calibration.")
    defaults = MODE_DEFAULTS[decision.mode]
    backbone_decision = decide_backbone(args.backbone or (cfg.model.backbone if cfg else "auto"), train_paths)
    print(backbone_decision.describe())
    backbone = backbone_decision.backbone
    out = args.out or os.path.join("checkpoints", (cfg.task.name if cfg else "luce"))
    argv: List[str] = ["--train", train_file, "--out", out, "--backbone", backbone] + decision.train_flags
    if val:
        argv += ["--val", val]
    argv += ["--lr", str(args.lr if args.lr is not None else (cfg.model.lr if cfg and cfg.model.lr else defaults["lr"]))]
    argv += ["--batch-size", str(args.batch_size if args.batch_size is not None else (cfg.model.batch_size if cfg and cfg.model.batch_size else defaults["batch_size"]))]
    argv += ["--grad-accum", str(defaults["grad_accum"])]
    argv += ["--epochs", str(args.epochs if args.epochs is not None else (cfg.model.epochs if cfg else 2))]
    if cfg:
        argv += ["--lora-r", str(cfg.model.lora_r), "--lora-alpha", str(cfg.model.lora_alpha), "--select-by", cfg.model.select_by, "--score-sigma", str(cfg.model.score_sigma)]
        argv += backbone_flags(cfg.model)
    if args.grad_checkpointing:
        argv.append("--grad-checkpointing")
    if decision.max_options > 8:
        argv += ["--eval-batch-size", "2"]  # many-option records: keep eval memory bounded
    argv += passthrough
    print("train args:", " ".join(argv))
    sys.argv = ["luce train"] + argv
    train_module.main()


def cmd_eval(args: argparse.Namespace, passthrough: List[str]) -> None:
    from . import eval as eval_module
    cfg = load_config_if_present(args.config)
    # 우선순위: --real > --synth (명시) > luce.yaml 의 eval.real. --synth 를 명시하면 설정의 real 이 있어도 합성 val 을 [in-synth] 로 평가한다.
    if args.real:
        real, data = args.real, args.real
    elif args.synth:
        real, data = None, args.synth
    elif cfg and cfg.eval.real:
        real, data = cfg.eval.real, cfg.eval.real
    else:
        real, data = None, None
    if not data:
        _die("eval needs --real real.jsonl (recommended, 100-300 labeled items) or --synth val.jsonl")
    if not real:
        print("[in-synth] WARNING: evaluating on synthetic data only. Calibration and headline numbers must come from a real labeled set (--real).")
    argv: List[str] = ["--checkpoint", args.checkpoint, "--data", data]
    if args.permute_seed is not None:
        argv += ["--permute-seed", str(args.permute_seed)]
    if args.dump:
        argv += ["--dump", args.dump]
    if args.fit_temperature:
        argv.append("--fit-temperature")
        if args.save:
            argv.append("--save")
    calibration = args.calibration or (cfg.eval.calibration if cfg else "per_type")
    argv += ["--calibration", calibration]
    if not real:
        argv.append("--in-synth")
    argv += passthrough
    sys.argv = ["luce eval"] + argv
    eval_module.main()


def cmd_convert(args: argparse.Namespace, passthrough: List[str]) -> None:
    from . import convert as convert_module
    sys.argv = ["luce convert"] + passthrough
    convert_module.main()


def cmd_serve(args: argparse.Namespace, passthrough: List[str]) -> None:
    cfg = load_config_if_present(args.config)
    os.environ["LUCE_ENGINE"] = "decision"
    os.environ["LUCE_CHECKPOINT"] = args.checkpoint
    review_path = args.review or (cfg.serve.review_path if cfg else "review.jsonl")
    threshold = args.review_threshold if args.review_threshold is not None else (cfg.serve.review_threshold if cfg else 0.9)
    os.environ["LUCE_REVIEW_PATH"] = review_path
    os.environ["LUCE_REVIEW_THRESHOLD"] = str(threshold)
    port = args.port or (cfg.serve.port if cfg else 8000)
    print(f"serving {args.checkpoint} on :{port}; answers with calibrated max-prob < {threshold} are appended to {review_path}")
    import uvicorn
    uvicorn.run("luce.server:app", host=args.host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="luce", description="Train a 1%-parameter decision head on an open LLM and serve typed probabilities.")
    parser.add_argument("--config", default=None, help=f"luce.yaml path (default: ./{DEFAULT_CONFIG_PATH} if present)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="scaffold luce.yaml from a task description (+ optional seed examples)")
    p.add_argument("description")
    p.add_argument("--examples", default=None, help="seeds.jsonl with 5-20 labeled examples; >=20 are split into seeds/real")
    p.add_argument("--teacher", default=None, help="URL|MODEL used to draft questions/options; without it a template is written")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=DEFAULT_CONFIG_PATH)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("synth", help="generate training data with a teacher LLM")
    p.add_argument("--n", type=_nonnegative_int, default=None, help="shared state count (default: synth.n; input modes use all input states unless set)")
    p.add_argument("--questions", type=_csv_names, default=None, help="question names to label, e.g. queue,angry")
    p.add_argument("--types", type=_csv_types, default=None, help="question types to include, e.g. choice,noul")
    p.add_argument("--counts", type=_count_pairs, default=None, help="target records per question, e.g. queue=100,angry=200")
    p.add_argument("--mode", choices=["new", "label", "relabel", "append"], default=None, help="generate states, label existing states, replace labels, or add missing records")
    p.add_argument("--input", default=None, metavar="FILE", help="existing state/record JSONL for label, relabel, or append")
    p.add_argument("--grid-ratios", type=_json_weights, default=None, metavar="JSON", help='grid weights, e.g. \'{"language":{"ko":70,"en":30}}\'')
    p.add_argument("--answer-ratios", type=_json_weights, default=None, metavar="JSON", help="optional answer weights by question; default preserves teacher-label proportions")
    p.add_argument("--max-rounds", type=_positive_int, default=None, help="maximum attempts to replenish generated records (default: synth.max_rounds)")
    p.add_argument("--teacher", default=None, help="URL|MODEL (labels; required unless set in luce.yaml)")
    p.add_argument("--writer", default=None, help="URL|MODEL (states; defaults to teacher)")
    p.add_argument("--votes", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="print planned calls, tokens and cost, then stop")
    p.add_argument("--out", default="data/synth")
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("baseline", help="prompting baseline (v0 log-prob readout) on a real labeled set")
    p.add_argument("--real", required=True)
    p.add_argument("--backbone", default=None)
    p.add_argument("--n-perm", type=int, default=4)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("train", help="train the decision head (mode auto|bi|isolated|label)")
    p.add_argument("--data", default=None, help="train JSONL (default data/synth/train.jsonl)")
    p.add_argument("--val", default=None)
    p.add_argument("--append", action="append", default=None, help="extra JSONL (e.g. review.jsonl) merged into training")
    p.add_argument("--mode", default=None, choices=["auto", "bi", "isolated", "label"])
    p.add_argument("--real", default=None, help="real labeled set used for checkpoint selection and temperature")
    p.add_argument("--backbone", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="evaluate a checkpoint; --real is required for reportable numbers")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--real", default=None)
    p.add_argument("--synth", default=None)
    p.add_argument("--permute-seed", type=int, default=None)
    p.add_argument("--dump", default=None)
    p.add_argument("--fit-temperature", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--calibration", default=None, choices=["per_type", "global"])
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("convert", help="build JSONL from public HuggingFace presets, a CSV, or the rule-based synthetic generator (args passed through)")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("serve", help="serve a checkpoint over HTTP (POST /v1/ask)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--review", default=None)
    p.add_argument("--review-threshold", type=float, default=None)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    args.func(args, passthrough)


if __name__ == "__main__":
    main()
