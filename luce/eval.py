"""
luce.eval — 저장된 결정 모델을 JSONL 로 평가.

    luce eval --checkpoint checkpoints/v2 --data data/val.jsonl
    luce eval --checkpoint checkpoints/v2 --data data/real_100.jsonl --fit-temperature --save

--fit-temperature : 이 데이터로 temperature 를 다시 맞춘다 (실제 데이터 소량으로 보정할 때).
--save            : 맞춘 temperature 를 체크포인트의 decision_config.json 에 기록한다.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import torch

import random

from .data import describe, load_examples, permute_example
from .metrics import fit_temperature, format_reliability_table, format_selective_risk, reliability, selective_risk, summarize
from .model import DecisionModel
from .train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate a luce decision model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-budget", type=int, default=None, help="forward 당 시퀀스 수 예산. 기본: 선택지 8개 초과 데이터면 batch_size*8")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None, help="지정하면 체크포인트 값 대신 사용")
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--dump", default=None, help="예제별 예측 확률을 JSONL 로 저장")
    parser.add_argument("--partial", default=None, help="예제별 원시 로짓을 바로 붙여 쓰는 파일 (기본: <dump>.partial.jsonl). 죽어도 남고 --resume 으로 이어감")
    parser.add_argument("--resume", action="store_true", help="--partial 파일에 있는 예제는 건너뛰고 나머지만 계산")
    parser.add_argument("--max-query-len", type=int, default=None, help="체크포인트의 max_query_len 을 평가 시 덮어씀 (긴 state 평가용; 백본 문맥 길이 이내)")
    parser.add_argument("--permute-seed", type=int, default=None, help="선택지 순서를 이 시드로 섞어 평가 (위치 불변성 측정). 고정 순서 결과와 비교")
    parser.add_argument("--calibration", default="per_type", choices=["per_type", "global"], help="--fit-temperature 시 타입별 T 를 맞출지(기본) 하나만 맞출지")
    parser.add_argument("--in-synth", action="store_true", help="합성 데이터 평가임을 출력에 표기 (luce eval 이 --real 없이 호출될 때)")
    args = parser.parse_args()

    examples = load_examples(args.data, score_sigma=0.0, limit=args.limit)
    if args.permute_seed is not None:
        rng = random.Random(args.permute_seed)
        examples = [permute_example(example, rng) for example in examples]
        print(f"options permuted with seed {args.permute_seed}")
    print("data:", json.dumps(describe(examples), ensure_ascii=False))

    model = DecisionModel.load(args.checkpoint, device=args.device)
    if args.max_query_len:
        # 긴 state 를 가진 평가셋(예: TypeSafe 공개 eval, 최대 12k 토큰)에서 학습 때의 512 토큰 좌측 절단을 풀어 준다.
        print(f"max_query_len: {model.config.max_query_len} -> {args.max_query_len} (eval override)")
        model.config.max_query_len = int(args.max_query_len)
    autocast_dtype = torch.bfloat16 if model.device_name == "cuda" else torch.float32
    temperature = args.temperature if args.temperature is not None else model.config.temperature
    tag = "[in-synth] " if args.in_synth else ""

    max_cost = max(example.num_options for example in examples)
    seq_budget = args.seq_budget if args.seq_budget is not None else (max(args.batch_size, 4) * 8 if max_cost > 8 else 0)
    partial_path = args.partial or (args.dump + ".partial.jsonl" if args.dump else None)
    if partial_path and not args.resume and os.path.exists(partial_path):
        os.remove(partial_path)  # 새 실행: 이전 부분 결과 폐기 (--resume 이면 이어서)
    if partial_path:
        print(f"partial results -> {partial_path} (append per example; rerun with --resume to continue)")
    result = evaluate(model, examples, args.batch_size, autocast_dtype, temperature=temperature, seq_budget=seq_budget, partial_path=partial_path, resume=args.resume)
    logits = result["_logits"]
    target = result["_target"]
    mask = result["_mask"]
    types = [example.qtype for example in examples]

    # 체크포인트에 타입별 T 가 있으면 로짓을 타입별로 나눠 적용한 뒤 T=1 로 평가한다.
    per_type_T = dict(getattr(model.config, "temperatures", {}) or {})
    if per_type_T and args.temperature is None:
        scaled = logits.clone()
        for i, t in enumerate(types):
            scaled[i] = logits[i] / float(per_type_T.get(t, temperature))
        logits = scaled
        temperature = 1.0
        print(f"{tag}temperature: per-type {json.dumps({k: round(v, 4) for k, v in per_type_T.items()})}")
    else:
        print(f"{tag}temperature: {temperature:.4f}")
    print(f"{tag}overall:", json.dumps(result["overall"] if not per_type_T else summarize(logits, target, mask, temperature=1.0)))
    for qtype, metrics in result["by_type"].items():  # type: ignore[union-attr]
        print(f"  {qtype}: {json.dumps(metrics)}")
    for source, metrics in result["by_source"].items():  # type: ignore[union-attr]
        print(f"  [{source}]: {json.dumps(metrics)}")
    print(f"{tag}reliability:")
    print(format_reliability_table(reliability(logits, target, mask, temperature=temperature)))  # type: ignore[arg-type]
    print(f"{tag}selective risk (calibrated max-prob thresholds):")
    print(format_selective_risk(selective_risk(logits, target, mask, temperature=temperature)))  # type: ignore[arg-type]

    if args.fit_temperature:
        if args.calibration == "per_type":
            fitted_by_type: Dict[str, float] = {}
            scaled = logits.clone()
            for qtype in sorted(set(types)):
                index = torch.tensor([i for i, t in enumerate(types) if t == qtype])
                t_q = fit_temperature(logits[index], target[index], mask[index])  # type: ignore[arg-type]
                fitted_by_type[qtype] = t_q
                scaled[index] = logits[index] / t_q
            after = summarize(scaled, target, mask, temperature=1.0)  # type: ignore[arg-type]
            print(f"\n{tag}refit temperature (per type): {json.dumps({k: round(v, 4) for k, v in fitted_by_type.items()})}")
            print(f"{tag}after:", json.dumps(after))
            print(format_reliability_table(reliability(scaled, target, mask, temperature=1.0)))  # type: ignore[arg-type]
            print(format_selective_risk(selective_risk(scaled, target, mask, temperature=1.0)))  # type: ignore[arg-type]
            if args.save:
                if args.in_synth:
                    print("NOT saved: temperatures fitted on synthetic data are not written to the checkpoint. Pass --real.")
                else:
                    model.config.temperatures = fitted_by_type
                    with open(os.path.join(args.checkpoint, "decision_config.json"), "w", encoding="utf-8") as handle:
                        json.dump(model.config.to_dict(), handle, ensure_ascii=False, indent=2)
                    print(f"saved per-type temperatures to {args.checkpoint}/decision_config.json")
        else:
            fitted = fit_temperature(logits, target, mask)  # type: ignore[arg-type]
            after = summarize(logits, target, mask, temperature=fitted)  # type: ignore[arg-type]
            print(f"\n{tag}refit temperature: T={fitted:.4f}")
            print(f"{tag}after:", json.dumps(after))
            print(format_reliability_table(reliability(logits, target, mask, temperature=fitted)))  # type: ignore[arg-type]
            print(format_selective_risk(selective_risk(logits, target, mask, temperature=fitted)))  # type: ignore[arg-type]
            if args.save:
                if args.in_synth:
                    print("NOT saved: a temperature fitted on synthetic data is not written to the checkpoint. Pass --real.")
                else:
                    model.config.temperature = fitted
                    model.config.temperatures = {}
                    with open(os.path.join(args.checkpoint, "decision_config.json"), "w", encoding="utf-8") as handle:
                        json.dump(model.config.to_dict(), handle, ensure_ascii=False, indent=2)
                    print(f"saved temperature to {args.checkpoint}/decision_config.json")

    if args.dump:
        probs = torch.softmax((logits / temperature).masked_fill(~mask, float("-inf")), dim=-1)  # type: ignore[operator]
        with open(args.dump, "w", encoding="utf-8") as handle:
            for i, example in enumerate(examples):
                k = example.num_options
                record = {
                    "type": example.qtype,
                    "option_keys": example.option_keys,
                    "probs": [float(x) for x in probs[i, :k].tolist()],
                    "target": example.target,
                    "pred": example.option_keys[int(probs[i, :k].argmax().item())],
                    "gold": example.option_keys[example.hard_label_index],
                    "meta": {k: v for k, v in example.meta.items() if k != "levels"},
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"dumped predictions to {args.dump}")


if __name__ == "__main__":
    main()
