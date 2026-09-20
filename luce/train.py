"""
luce.train — 결정 모델 학습.

실행 예 (4080; 기본 백본 Qwen3-4B-Base, 실험 표의 Qwen2.5-3B 는 --backbone 으로):
    luce train \
        --train data/train.jsonl --val data/val.jsonl \
        --out checkpoints/v2 \
        --backbone Qwen/Qwen2.5-3B \
        --epochs 3 --batch-size 8 --grad-accum 2 --lr 2e-4 \
        --grad-checkpointing

손실 = 소프트 타겟 cross-entropy + brier_weight * Brier
  - cross-entropy 는 proper scoring rule 이라 그 자체로 캘리브레이션을 밀어준다.
  - Brier 항은 과신을 추가로 벌한다. 0 으로 두면 순수 CE.
학습 종료 후 검증셋에서 temperature 를 맞춰 함께 저장한다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import random
import time
from typing import Dict, List, Sequence, Tuple, Optional

import torch

from .data import Example, batches, batches_by_budget, describe, load_examples, sequence_cost, subsample_options
from .metrics import brier, fit_temperature, format_reliability_table, nll, reliability, summarize
from .model import DecisionConfig, DecisionModel


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_logits(
    model: DecisionModel,
    examples: Sequence[Example],
    batch_size: int,
    autocast_dtype: torch.dtype,
    seq_budget: int = 0,
    partial_path: Optional[str] = None,
    resume: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """전체 데이터셋의 로짓/타겟/마스크를 하나의 (N, Kmax) 텐서로 모은다.
    partial_path 를 주면 예제마다 원시 로짓을 그 파일에 바로 붙여 쓰고(죽어도 남음), resume=True 면 거기 있는 예제는 건너뛴다."""
    model.eval()
    k_max = max(example.num_options for example in examples)
    n_all = len(examples)
    rows: Dict[int, Tuple[List[float], List[float], str]] = {}   # 원래 인덱스 -> (logits, target, type)
    if partial_path and resume and os.path.exists(partial_path):
        with open(partial_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    r = json.loads(line)
                    if 0 <= r["i"] < n_all and examples[r["i"]].num_options == len(r["logits"]):
                        rows[r["i"]] = (r["logits"], r["target"], r["type"])
        print(f"resume: {len(rows)}/{n_all} examples already in {partial_path}", file=sys.stderr, flush=True)
    todo = [i for i in range(n_all) if i not in rows]
    index_of = {id(examples[i]): i for i in todo}
    pending = [examples[i] for i in todo]
    partial = open(partial_path, "a", encoding="utf-8") if partial_path else None
    option_cache: Dict[str, torch.Tensor] = {}
    iterator = batches_by_budget(pending, seq_budget, shuffle=False, seed=0) if seq_budget and seq_budget > 0 else batches(pending, batch_size, shuffle=False, seed=0)
    total = len(pending); done = 0; started = time.time(); last_report = started
    for batch in iterator:
        with torch.autocast(device_type=_autocast_device(model.device_name), dtype=autocast_dtype, enabled=_autocast_enabled(model.device_name)):
            logits, target, mask = model.forward_examples(batch, option_cache=option_cache)
        for j, example in enumerate(batch):
            k = example.num_options
            i = index_of[id(example)]
            rows[i] = (logits[j, :k].float().cpu().tolist(), target[j, :k].float().cpu().tolist(), example.qtype)
            if partial is not None:
                partial.write(json.dumps({"i": i, "type": example.qtype, "logits": rows[i][0], "target": rows[i][1]}) + "\n")
                partial.flush()
        done += len(batch)
        now = time.time()
        if now - last_report >= 30 and done < total:
            # 30초마다 진행률과 남은 시간 (긴 state 평가에서 "언제 끝나는지" 를 보기 위해)
            rate = done / max(now - started, 1e-6)
            print(f"eval progress: {done}/{total} ({100 * done / total:.0f}%), {now - started:.0f}s elapsed, ETA {(total - done) / rate:.0f}s", file=sys.stderr, flush=True)
            last_report = now
    if partial is not None:
        partial.close()
    # 원래 순서로 (N, Kmax) 텐서 조립
    out_logits = torch.full((n_all, k_max), float("-inf"))
    out_target = torch.zeros((n_all, k_max))
    out_mask = torch.zeros((n_all, k_max), dtype=torch.bool)
    types: List[str] = []
    for i in range(n_all):
        lg, tg, qtype = rows[i]
        k = len(lg)
        out_logits[i, :k] = torch.tensor(lg); out_target[i, :k] = torch.tensor(tg); out_mask[i, :k] = True
        types.append(qtype)
    return out_logits, out_target, out_mask, types


def evaluate(
    model: DecisionModel,
    examples: Sequence[Example],
    batch_size: int,
    autocast_dtype: torch.dtype,
    temperature: float = 1.0,
    seq_budget: int = 0,
    partial_path: Optional[str] = None,
    resume: bool = False,
) -> Dict[str, object]:
    logits, target, mask, types = collect_logits(model, examples, batch_size, autocast_dtype, seq_budget=seq_budget, partial_path=partial_path, resume=resume)
    overall = summarize(logits, target, mask, temperature=temperature)
    by_type: Dict[str, Dict[str, float]] = {}
    for qtype in sorted(set(types)):
        index = torch.tensor([i for i, t in enumerate(types) if t == qtype])
        by_type[qtype] = summarize(logits[index], target[index], mask[index], temperature=temperature)
    # 데이터셋(source)별 분해. convert.py 가 넣는 "source" 필드가 있을 때만.
    sources = [str(example.meta.get("source", "")) for example in examples]
    by_source: Dict[str, Dict[str, float]] = {}
    if any(sources):
        for source in sorted(set(s for s in sources if s)):
            index = torch.tensor([i for i, s in enumerate(sources) if s == source])
            by_source[source] = summarize(logits[index], target[index], mask[index], temperature=temperature)
    rel = reliability(logits, target, mask, temperature=temperature)
    return {"overall": overall, "by_type": by_type, "by_source": by_source, "reliability": rel, "_logits": logits, "_target": target, "_mask": mask}


def _autocast_device(device_name: str) -> str:
    return "cuda" if device_name == "cuda" else "cpu"


def _autocast_enabled(device_name: str) -> bool:
    return device_name == "cuda"


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------

def _parse_overrides(items) -> Dict[str, object]:
    """["total_ut_steps=2", "early_exit_threshold=0.9"] -> {"total_ut_steps": 2, "early_exit_threshold": 0.9}"""
    out: Dict[str, object] = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        if not key or not raw:
            raise ValueError(f"--backbone-override expects key=value, got {item!r}")
        value: object = raw
        for cast in (int, float):
            try:
                value = cast(raw); break
            except ValueError:
                continue
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        out[key] = value
    return out


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)  # cross+options_in_prefix 의 선택지 셔플 재현성

    train_examples = load_examples(args.train, score_sigma=args.score_sigma, limit=args.limit)
    val_examples = load_examples(args.val, score_sigma=0.0, limit=args.limit) if args.val else []
    print("train:", json.dumps(describe(train_examples), ensure_ascii=False))
    if val_examples:
        print("val:  ", json.dumps(describe(val_examples), ensure_ascii=False))

    config = DecisionConfig(
        backbone=args.backbone,
        scorer=args.scorer,
        lm_prior=args.lm_prior,
        options_in_prefix=args.options_in_prefix,
        continuation=args.continuation,
        label_overflow=args.label_overflow,
        shuffle_options=not args.no_shuffle_options,
        pooling=args.pooling,
        proj_dim=args.proj_dim,
        head_dropout=args.head_dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_query_len=args.max_query_len,
        max_option_len=args.max_option_len,
        trust_remote_code=args.trust_remote_code,
        backbone_overrides=_parse_overrides(args.backbone_override),
    )
    model = DecisionModel(config, device=args.device, use_lora=not args.no_lora, gradient_checkpointing=args.grad_checkpointing)
    print("params:", json.dumps(model.count_parameters()))
    print("device:", model.device_name, "dtype:", model.dtype)

    autocast_dtype = torch.bfloat16 if model.device_name == "cuda" else torch.float32

    params = model.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    target_examples = args.batch_size * args.grad_accum           # 옵티마이저 스텝당 예제 수 (유효 배치)
    optimizer_steps_per_epoch = math.ceil(len(train_examples) / target_examples)
    total_steps = optimizer_steps_per_epoch * args.epochs
    max_cost = max(sequence_cost(e) for e in train_examples)
    seq_budget = args.seq_budget if args.seq_budget is not None else (max(args.batch_size, 4) * 8 if max_cost > 8 else 0)
    eval_seq_budget = args.eval_seq_budget if args.eval_seq_budget is not None else (max(args.eval_batch_size, 4) * 8 if max_cost > 8 else 0)
    if seq_budget:
        print(f"sequence-budget batching: train {seq_budget} sequences/forward, eval {eval_seq_budget}, effective batch {target_examples} examples (max options {max_cost})")
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cosine_with_warmup(s, total_steps, warmup_steps))

    os.makedirs(args.out, exist_ok=True)
    best_val_nll = float("inf")
    global_step = 0
    history: List[Dict[str, object]] = []

    if args.eval_init and val_examples:
        # 학습 전 한 번 평가. lm_prior 면 이 값이 "선택지 우도 채점"의 성능이고 cross 경로의 첫 검증이다.
        result = evaluate(model, val_examples, args.eval_batch_size, autocast_dtype, seq_budget=eval_seq_budget)
        print(f"init (step 0) val: {json.dumps(result['overall'])}")
        for qtype, metrics in result["by_type"].items():  # type: ignore[union-attr]
            print(f"  {qtype}: {json.dumps(metrics)}")
        for source, metrics in result["by_source"].items():  # type: ignore[union-attr]
            print(f"  [{source}]: {json.dumps(metrics)}")
        # 0스텝도 이 val 에서 T 를 맞춘 값을 같이 찍는다 (v0 의 --fit-temperature 결과와 같은 조건으로 NLL 비교용).
        init_temperature = fit_temperature(result["_logits"], result["_target"], result["_mask"])  # type: ignore[arg-type]
        init_calibrated = summarize(result["_logits"], result["_target"], result["_mask"], temperature=init_temperature)  # type: ignore[arg-type]
        print(f"init (step 0) val (calibrated, T={init_temperature:.4f}): {json.dumps(init_calibrated)}")
        sources0 = [str(example.meta.get("source", "")) for example in val_examples]
        for source in sorted(set(x for x in sources0 if x)):
            index = torch.tensor([i for i, x in enumerate(sources0) if x == source])
            print(f"  [{source}] (calibrated): {json.dumps(summarize(result['_logits'][index], result['_target'][index], result['_mask'][index], temperature=init_temperature))}")  # type: ignore[index]
        history.append({"epoch": 0, "val": result["overall"], "val_by_type": result["by_type"], "val_by_source": result["by_source"], "val_calibrated": init_calibrated, "val_temperature": init_temperature})

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_ce = 0.0
        running_brier = 0.0
        running_count = 0
        optimizer.zero_grad(set_to_none=True)

        epoch_examples = train_examples
        if args.max_train_options > 0:
            # 학습용 음성 표본추출: 선택지가 많은 choice 예제는 정답 + 무작위 음성 (K-1) 개만 채점 (epoch 마다 다른 표본). 평가는 전체.
            sub_rng = random.Random(args.seed * 1000 + epoch)
            epoch_examples = [subsample_options(e, args.max_train_options, sub_rng) for e in train_examples]
            if epoch == 0:
                n_sub = sum(1 for e in epoch_examples if "subsampled_from" in e.meta)
                print(f"max_train_options={args.max_train_options}: {n_sub}/{len(epoch_examples)} examples subsampled for training (eval uses all options)")
        iterator = batches_by_budget(epoch_examples, seq_budget, shuffle=True, seed=args.seed + epoch) if seq_budget else batches(epoch_examples, args.batch_size, shuffle=True, seed=args.seed + epoch)
        seen_since_step = 0
        seen_epoch = 0
        for batch in iterator:
            with torch.autocast(device_type=_autocast_device(model.device_name), dtype=autocast_dtype, enabled=_autocast_enabled(model.device_name)):
                logits, target, mask = model.forward_examples(batch)
            logits = logits.float()
            loss_ce = nll(logits, target, mask)
            loss_brier = brier(logits, target, mask)
            loss = loss_ce + args.brier_weight * loss_brier
            # 배치 크기가 가변이므로 예제 수로 가중해 유효 배치(target_examples)의 평균이 되게 한다
            (loss * (len(batch) / target_examples)).backward()

            running_loss += float(loss.item()) * len(batch)
            running_ce += float(loss_ce.item()) * len(batch)
            running_brier += float(loss_brier.item()) * len(batch)
            running_count += len(batch)
            seen_since_step += len(batch)
            seen_epoch += len(batch)

            if seen_since_step >= target_examples or seen_epoch >= len(train_examples):
                seen_since_step = 0
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"epoch {epoch + 1} step {global_step}/{total_steps} "
                        f"loss {running_loss / running_count:.4f} ce {running_ce / running_count:.4f} "
                        f"brier {running_brier / running_count:.4f} lr {lr_now:.2e}"
                    )
                    running_loss = 0.0
                    running_ce = 0.0
                    running_brier = 0.0
                    running_count = 0

        epoch_time = time.time() - epoch_start
        record: Dict[str, object] = {"epoch": epoch + 1, "train_seconds": round(epoch_time, 1)}

        if val_examples:
            result = evaluate(model, val_examples, args.eval_batch_size, autocast_dtype, seq_budget=eval_seq_budget)
            overall = result["overall"]
            record["val"] = overall
            record["val_by_type"] = result["by_type"]
            print(f"epoch {epoch + 1} val: {json.dumps(overall)}")
            for qtype, metrics in result["by_type"].items():  # type: ignore[union-attr]
                print(f"  {qtype}: {json.dumps(metrics)}")
            for source, metrics in result["by_source"].items():  # type: ignore[union-attr]
                print(f"  [{source}]: {json.dumps(metrics)}")
            record["val_by_source"] = result["by_source"]

            # 확률을 내놓는 모델이므로 베스트 선택 기준도 "보정된 확률의 질" 이어야 한다.
            # 보정 전 NLL 로 고르면 더 맞히지만 과신하는 epoch 을 버리게 된다 (temperature 가 잡을 수 있는 결함인데도).
            epoch_temperature = fit_temperature(result["_logits"], result["_target"], result["_mask"])  # type: ignore[arg-type]
            calibrated = summarize(result["_logits"], result["_target"], result["_mask"], temperature=epoch_temperature)  # type: ignore[arg-type]
            record["val_calibrated"] = calibrated
            record["val_temperature"] = epoch_temperature
            print(f"epoch {epoch + 1} val (calibrated, T={epoch_temperature:.4f}): {json.dumps(calibrated)}")

            if args.select_by == "calibrated_nll":
                score = float(calibrated["nll"])
            elif args.select_by == "nll":
                score = float(overall["nll"])  # type: ignore[index]
            else:
                score = -float(overall["accuracy"])  # type: ignore[index]
            if score < best_val_nll:
                best_val_nll = score
                model.save(args.out)
                print(f"  saved best checkpoint to {args.out} ({args.select_by} {score:.4f})")
            if not args.no_save_last:
                model.save(os.path.join(args.out, "last"))
        else:
            model.save(args.out)
            print(f"  saved checkpoint to {args.out}")

        history.append(record)
        with open(os.path.join(args.out, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)

    # -- temperature 보정: 베스트 체크포인트를 다시 로드해서 검증셋에 맞춘다 ------
    if args.epochs <= 0:
        # --epochs 0 --eval-init: 학습 없이 출발점(0스텝)만 평가하고 끝낸다. 체크포인트가 없으므로 보정도 건너뛴다.
        return
    if val_examples:
        # 학습 모델을 먼저 내려 16GB 카드에서 백본이 두 번 올라가지 않게 한다.
        # 이전 실행에서 재로드 뒤 GPU 메모리가 두 배(12→22GB)로 남아 보정 평가가 3배 느려졌다. 루프 지역변수(logits/loss/batch 등)와
        # peft 래퍼가 잡고 있는 참조까지 끊고 gc 를 돌린 뒤에 재로드한다.
        for name in ("logits", "target", "mask", "loss", "loss_ce", "loss_brier", "batch", "result", "overall"):
            if name in locals():
                del locals()[name]
        del optimizer
        del scheduler
        del params
        model.backbone = None  # type: ignore[assignment]
        model.head = None      # type: ignore[assignment]
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"gpu memory after release: {torch.cuda.memory_allocated() / 2**30:.1f} GB allocated")
        best = DecisionModel.load(args.out, device=args.device)
        result = evaluate(best, val_examples, args.eval_batch_size, autocast_dtype, seq_budget=eval_seq_budget)
        logits = result["_logits"]
        target = result["_target"]
        mask = result["_mask"]
        temperature = fit_temperature(logits, target, mask)  # type: ignore[arg-type]
        before = summarize(logits, target, mask, temperature=1.0)  # type: ignore[arg-type]
        after = summarize(logits, target, mask, temperature=temperature)  # type: ignore[arg-type]
        print(f"temperature scaling: T={temperature:.4f}")
        print(f"  before: {json.dumps(before)}")
        print(f"  after:  {json.dumps(after)}")
        print("reliability (after):")
        print(format_reliability_table(reliability(logits, target, mask, temperature=temperature)))  # type: ignore[arg-type]

        best.config.temperature = temperature
        with open(os.path.join(args.out, "decision_config.json"), "w", encoding="utf-8") as handle:
            json.dump(best.config.to_dict(), handle, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out, "calibration.json"), "w", encoding="utf-8") as handle:
            json.dump({"temperature": temperature, "before": before, "after": after}, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="train a luce decision model")
    parser.add_argument("--train", required=True, help="학습 JSONL")
    parser.add_argument("--val", default=None, help="검증 JSONL (캘리브레이션과 베스트 선택에 사용)")
    parser.add_argument("--out", required=True, help="체크포인트 디렉터리")
    parser.add_argument("--backbone", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--trust-remote-code", action="store_true", help="custom_code 백본 (예: ByteDance/Ouro-2.6B)")
    parser.add_argument("--backbone-override", action="append", default=None, help="백본 config 덮어쓰기 key=value (예: total_ut_steps=2). 반복 가능")
    parser.add_argument("--device", default=None)
    parser.add_argument("--scorer", default="bi", choices=["bi", "cross"], help="bi: 선택지 독립 인코딩(캐시 가능, 닫힌 선택지용). cross: 질문+선택지 한 시퀀스(객관식/후보 검증용)")
    parser.add_argument("--lm-prior", action="store_true", help="cross 전용: 선택지 토큰 LM 로그확률을 점수의 출발점으로 쓴다 (헤드 마지막층 0-init)")
    parser.add_argument("--options-in-prefix", action="store_true", help="cross 전용: 접두부에 선택지 전체를 나열해 비교 문맥을 준다")
    parser.add_argument("--continuation", default="text", choices=["text", "label"], help="cross 전용: 이어붙는 부분. label 은 options-in-prefix 필요, 0스텝 = v0 라벨 로그확률")
    parser.add_argument("--max-train-options", type=int, default=0, help="학습 시 choice 선택지 상한: 정답 + 무작위 음성 (K-1) 개만 채점 (0 = 끔). 평가는 전체 선택지. banking77 처럼 선택지가 많은 과제의 비용 절감")
    parser.add_argument("--label-overflow", default="isolated", choices=["isolated", "text"], help="label 모드에서 선택지 26개 초과 예제: isolated(나열 없이 원문 채점, 기본) | text(나열 유지, 느림)")
    parser.add_argument("--no-shuffle-options", action="store_true", help="cross+options-in-prefix: 학습 중 선택지 순서 섞기 끄기")
    parser.add_argument("--pooling", default="last", choices=["last", "mean"])
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-lora", action="store_true", help="백본 동결, 헤드만 학습")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--max-query-len", type=int, default=512)
    parser.add_argument("--max-option-len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--seq-budget", type=int, default=None, help="forward 당 시퀀스 수 예산 (선택지 수 합). 기본: 선택지 8개 초과 데이터면 batch_size*8, 아니면 끔(고정 배치)")
    parser.add_argument("--eval-seq-budget", type=int, default=None, help="평가용 시퀀스 예산. 기본: 선택지 8개 초과면 eval_batch_size*8")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument("--score-sigma", type=float, default=0.0, help="Score 타입 거리 인지 소프트 타겟의 폭. 기본 0(one-hot). >0 이면 argmax 기준 ECE 가 과소확신으로 잡히고 전역 temperature 가 타입 간에 어긋난다. 기대값(score)을 소비하며 그 오차로 보정할 때만 사용.")
    parser.add_argument("--eval-init", action="store_true", help="학습 전에 검증셋을 한 번 평가 (lm_prior 출발점 확인, cross 경로 검증)")
    parser.add_argument("--select-by", default="calibrated_nll", choices=["calibrated_nll", "nll", "accuracy"], help="베스트 체크포인트 선택 기준. 기본: epoch 마다 검증셋으로 T 를 맞춘 뒤의 NLL")
    parser.add_argument("--no-save-last", action="store_true", help="마지막 epoch 체크포인트를 <out>/last 에 저장하지 않는다")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="디버그용: 앞 N개만 사용")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
