"""Execution of new, label, relabel, and append synthesis jobs."""
from __future__ import annotations

import copy
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any, Dict, List, Optional

from . import synth as api
from .config import LuceConfig
from .synth_io import read_inputs, read_output, state_key, write_output
from .synth_plan import label_key, new_slots, select_rows


def _name(row: dict, cfg: LuceConfig) -> Optional[str]:
    explicit = (row.get("meta") or {}).get("question_name") or row.get("question_name")
    if explicit:
        return explicit
    matches = [q.name for q in cfg.questions if q.type == row.get("type") and q.prompt == row.get("question")]
    return matches[0] if len(matches) == 1 else None


def _labeled(row: dict) -> bool:
    return row.get("label") is not None or row.get("label_probs") is not None


def _path(value: Optional[str], cfg: LuceConfig) -> Optional[str]:
    if value and not os.path.isabs(value):
        return os.path.join(os.path.dirname(cfg.path or "."), value)
    return value


def _annotate(cfg, teacher, jobs, votes, concurrency, errors):
    """Label every selected question independently; never expose previous labels."""
    prompts = [api._label_prompt(cfg, job["state"], list(job["questions"].values()))
               for job in jobs for _ in range(votes)]
    if not prompts:
        return []
    max_questions = max(len(job["questions"]) for job in jobs)
    replies = api.chat_many(teacher, prompts, temperature=0.5 if votes > 1 else 0.0,
                            max_tokens=max(300, max_questions * 80), concurrency=concurrency,
                            on_progress=lambda i, message: errors.append(message))
    result = []
    min_agreement = min(votes, teacher.min_agreement or max(1, (votes + 1) // 2))
    for i, job in enumerate(jobs):
        by_question = {name: [] for name in job["questions"]}
        for reply in replies[i * votes:(i + 1) * votes]:
            try:
                answers = api.extract_json(reply)
            except (ValueError, TypeError):
                continue
            if not isinstance(answers, dict):
                continue
            for name, question in job["questions"].items():
                value = api._coerce_answer(question, answers.get(name))
                if value is not None:
                    by_question[name].append(value)
        rows = {}
        for name, values in by_question.items():
            if not values:
                continue
            question = job["questions"][name]
            top, agreement = Counter(json.dumps(v) for v in values).most_common(1)[0]
            hard = agreement < min_agreement
            if hard and cfg.synth.on_disagreement == "drop":
                continue
            row = {"state": job["state"], **question.to_record_fields(), "label": json.loads(top),
                   "source": job.get("source", "synth"),
                   "meta": {**job.get("meta", {}), "question_name": name, "votes": values, "hard": hard}}
            if hard:
                if question.type == "choice":
                    row["label_probs"] = {key: values.count(key) / len(values) for key in question.options}
                elif question.type == "score":
                    row["label_probs"] = [values.count(index) / len(values) for index in range(len(question.levels))]
                else:
                    row["label_probs"] = values.count(True) / len(values)
            rows[name] = row
        result.append(rows)
    return result


def _generated_job(raw, cfg, slot, persona, near_miss=None):
    dynamic = [q for q in cfg.questions if q.dynamic_options]
    options = {}
    state = raw
    if dynamic:
        if not isinstance(raw, dict) or "state" not in raw or not isinstance(raw.get("options"), dict):
            raise ValueError("dynamic choices require a state/options envelope")
        state, options = raw["state"], raw["options"]
    fields = cfg.task.state_fields
    if fields:
        if not isinstance(state, dict) or any(field not in state for field in fields):
            raise ValueError("generated state is missing configured fields")
        state = {field: str(state[field]) for field in fields}
    elif not isinstance(state, str) or not state.strip():
        raise ValueError("generated state must be a nonempty string")
    questions = {}
    for q in cfg.questions:
        if q.dynamic_options:
            candidates = options.get(q.name)
            if not isinstance(candidates, dict) or len(candidates) < 2 or any(
                not isinstance(key, str) or not key or not isinstance(value, str) or not value.strip()
                for key, value in candidates.items()
            ):
                raise ValueError(f"invalid generated options for {q.name}")
            q = replace(q, options=candidates)
        questions[q.name] = q
    return {"state": state, "questions": questions, "slot": slot, "source": "synth",
            "meta": {"grid": slot["grid"], "persona": persona, "intent": slot["intents"] or None,
                     "near_miss": near_miss}}


def _generate(cfg, teacher, writer, slots, votes, rng, concurrency, errors, references):
    if not slots:
        return [], [], {"rounds": 0, "invalid_generated": 0}
    personas = []
    if cfg.synth.personas:
        try:
            response = api.extract_json(api.chat(teacher, api._persona_prompt(cfg, cfg.synth.personas),
                                                  temperature=0.9, max_tokens=1500))
            if isinstance(response, list):
                personas = [str(p) for p in response[:cfg.synth.personas]]
        except (ValueError, TypeError, RuntimeError) as error:
            errors.append(f"personas: {error}")
    pending = list(slots)
    train, val = [], []
    stats = {"rounds": 0, "invalid_generated": 0}
    for attempt in range(cfg.synth.max_rounds):
        stats["rounds"] = attempt + 1
        buckets = defaultdict(list)
        for slot in pending:
            buckets[json.dumps([slot["grid"], slot["intents"]], sort_keys=True)].append(slot)
        batches, prompts = [], []
        for bucket in buckets.values():
            for start in range(0, len(bucket), api.STATES_PER_CALL):
                batch = bucket[start:start + api.STATES_PER_CALL]
                persona = rng.choice(personas) if personas else "a typical user"
                refs = rng.sample(references, min(2, len(references))) if references else []
                near = None
                if cfg.synth.distractors == "near_miss" and rng.random() < cfg.synth.distractor_share:
                    choice = next((q for q in cfg.questions if q.type == "choice"), None)
                    if choice:
                        desired = batch[0]["intents"].get(choice.name)
                        alternatives = [description for key, description in choice.options.items() if key != desired]
                        near = rng.choice(alternatives) if desired and alternatives else choice.prompt
                prompts.append(api._state_prompt(cfg, batch[0]["grid"], persona, None, near, refs, len(batch),
                                                 intents=batch[0]["intents"]))
                batches.append((batch, persona, near))
        responses = api.chat_many(writer, prompts, temperature=0.9,
                                  max_tokens=max(2000, 600 * api.STATES_PER_CALL * (1 + sum(q.dynamic_options for q in cfg.questions))),
                                  concurrency=concurrency, on_progress=lambda i, message: errors.append(message))
        jobs = []
        for response, (batch, persona, near) in zip(responses, batches):
            try:
                values = api.extract_json(response)
                if not isinstance(values, list):
                    raise ValueError("writer must return an array")
            except (ValueError, TypeError):
                stats["invalid_generated"] += len(batch)
                continue
            for raw, slot in zip(values, batch):
                try:
                    jobs.append(_generated_job(raw, cfg, slot, persona, near))
                except (ValueError, TypeError):
                    stats["invalid_generated"] += 1
        annotations = _annotate(cfg, teacher, jobs, votes, concurrency, errors)
        done = set()
        for job, labels in zip(jobs, annotations):
            slot = job["slot"]
            if any(name not in labels for name in slot["wanted"]):
                continue
            if any(label_key(labels[name]["label"]) != answer for name, answer in slot["intents"].items()):
                continue
            destination = val if slot["split"] == "val" else train
            destination.extend(labels[name] for name in slot["wanted"])
            done.add(slot["id"])
        pending = [slot for slot in pending if slot["id"] not in done]
        print(f"round {attempt + 1}: {len(slots) - len(pending)}/{len(slots)} shared states completed")
        if not pending:
            return train, val, stats
    missing = Counter(name for slot in pending for name in slot["wanted"])
    raise ValueError(f"generation quotas not met after {cfg.synth.max_rounds} rounds: {dict(missing)}; "
                     "no output replaced. Adjust quotas/ratios or --max-rounds.")


def _assign_splits(groups, fraction, rng, old_train, old_val):
    existing = {state_key(row["state"]): "train" for row in old_train}
    for row in old_val:
        key = state_key(row["state"])
        # Repeated observations may already occur in both splits. Keep all old
        # rows in place and prefer train for a newly imported matching state.
        existing.setdefault(key, "val")
    fresh = list(dict.fromkeys(state_key(group["state"]) for group in groups if state_key(group["state"]) not in existing))
    rng.shuffle(fresh)
    n_val = int(round(len(fresh) * fraction))
    existing.update({key: "val" if i < n_val else "train" for i, key in enumerate(fresh)})
    for group in groups:
        group["split"] = existing[state_key(group["state"])]


def _imported(cfg, active, teacher, groups, mode, counts, votes, rng, concurrency, errors):
    jobs, baseline = [], []
    for group in groups:
        old = {name: row for row in group["records"] if (name := _name(row, cfg)) is not None}
        needed = [name for name in group["questions"]
                  if (mode == "relabel" or not _labeled(old.get(name, {}))) and counts.get(name, 1) != 0]
        baseline.extend((group["split"], row) for row in group["records"] if _labeled(row))
        if needed:
            jobs.append({**group, "needed": needed, "old": old, "source": "teacher"})
    pending = list(jobs)
    candidates = []
    for attempt in range(cfg.synth.max_rounds):
        if not pending:
            break
        labels = _annotate(active, teacher, pending, votes, concurrency, errors)
        remaining = []
        for job, answers in zip(pending, labels):
            for name in list(job["needed"]):
                if name not in answers:
                    continue
                old = job["old"].get(name, {})
                row = copy.deepcopy(old)
                row.pop("label", None)
                row.pop("label_probs", None)
                row.update(answers[name])
                row["meta"] = {**old.get("meta", {}), **answers[name]["meta"]}
                if "source" in old:
                    row["meta"]["input_source"] = old["source"]
                candidates.append((job["split"], name, row, old))
                job["needed"].remove(name)
            if job["needed"]:
                remaining.append(job)
        pending = remaining
    if pending:
        missing = Counter(name for job in pending for name in job["needed"])
        raise ValueError(f"Teacher did not label all requested questions: {dict(missing)}; no output replaced")
    chosen_ids = set()
    for q in active.questions:
        train = [row for split, name, row, _ in candidates if split == "train" and name == q.name]
        val = [row for split, name, row, _ in candidates if split == "val" and name == q.name]
        total = counts.get(q.name)
        n_val = None if total is None else min(len(val), int(round(total * cfg.synth.val_fraction)))
        n_train = None if total is None else min(len(train), total - n_val)
        if total is not None:
            n_val = total - n_train
        try:
            # No label-aware selection is ever applied to validation rows.
            chosen = select_rows(val, n_val, None, rng)
            chosen += select_rows(train, n_train, cfg.synth.answer_ratios.get(q.name), rng)
        except ValueError as error:
            raise ValueError(f"question {q.name}: {error}") from error
        chosen_ids.update(id(row) for row in chosen)
    replaced = {id(old) for _, _, row, old in candidates if id(row) in chosen_ids and old}
    output = {"train": [], "val": []}
    for split, row in baseline:
        if id(row) not in replaced:
            output[split].append(row)
    for split, _, row, _ in candidates:
        if id(row) in chosen_ids:
            output[split].append(row)
    return output["train"], output["val"], {"labeled": len(chosen_ids), "retained": len(baseline) - len(replaced)}


def run(cfg, teacher, writer, n, out, *, dry_run=False, votes=None, seed=None, concurrency=8,
        mode=None, input_path=None, questions=None, types=None, counts=None, grid_ratios=None,
        answer_ratios=None, max_rounds=None):
    cfg = copy.deepcopy(cfg)
    cfg.synth.mode = mode if mode is not None else cfg.synth.mode
    cfg.synth.questions = questions if questions is not None else cfg.synth.questions
    cfg.synth.types = types if types is not None else cfg.synth.types
    cfg.synth.counts.update(counts or {})
    cfg.synth.grid.update(grid_ratios or {})
    cfg.synth.answer_ratios.update(answer_ratios or {})
    if max_rounds is not None:
        cfg.synth.max_rounds = max_rounds
    # Roundtrip canonicalizes YAML boolean answer keys and validates direct overrides.
    cfg = LuceConfig.from_dict(cfg.to_dict(), path=cfg.path)
    cfg.validate()
    if not 0 <= cfg.synth.val_fraction < 1:
        raise ValueError("synth.val_fraction must be >= 0 and < 1")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 0):
        raise ValueError("--n must be a nonnegative integer")
    selected = [q for q in cfg.questions if (not cfg.synth.questions or q.name in cfg.synth.questions)
                and (not cfg.synth.types or q.type in cfg.synth.types)]
    if not selected:
        raise ValueError("question/type selection is empty")
    # An explicit zero is a request to produce no records for this question.
    # It must not require dynamic candidates or consume Teacher calls.
    selected = [q for q in selected if cfg.synth.counts.get(q.name) != 0]
    active = replace(cfg, questions=selected)
    mode = cfg.synth.mode
    input_path = input_path if input_path is not None else _path(cfg.synth.input, cfg)
    if mode in ("label", "relabel") and not input_path:
        raise ValueError(f"--mode {mode} needs --input JSONL or synth.input")
    if mode == "new" and input_path:
        raise ValueError("--input is for label, relabel, or append; use --mode label for existing text")
    importing = bool(input_path)
    writer = writer or cfg.synth.writer or teacher
    votes = votes if votes is not None else teacher.votes or 3
    if isinstance(votes, bool) or not isinstance(votes, int) or votes < 1:
        raise ValueError("votes must be a positive integer")
    rng = random.Random(cfg.synth.seed if seed is None else seed)
    old_train, old_val = read_output(out) if mode == "append" else ([], [])
    destinations = {os.path.realpath(os.path.join(out, name)) for name in ("train.jsonl", "val.jsonl", "synth_report.json")}
    for protected in (input_path, _path(cfg.eval.real, cfg), _path(cfg.examples, cfg)):
        if protected and os.path.realpath(protected) in destinations:
            raise ValueError("output would overwrite an input/reference/evaluation file; choose another --out directory")
    quotas = {q.name: cfg.synth.counts[q.name] for q in selected if q.name in cfg.synth.counts}
    if importing:
        groups = read_inputs(input_path, cfg, selected if n != 0 else [])
        if n is not None:
            # Limit work, while retaining already-labeled rows outside the selected prefix.
            for group in groups[n:]:
                group["questions"] = {}
        _assign_splits(groups, cfg.synth.val_fraction, rng, old_train, old_val)
        states_requested = sum(bool(g["questions"]) for g in groups)
        slots = []
    else:
        default_count = n if n is not None else cfg.synth.n
        if default_count < 0:
            raise ValueError("synth.n must be nonnegative")
        quotas = {q.name: quotas.get(q.name, default_count) for q in selected}
        slots = new_slots(quotas, cfg.synth.grid, cfg.synth.answer_ratios, cfg.synth.val_fraction, rng)
        states_requested = len(slots)
        groups = []
    plan = {"mode": mode, "input": input_path, "questions": [q.name for q in selected],
            "states": states_requested, "counts": quotas, "answer_distribution": cfg.synth.answer_ratios or "natural",
            "grid": cfg.synth.grid, "max_rounds": cfg.synth.max_rounds,
            "teacher": teacher.describe(), "writer": None if importing else writer.describe()}
    print("plan:", json.dumps(plan, ensure_ascii=False))
    if importing:
        writer_calls = 0
        label_calls = sum(bool(group["questions"]) for group in groups) * votes
        input_chars = sum(len(json.dumps(group["state"], ensure_ascii=False)) for group in groups)
    else:
        buckets = Counter(json.dumps([s["grid"], s["intents"]], sort_keys=True) for s in slots)
        writer_calls = sum(math.ceil(count / api.STATES_PER_CALL) for count in buckets.values())
        label_calls = len(slots) * votes
        input_chars = len(slots) * 400
    prompt_tokens = api.estimate_tokens(json.dumps(active.to_dict(), ensure_ascii=False))
    tokens_in = (writer_calls + label_calls) * prompt_tokens + api.estimate_tokens("x" * min(input_chars, 1000000)) * votes
    tokens_out = (states_requested * 120 if not importing else 0) + label_calls * len(selected) * 30
    print(f"initial plan: {writer_calls} writer calls, up to {label_calls} label calls; ~{tokens_in:,} input / {tokens_out:,} output tokens. Cost = tokens × your endpoint rates; retries may add calls.")
    if dry_run:
        print("dry run: no API calls or output writes made.")
        return
    errors = []
    if importing:
        train, val, stats = _imported(cfg, active, teacher, groups, "relabel" if mode == "relabel" else "label",
                                     quotas, votes, rng, concurrency, errors)
    else:
        references = [row["state"] for row in old_train + old_val]
        examples_path = _path(cfg.examples, cfg)
        if examples_path and os.path.exists(examples_path):
            with open(examples_path, encoding="utf-8") as handle:
                references.extend(row["state"] for line in handle if line.strip()
                                  if isinstance((row := json.loads(line)), dict) and "state" in row)
        train, val, stats = _generate(active, teacher, writer, slots, votes, rng, concurrency, errors, references)
    if mode == "append":
        # Append preserves every occurrence, including records equal to existing rows.
        train, val = old_train + train, old_val + val
    distribution = {}
    for split, rows in (("train", train), ("val", val)):
        distribution[split] = {q.name: dict(Counter(label_key(row.get("label")) for row in rows if _name(row, cfg) == q.name)) for q in selected}
    state_grids = {state_key(row["state"]): (row.get("meta") or {}).get("grid", {}) for row in train + val}
    report = {**plan, **stats, "n_requested": states_requested, "records_train": len(train), "records_val": len(val),
              "votes": votes, "call_errors": errors[:20], "label_distribution": distribution,
              "question_counts": {split: {q.name: sum(_name(row, cfg) == q.name for row in rows) for q in selected}
                                  for split, rows in (("train", train), ("val", val))},
              "grid_distribution": {axis: dict(Counter(grid[axis] for grid in state_grids.values() if axis in grid))
                                    for axis in cfg.synth.grid},
              "answer_ratios_scope": "training only; validation is never label-balanced"}
    write_output(out, train, val, report)
    print(f"wrote {len(train)} -> {out}/train.jsonl, {len(val)} -> {out}/val.jsonl")
    print(f"report: {out}/synth_report.json")
