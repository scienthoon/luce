"""
luce.synth — generate training data for a luce task with a teacher LLM.

Pipeline (each stage prints its counts):
  1. personas       one call: N short author personas
  2. states         writer LLM with weighted grid cells and shared questions; no intended answer by default.
                    Explicit answer ratios target training states only. Imported states skip generation.
  3. labels         teacher answers all questions per state in one call, `votes` times; majority label; disagreement
                    below min_agreement -> dropped, or kept as soft label (label_probs = vote fractions, meta.hard)
  4. output         selected questions/quotas; shared-state train/val split, or append to existing splits

`--dry-run` prints the planned calls and a token estimate and stops before any API call.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Endpoint, LuceConfig, QuestionSpec
from .llm import chat, chat_many, estimate_tokens, extract_json

STATES_PER_CALL = 5


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def _persona_prompt(cfg: LuceConfig, n: int) -> List[Dict[str, str]]:
    return [{"role": "system", "content": "You invent short, concrete author personas for realistic input data. Return ONLY a JSON array of strings."},
            {"role": "user", "content": f"Task: {cfg.task.description}\nWrite {n} distinct personas of people who would produce inputs for this task "
                                        f"(one line each: role, situation, writing habits). Vary age, fluency, mood, and level of detail."}]


def _state_prompt(cfg: LuceConfig, cell: Dict[str, str], persona: str, intent: Optional[Tuple[QuestionSpec, str]],
                  near_miss: Optional[str], seeds: List[Any], k: int,
                  intents: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    fields = cfg.task.state_fields
    shape = ("a JSON object with exactly these fields: " + ", ".join(fields)) if fields else "a string"
    lines = [f"Task: {cfg.task.description}", f"Write {k} realistic, distinct inputs. Each input is {shape}.",
             "Attributes every input must have: " + ", ".join(f"{a}={v}" for a, v in cell.items()),
             f"Author persona: {persona}"]
    lines.append("The inputs will be used for ALL of these questions (do not answer them in the input):")
    for q in cfg.questions:
        detail = json.dumps(q.options, ensure_ascii=False) if q.type == "choice" else json.dumps(q.levels, ensure_ascii=False) if q.type == "score" else "true / false"
        lines.append(f"- {q.name} ({q.type}): {q.prompt}; answers: {detail}")
    if not intent and not intents:
        lines.append("Use naturally occurring situations for the task; do not balance or force answer classes.")
    if intent is not None:
        q, key = intent
        lines.append(f"Every input must clearly belong to this answer for the question \"{q.prompt}\": {key} = {q.options[key]}.")
    if near_miss:
        if intent or intents:
            lines.append(f"Make each input a near miss: it should superficially resemble \"{near_miss}\" (mention it in passing) while truly being about the requested answer.")
        else:
            lines.append(f"Include realistic ambiguity relevant to {near_miss}; do not prescribe which answer is correct.")
    for name, label in (intents or {}).items():
        q = cfg.question(name)
        detail = q.options.get(label, label) if q.type == "choice" else q.levels[int(label)] if q.type == "score" else label
        lines.append(f"Requested training example for {name}: the correct answer should be {label} ({detail}). The teacher will independently verify this.")
    if seeds:
        lines.append("Style references (imitate tone and length; do NOT copy or paraphrase them):")
        for s in seeds:
            lines.append("  - " + json.dumps(s, ensure_ascii=False)[:400])
    dynamic = [q for q in cfg.questions if q.dynamic_options]
    if dynamic:
        lines.append('Return ONLY a JSON array of ' + str(k) + ' objects, each with "state" containing the input and "options" mapping question names to candidate dictionaries.')
        for q in dynamic:
            lines.append(f'options["{q.name}"] must contain at least two distinct string IDs mapped to nonempty candidate descriptions. Generate candidates specific to this state. {q.options_prompt}')
    else:
        lines.append("Do not number the inputs. Do not include answer labels in the text. Return ONLY a JSON array of " + str(k) + (" objects." if fields else " strings."))
    return [{"role": "system", "content": "You generate realistic, varied data. Output JSON only."}, {"role": "user", "content": "\n".join(lines)}]


def _label_prompt(cfg: LuceConfig, state: Any, questions: Optional[Sequence[QuestionSpec]] = None) -> List[Dict[str, str]]:
    qs = []
    for q in cfg.questions if questions is None else questions:
        if q.type == "choice":
            qs.append(f'- "{q.name}" (choice): {q.prompt}\n    options: ' + "; ".join(f"{k} = {v}" for k, v in q.options.items()) + "\n    answer with the option key.")
        elif q.type == "score":
            qs.append(f'- "{q.name}" (score): {q.prompt}\n    levels (lowest first): ' + "; ".join(f"{i} = {l}" for i, l in enumerate(q.levels)) + "\n    answer with the level index (integer).")
        else:
            qs.append(f'- "{q.name}" (true/false): {q.prompt}\n    answer with true or false.')
    user = (f"Task: {cfg.task.description}\n\nInput:\n{json.dumps(state, ensure_ascii=False) if not isinstance(state, str) else state}\n\n"
            "Answer every question. Return ONLY a JSON object mapping question name to answer.\n" + "\n".join(qs))
    return [{"role": "system", "content": "You are a careful annotator. Output JSON only."}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _grid_cells(grid: Dict[str, List[str]], n: int, rng: random.Random) -> List[Dict[str, str]]:
    from .synth_plan import grid_cells
    return grid_cells(grid, n, rng)


def _coerce_answer(q: QuestionSpec, raw: Any) -> Optional[Any]:
    if q.type == "choice":
        if isinstance(raw, str):
            if raw in q.options:
                return raw
            low = raw.strip().lower()
            for k, v in q.options.items():
                if low == k.lower() or low == v.lower():
                    return k
        return None
    if q.type == "score":
        try:
            i = int(raw)
        except (TypeError, ValueError):
            if isinstance(raw, str):
                for i, l in enumerate(q.levels):
                    if raw.strip().lower() == l.lower():
                        return i
            return None
        return i if 0 <= i < len(q.levels) else None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v in ("true", "yes", "1"): return True
        if v in ("false", "no", "0"): return False
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(raw)
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_synth(cfg: LuceConfig, teacher: Endpoint, writer: Optional[Endpoint], n: Optional[int], out: str,
              dry_run: bool = False, votes: Optional[int] = None, seed: Optional[int] = None, concurrency: int = 8,
              *, mode: Optional[str] = None, input_path: Optional[str] = None,
              questions: Optional[List[str]] = None, types: Optional[List[str]] = None,
              counts: Optional[Dict[str, int]] = None, grid_ratios: Optional[Dict[str, Any]] = None,
              answer_ratios: Optional[Dict[str, Any]] = None, max_rounds: Optional[int] = None) -> None:
    from .synth_run import run
    run(cfg, teacher, writer, n, out, dry_run=dry_run, votes=votes, seed=seed, concurrency=concurrency,
        mode=mode, input_path=input_path, questions=questions, types=types, counts=counts,
        grid_ratios=grid_ratios, answer_ratios=answer_ratios, max_rounds=max_rounds)
