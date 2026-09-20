"""
luce.scaffold — `luce init`: turn a task description (+ optional seed examples) into luce.yaml.

With a teacher the questions, options, state fields and diversity grid are drafted by the LLM and written for you to
edit. Without one, a commented template is written. Seed examples are synth references only; 20 or more are split
half/half into seeds.jsonl (synth references) and real.jsonl (evaluation) so the evaluation set never overlaps the seeds.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List, Optional

from .config import Endpoint, EvalSpec, LuceConfig, ModelSpec, QuestionSpec, ServeSpec, SynthSpec, TaskSpec

DRAFT_PROMPT = """You design decision tasks for a small classifier. The user describes a task; you return a JSON spec.

Rules:
- Every question is one of three types: "choice" (pick one of named options), "score" (an ordered scale, 2..10 levels,
  lowest first), "noul" (a statement that is true or false about the input).
- Options and levels are short descriptions a non-expert would understand; keys are snake_case identifiers.
- Include only questions the input text can actually answer. 1 to 5 questions.
- "state_fields": the fields an input record has (e.g. ["subject", "body"]); [] if the input is free text.
- "grid": 2 to 5 attributes that should vary across synthetic inputs, each with 2..6 SHORT categorical values
  (e.g. channel: [email, chat], sentiment: [calm, frustrated], length: [short, long], locale: [en, ko], topic: [...]).
  Never put free-text fields (subject, body, message...) or example sentences into the grid; the grid describes variation, not content.
- "name": short snake_case task name.

Return ONLY JSON:
{"name": "...", "state_fields": [...], "questions": {"<name>": {"type": "choice", "prompt": "...", "options": {"key": "description"}},
 "<name2>": {"type": "score", "prompt": "...", "levels": ["Low", "High"]}, "<name3>": {"type": "noul", "prompt": "..."}},
 "grid": {"attribute": ["value", "value"]}}"""


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:40] or "task"


def _load_examples(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def draft_with_teacher(description: str, examples: List[Dict[str, Any]], teacher: Endpoint) -> Dict[str, Any]:
    from .llm import chat, extract_json
    sample = "\n".join(json.dumps(r, ensure_ascii=False)[:500] for r in examples[:8])
    user = f"Task description:\n{description}\n"
    if sample:
        user += f"\nExample records (may include labels; use them to infer the questions and answer sets):\n{sample}\n"
    reply = chat(teacher, [{"role": "system", "content": DRAFT_PROMPT}, {"role": "user", "content": user}], temperature=0.2, max_tokens=1500)
    spec = extract_json(reply)
    if not isinstance(spec, dict) or "questions" not in spec:
        raise ValueError(f"teacher returned an unusable spec: {reply[:300]!r}")
    return spec


def template_spec(description: str) -> Dict[str, Any]:
    return {
        "name": _slug(description),
        "state_fields": [],
        "questions": {
            "category": {"type": "choice", "prompt": "TODO: what to decide? e.g. Which category does this belong to?",
                         "options": {"option_a": "TODO describe option A", "option_b": "TODO describe option B"}},
            "priority": {"type": "score", "prompt": "TODO: e.g. How urgent is this?", "levels": ["Low", "Normal", "High"]},
            "flag": {"type": "noul", "prompt": "TODO: a statement that is true or false about the input."},
        },
        "grid": {"length": ["short", "long"], "tone": ["neutral", "frustrated"]},
    }


def run_init(description: str, examples: Optional[str], teacher: Optional[Endpoint], out: str, name: Optional[str] = None) -> LuceConfig:
    rows = _load_examples(examples) if examples else []
    if teacher is not None:
        print(f"drafting spec with {teacher.describe()} ...")
        spec = draft_with_teacher(description, rows, teacher)
    else:
        print("no --teacher: writing a template luce.yaml with TODO placeholders (edit questions/options before `luce synth`)")
        spec = template_spec(description)

    questions: List[QuestionSpec] = []
    for qname, q in (spec.get("questions") or {}).items():
        questions.append(QuestionSpec(name=str(qname), type=str(q.get("type", "")).lower(), prompt=str(q.get("prompt", "")),
                                      options={str(k): str(v) for k, v in (q.get("options") or {}).items()},
                                      levels=[str(x) for x in (q.get("levels") or [])]))
    task = TaskSpec(name=name or str(spec.get("name") or _slug(description)), description=description,
                    state_fields=[str(x) for x in (spec.get("state_fields") or [])])
    grid = {}
    # 씨앗 레코드의 필드 값을 그대로 grid 값으로 가져온 경우(예: subject 문장)는 변주 축이 아니라 내용 복사 → 버린다
    seed_values: Dict[str, set] = {}
    for r in rows:
        st = r.get("state")
        if isinstance(st, dict):
            for f, val in st.items():
                seed_values.setdefault(str(f), set()).add(str(val).strip().lower())
    for k, v in (spec.get("grid") or {}).items():
        values = [str(x).strip() for x in (v or []) if str(x).strip()][:8]
        if len(values) < 2:
            continue
        copied = [x for x in values if x.lower() in seed_values.get(str(k), set())]
        if str(k) in seed_values and len(copied) >= max(1, len(values) // 2) and any(len(x) > 12 or len(x.split()) > 2 for x in values):
            print(f"grid: dropped '{k}' (values copied from the example field, not an attribute)")
            continue
        if sum(len(x) for x in values) / len(values) > 25 or any(len(x.split()) > 4 for x in values):
            print(f"grid: dropped '{k}' (values look like free text, not attributes)")
            continue
        grid[str(k)] = values
    # 변주 축이 3개 미만이면 감정·길이 축을 기본으로 채운다 (없으면 생성물이 한 톤으로 쏠린다: 화남 49/11 같은 분포)
    for key, values in (("sentiment", ["calm", "frustrated"]), ("length", ["short", "long"])):
        if len(grid) >= 3:
            break
        if key not in grid and not any(k in ("tone", "mood", "emotion") for k in grid) if key == "sentiment" else key not in grid:
            grid[key] = values
            print(f"grid: added default axis '{key}': {values}")

    out_dir = os.path.dirname(os.path.abspath(out))
    seeds_path: Optional[str] = None
    real_path: Optional[str] = None
    if rows:
        rng = random.Random(0)
        # 같은 state(입력)에 붙은 여러 질문 레코드는 한 그룹으로 움직인다: 한 티켓이 seeds 와 real 양쪽에 갈리면 누수.
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(json.dumps(r.get("state"), sort_keys=True, ensure_ascii=False), []).append(r)
        keys = list(groups.keys()); rng.shuffle(keys)
        if len(keys) >= 20:
            half = len(keys) // 2
            seeds_path = os.path.join(out_dir, "seeds.jsonl"); real_path = os.path.join(out_dir, "real.jsonl")
            n_seed = n_real = 0
            with open(seeds_path, "w", encoding="utf-8") as h:
                for k in keys[:half]:
                    for r in groups[k]: h.write(json.dumps(r, ensure_ascii=False) + "\n"); n_seed += 1
            with open(real_path, "w", encoding="utf-8") as h:
                for k in keys[half:]:
                    for r in groups[k]: h.write(json.dumps({**r, "source": "real"}, ensure_ascii=False) + "\n"); n_real += 1
            print(f"{len(keys)} example inputs -> {half} seeds ({n_seed} records, synth references only) + {len(keys) - half} real ({n_real} records, evaluation only): {seeds_path}, {real_path}")
        else:
            seeds_path = os.path.join(out_dir, "seeds.jsonl")
            with open(seeds_path, "w", encoding="utf-8") as h:
                for k in keys:
                    for r in groups[k]: h.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{len(keys)} example inputs -> seeds only ({seeds_path}). Give 20+ distinct inputs to get an automatic real.jsonl split, or label 100-300 real items for `luce eval --real`.")

    cfg = LuceConfig(task=task, questions=questions, examples=os.path.relpath(seeds_path, out_dir) if seeds_path else None,
                     synth=SynthSpec(teacher=teacher, grid=grid), model=ModelSpec(),
                     eval=EvalSpec(real=os.path.relpath(real_path, out_dir) if real_path else None), serve=ServeSpec())
    try:
        cfg.validate()
    except ValueError as error:
        print(f"warning: spec needs editing before use: {error}")
    cfg.save(out)
    print(f"wrote {out}: {len(questions)} questions ({', '.join(q.name + ':' + q.type for q in questions)}), grid {list(grid.keys())}")
    return cfg
