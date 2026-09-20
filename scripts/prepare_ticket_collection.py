"""Prepare rule references and a Teacher-synthesis config without editing Luce."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from luce.config import LuceConfig
from luce.convert import build_synthetic, _QUEUE_OPTIONS, _PRIORITY_LEVELS, _SYNTH_TEMPLATES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/four_tasks/rule_tickets"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        shutil.copyfile(ROOT / f"data/synth/{split}.jsonl", OUT / f"raw/original_{split}.jsonl")
    original = [json.loads(line) for line in (OUT / "raw/original_train.jsonl").read_text().splitlines()]
    observations = [original[index]["state"] for index in range(0, len(original), 3)]
    random.Random(20260920).shuffle(observations)
    seeds = [{"state": state, "source": "local_rule_seed"} for state in observations[:30]]
    write_rows(OUT / "seeds.jsonl", seeds)
    refs = build_synthetic(2000, label_noise=0, seed=20260920)
    write_rows(OUT / "raw/rule_reference_all.jsonl", refs)
    key = lambda state: json.dumps(state, sort_keys=True, ensure_ascii=False)
    groups = sorted({key(row["state"]) for row in refs})
    random.Random(20260920).shuffle(groups)
    assignment = {group: "val" if i < len(groups) // 4 else "calibration" if i < len(groups) // 2 else "test"
                  for i, group in enumerate(groups)}
    seed_keys = {key(row["state"]) for row in seeds}
    # Keep a repeated seed observation, but reserve it as a reference instead of holdout.
    assignment.update({group: "seed_reference" for group in seed_keys if group in assignment})
    output = {split: [] for split in ("val", "calibration", "test", "seed_reference")}
    for i, row in enumerate(refs):
        name = {"choice": "queue", "score": "priority", "noul": "angry"}[row["type"]]
        row.update(source="rule_simulator", meta={"question_name": name, "state_id": f"rule-{i//3}",
                    "source_group_id": hashlib.sha256(key(row["state"]).encode()).hexdigest(),
                    "label_noise": 0, "seed": 20260920})
        output[assignment[key(row["state"])]].append(row)
    for split, rows in output.items():
        write_rows(OUT / f"{split}.jsonl", rows)
    rules = """Route customer-support tickets and apply the organization's rules.
Queue: billing for payments/refunds/invoices; shipping for delivery/lost/damaged packages;
technical for application bugs/login/password/export failures; general for policy/product questions,
feedback and feature suggestions. Use customer_tier free, standard, gold or enterprise.
Angry is true when the customer expresses anger, an angry complaint, or threats to dispute or review;
a neutral request without anger is false. Priority levels are Low=0, Normal=1, High=2, Critical=3.
Start from the base urgency below. Add 1 if angry. After that, add 1 for gold or enterprise only
if the current priority is at least 1. Clamp to 0..3. Apply the same rules to Korean and English.
Base urgency examples (generalize the problem categories, do not copy the example wording):
"""
    for queue, templates in _SYNTH_TEMPLATES.items():
        rules += "\n" + queue + ": " + "; ".join(f"{t['subject']} -> base {t['urgency']}" for t in templates)
    (OUT / "rules.txt").write_text(rules + "\n")
    cfg = {
        "task": {"name": "organization_rule_tickets", "description": rules,
                 "state": {"fields": ["channel", "customer_tier", "subject", "body"]}},
        "questions": {
            "queue": {"type": "choice", "prompt": "Which support queue should handle this ticket?", "options": _QUEUE_OPTIONS},
            "priority": {"type": "score", "prompt": "How should this ticket be prioritized?", "levels": _PRIORITY_LEVELS},
            "angry": {"type": "noul", "prompt": "The customer sounds angry."},
        },
        "examples": "seeds.jsonl",
        "synth": {"n": 1000, "teacher": {"url": "https://openrouter.ai/api/v1", "model": "deepseek/deepseek-v4.1-flash", "votes": 1},
                  "personas": 16, "val_fraction": 0, "distractors": "none", "seed": 20260920, "max_rounds": 4,
                  "grid": {"language": {"ko": 1, "en": 1}, "length": ["short", "medium", "long"]}},
        "model": {"mode": "label"},
    }
    LuceConfig.from_dict(cfg).validate()
    (OUT / "luce.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    print("ticket references", {split: len(rows) for split, rows in output.items()}, "seeds", len(seeds))


if __name__ == "__main__":
    main()
