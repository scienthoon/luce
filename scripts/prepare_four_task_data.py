"""Convert collected sources to Luce records and construct exact maze targets.

No changes to Luce and no model training. Source observations are retained in raw/.
Repeated content is never removed; split grouping only prevents holdout overlap.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random
import shutil
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from luce.config import LuceConfig
from luce.data import example_from_record

ROOT = Path(__file__).resolve().parents[1] / "data/four_tasks"
SEED = 20260920
SPLITS = ("train", "val", "calibration", "test")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(key(value).encode()).hexdigest()


def write_rows(path, rows, validate=True):
    if validate:
        for row in rows:
            example_from_record(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def save_config(folder, questions, description, state_fields=None):
    cfg = {"task": {"name": folder.name, "description": description}, "questions": questions,
           "model": {"mode": "label"}}
    if state_fields:
        cfg["task"]["state"] = {"fields": state_fields}
    LuceConfig.from_dict(cfg).validate()
    (folder / "luce.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))


def stats(folder):
    result = {}
    groups = {}
    for split in (*SPLITS, "ood"):
        path = folder / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = read_rows(path)
        for row in rows:
            example_from_record(row)
        groups[split] = {r["meta"]["source_group_id"] for r in rows}
        result[split] = {"records": len(rows), "observations": len({r["meta"]["state_id"] for r in rows}),
            "groups": len(groups[split]), "types": dict(Counter(r["type"] for r in rows)),
            "questions": dict(Counter(r["meta"]["question_name"] for r in rows)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for a, b in itertools.combinations(groups, 2):
        assert not groups[a] & groups[b], (folder, a, b, "group overlap")
    result["split_group_overlap"] = 0
    write_json(folder / "dataset_manifest.json", result)
    print(folder.relative_to(ROOT), json.dumps(result, ensure_ascii=False), flush=True)
    return result


def prepare_phishing():
    folder = ROOT / "phishing"
    path = folder / "raw/core_emails.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "cebb407ff8630491a97400e37464b8db8dfc4299164fca51fcb4ac7eec8204ef"
    with path.open() as handle:
        source = list(csv.DictReader(handle))
    assert len(source) == 2000
    source_by_label = {label: [r for r in source if int(r["phish_label"]) == label] for label in (0, 1)}
    split_rows = {s: [] for s in SPLITS}
    for label, values in source_by_label.items():
        random.Random(SEED + label).shuffle(values)
        for split, start, end in (("train", 0, 500), ("val", 500, 625), ("calibration", 625, 750), ("test", 750, 1000)):
            split_rows[split].extend(values[start:end])
    q = {"type": "noul", "prompt": "This email is phishing rather than a legitimate email."}
    cq = {"type": "choice", "prompt": "Classify this email as phishing or legitimate.",
          "options": {"phishing": "Phishing email", "legitimate": "Legitimate email"}}
    def convert(row, choice=False):
        email = json.loads(row["email_content"])
        # Only email fields are model input. Reputation labels/taxonomy stay in metadata.
        state = {field: email[field] for field in ("sender", "from", "subject", "body", "link_display_text", "link_url")}
        meta = {"state_id": row["id"], "source_group_id": digest(state), "question_name": "verdict" if choice else "is_phishing",
                "upstream_id": row["id"], "datasource": row["datasource"], "strategy": row["strategy"],
                "url_category": row["url_category"], "supervision": "upstream_phish_label"}
        if choice:
            return {"state": state, "type": "choice", "question": cq["prompt"], "options": cq["options"],
                    "label": "phishing" if int(row["phish_label"]) else "legitimate", "source": "PhishNChips_v5.2", "meta": meta}
        return {"state": state, "type": "noul", "question": q["prompt"], "label": bool(int(row["phish_label"])),
                "source": "PhishNChips_v5.2", "meta": meta}
    for split, values in split_rows.items():
        random.Random(SEED).shuffle(values)
        write_rows(folder / f"{split}.jsonl", [convert(row) for row in values])
        write_rows(folder / "choice" / f"{split}.jsonl", [convert(row, True) for row in values])
    with (folder / "raw/real_phishing_validation.csv").open() as handle:
        real = list(csv.DictReader(handle))
    assert all(row["phish_label"] == "1" for row in real)
    write_rows(folder / "human_phishing_positive_only.jsonl", [convert(row) for row in real])
    save_config(folder, {"is_phishing": q}, "PhishNChips v5.2 core phishing classification.")
    save_config(folder / "choice", {"verdict": cq}, "Choice view of the same PhishNChips observations and splits.")
    stats(folder); stats(folder / "choice")


def prepare_github():
    folder = ROOT / "github_issues"
    source = read_rows(folder / "raw/issues.jsonl")
    options = {"bug": "Bug report", "feature": "Feature request", "support": "Support question", "documentation": "Documentation issue"}
    priorities = ["priority/backlog", "priority/important-longterm", "priority/important-soon", "priority/critical-urgent"]
    kinds = {"kind/" + name: name for name in options}
    q = {"kind": {"type": "choice", "prompt": "What kind of Kubernetes issue is this?", "options": options},
         "priority": {"type": "score", "prompt": "What priority does Kubernetes triage assign to this issue?",
                      "levels": ["Backlog", "Important long-term", "Important soon", "Critical urgent"]}}
    proxyq = {"type": "noul", "prompt": "This issue was marked as a duplicate by Kubernetes repository triage."}
    output, proxy = {s: [] for s in SPLITS}, {s: [] for s in SPLITS}
    pending = []
    coverage = Counter()
    for issue in source:
        state = {"repository": "kubernetes/kubernetes", "title": issue["title"], "body": issue.get("body") or ""}
        group = digest(state)
        bucket = int(group[:8], 16) / 2**32
        split = "train" if bucket < .6 else "val" if bucket < .7 else "calibration" if bucket < .8 else "test"
        labels = [label["name"] for label in issue["labels"]]
        meta = {"state_id": str(issue["id"]), "source_group_id": group, "issue_number": issue["number"],
                "url": issue["html_url"], "created_at": issue["created_at"], "upstream_labels": labels,
                "supervision": "repository_labels_at_collection_time"}
        def row(name, spec):
            result = {"state": state, "type": spec["type"], "question": spec["prompt"], "source": "kubernetes_issues",
                      "meta": {**meta, "question_name": name}}
            for field in ("options", "levels"):
                if field in spec:
                    result[field] = spec[field]
            return result
        found_kind = [kinds[label] for label in labels if label in kinds]
        found_priority = [priorities.index(label) for label in labels if label in priorities]
        for name, candidates in (("kind", found_kind), ("priority", found_priority)):
            record = row(name, q[name])
            if len(candidates) == 1:
                record["label"] = candidates[0]
                output[split].append(record)
                coverage[name + "_labeled"] += 1
            else:
                record["meta"].update(split=split, label_candidates=candidates,
                                      missing_reason="no_target_label" if not candidates else "multiple_target_labels")
                pending.append(record)
                coverage[name + "_unresolved"] += 1
        record = row("duplicate_marked", proxyq)
        record["label"] = "triage/duplicate" in labels
        record["meta"]["supervision"] = "observed_duplicate_marker_only_not_semantic_duplicate_ground_truth"
        proxy[split].append(record)
        coverage["duplicate_marker_true" if record["label"] else "duplicate_marker_false"] += 1
    for split in SPLITS:
        write_rows(folder / f"{split}.jsonl", output[split])
        write_rows(folder / "duplicate_marker_proxy" / f"{split}.jsonl", proxy[split])
    write_rows(folder / "unresolved_labels.jsonl", pending, validate=False)
    write_rows(folder / "inputs.jsonl", [{"state": {"repository": "kubernetes/kubernetes", "title": r["title"], "body": r.get("body") or ""},
               "record_id": str(r["id"]), "meta": {"url": r["html_url"]}} for r in source], validate=False)
    write_json(folder / "label_coverage.json", {"issues": len(source), **coverage})
    save_config(folder, q, "Kubernetes issue type and priority from actual repository labels.")
    save_config(folder / "duplicate_marker_proxy", {"duplicate_marked": proxyq},
                "Predict the observed duplicate marker. An absent marker is NOT verified semantic non-duplication.")
    stats(folder); stats(folder / "duplicate_marker_proxy")


def convert_nano(record):
    converted = []
    for name, q in record["questions"].items():
        qtype = "noul" if q["type"] == "boolean" else q["type"]
        probs = record["gold_probs"][name]
        row = {"state": record["state"], "type": qtype, "question": q["instructions"], "source": "NanoJev_programmatic_gold",
               "meta": {"state_id": record["state_id"], "source_group_id": record["metadata"]["source_group_id"],
                        "question_name": name, "upstream_id": record["id"], "family_id": record["family_id"],
                        "target_semantics": record.get("gold_probs_kind", {}).get(name)}}
        if qtype == "choice":
            row.update(options=q["criteria"], label_probs=probs)
        elif qtype == "score":
            row.update(levels=q["criteria"], label_probs=[probs[str(i)] for i in range(len(q["criteria"]))])
        else:
            row["label_probs"] = probs["true"]
        if name in record.get("gold", {}):
            row["label"] = record["gold"][name]
        converted.append(row)
    return converted


def prepare_maze():
    folder = ROOT / "maze"
    raw = folder / "raw"
    # Verify all downloaded payloads covered by upstream SHA-256 manifests.
    checked = 0
    for manifest, prefix in ((raw / "manifest.json", raw), (raw / "games_v4/manifest.json", raw / "games_v4")):
        for rel, info in json.loads(manifest.read_text())["files"].items():
            path = prefix / rel
            if path.is_file():
                assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], path
                checked += 1
    write_json(folder / "upstream_integrity.json", {"verified_files": checked, "sha256_matches": True})
    variants = {
        "navigation": ("stage2", "grid_navigation_coordinates_v3"),
        "local_safety": ("games_v4/data/local_maze_v1", "scaled_maze"),
        "one_step_probability": ("games_v4/data/scaled_games_v4b/events", "scaled_maze_noisy_actuator"),
    }
    for variant, (directory, family) in variants.items():
        for split in ("train", "dev", "calibration", "test", "ood"):
            rows = [r for r in read_rows(raw / directory / f"{split}.jsonl") if r["family_id"] == family]
            converted = [item for row in rows for item in convert_nano(row)]
            write_rows(folder / "official" / variant / f"{'val' if split == 'dev' else split}.jsonl", converted)
        stats(folder / "official" / variant)

    directions = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}
    q = {
        "safe_move": {"type": "choice", "prompt": "Choose the safest first move: maximize survival through three total moves. After this first move, choose each of north/east/south/west uniformly at every step. A wall or boundary collision kills; reaching the goal stops safely.",
                      "options": {name: "Move one cell " + name for name in directions}},
        "death_within_three": {"type": "noul", "prompt": "The agent dies within three moves when every move independently chooses north/east/south/west uniformly. A wall or boundary collision kills; reaching the goal stops safely."},
        "risk": {"type": "score", "prompt": "Classify the exact probability of dying within three moves under the stated uniform random policy.",
                 "levels": ["Zero risk: p = 0", "Low risk: 0 < p <= 0.25", "Moderate risk: 0.25 < p <= 0.5", "High risk: 0.5 < p <= 1"]},
    }
    for split in ("train", "dev", "calibration", "test", "ood"):
        sources = read_rows(raw / "games_v4/data/scaled_games_v4b/policy" / f"{split}.jsonl")
        maps = {}
        for source in sources:
            if source["family_id"] == "scaled_maze":
                maps.setdefault(source["metadata"]["source_group_id"], source["metadata"]["environment_state"])
        output = []
        for group, env in maps.items():
            size, walls, goal = env["size"], set(map(tuple, env["walls"])), tuple(env["goal"])
            valid = {(r, c) for r in range(size) for c in range(size)} - walls

            @lru_cache(None)
            def death(position, remaining):
                if position not in valid:
                    return 1.0
                if position == goal or remaining == 0:
                    return 0.0
                return sum(death((position[0] + dr, position[1] + dc), remaining - 1)
                           for dr, dc in directions.values()) / 4

            positions = sorted(valid)
            random.Random(SEED + int(hashlib.sha256(group.encode()).hexdigest()[:8], 16)).shuffle(positions)
            for position in positions[:100]:
                p = death(position, 3)
                survival = {name: 1 - (death((position[0] + dr, position[1] + dc), 2) if position != goal else 0)
                            for name, (dr, dc) in directions.items()}
                best = max(survival.values())
                winners = [name for name, value in survival.items() if value == best]
                distribution = {name: 1 / len(winners) if name in winners else 0 for name in directions}
                local = []
                for r in range(position[0] - 3, position[0] + 4):
                    chars = []
                    for c in range(position[1] - 3, position[1] + 4):
                        cell = (r, c)
                        chars.append("@" if cell == position == goal else "A" if cell == position else
                                     "X" if not (0 <= r < size and 0 <= c < size) else
                                     "#" if cell in walls else "G" if cell == goal else ".")
                    local.append("".join(chars))
                state = {"size": size, "position": list(position), "goal": list(goal),
                         "local_map": "\n".join(local),
                         "legend": "7x7 centered on A. Rows increase south; columns east. # wall, X outside, . open, G goal, @ agent at goal.",
                         "horizon": 3, "policy": "Uniform independent N/E/S/W choices; collision is death; stop safely at goal."}
                state_id = "three_step:" + digest([group, position])
                meta = {"state_id": state_id, "source_group_id": group, "source_map_seed": env["seed"],
                        "exact_death_probability": p, "survival_probability_by_first_move": survival,
                        "target_method": "exact finite-horizon recursion, not Monte Carlo or LLM"}
                for name, spec in q.items():
                    row = {"state": state, "type": spec["type"], "question": spec["prompt"], "source": "NanoJev_maps_three_step_extension",
                           "meta": {**meta, "question_name": name}}
                    if name == "safe_move":
                        row.update(options=spec["options"], label=winners[0], label_probs=distribution)
                        row["meta"]["target_semantics"] = "uniform over safest moves; not independent action success probabilities"
                    elif name == "death_within_three":
                        row["label_probs"] = p
                    else:
                        row.update(levels=spec["levels"], label=0 if p == 0 else 1 if p <= .25 else 2 if p <= .5 else 3)
                    output.append(row)
                # Independent enumeration checks the probability on every selected state.
                deaths = 0
                for moves in itertools.product(directions.values(), repeat=3):
                    current = position
                    for dr, dc in moves:
                        if current == goal:
                            break
                        current = (current[0] + dr, current[1] + dc)
                        if current not in valid:
                            deaths += 1
                            break
                assert deaths / 64 == p, (group, position, p, deaths)
        write_rows(folder / f"{'val' if split == 'dev' else split}.jsonl", output)
    save_config(folder, q, "Three-step survival on official NanoJev maps, using the sufficient 7x7 local observation; exact simulator probability targets. This is a new task protocol, separate from official NanoJev scores.")
    stats(folder)


def finalize_tickets():
    folder = ROOT / "rule_tickets"
    source = read_rows(folder / "generated/train.jsonl")
    # Assign observation IDs per consecutive question group; retain every generated occurrence.
    for i, row in enumerate(source):
        row["meta"].update(state_id=f"llm-ticket-{i//3}", source_group_id=digest(row["state"]),
                           teacher_model="deepseek/deepseek-v4.1-flash")
    write_rows(folder / "train.jsonl", source)
    train_keys = {digest(row["state"]) for row in source}
    for split in ("val", "calibration", "test"):
        # Refuse to hide overlap by deleting observations.
        assert not train_keys & {digest(row["state"]) for row in read_rows(folder / f"{split}.jsonl")}, split
    stats(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["phishing", "github", "maze", "tickets"])
    task = parser.parse_args().task
    {"phishing": prepare_phishing, "github": prepare_github, "maze": prepare_maze, "tickets": finalize_tickets}[task]()
