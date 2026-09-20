"""Read-only checks of delivered records; writes one validation summary."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from luce.config import LuceConfig
from luce.data import example_from_record

ROOT = Path(__file__).resolve().parents[1] / "data/four_tasks"


def main():
    results = {}
    for manifest_path in sorted(ROOT.rglob("dataset_manifest.json")):
        folder = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        state_sets, group_sets = {}, {}
        total = 0
        for split in ("train", "val", "calibration", "test", "ood"):
            path = folder / f"{split}.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            assert len(rows) == manifest[split]["records"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[split]["sha256"]
            for row in rows:
                ex = example_from_record(row)
                assert abs(sum(ex.target) - 1) < 1e-8
                assert all(0 <= p <= 1 for p in ex.target)
                if "options" in row:
                    assert len(row["options"]) >= 2
                if folder == ROOT / "maze" and row["type"] == "noul":
                    # Verify that the delivered compact observation is sufficient
                    # to reconstruct the target, without reading oracle metadata.
                    local = row["state"]["local_map"].splitlines()
                    deaths = 0
                    for moves in itertools.product(((-1, 0), (0, 1), (1, 0), (0, -1)), repeat=3):
                        r, c = 3, 3
                        for dr, dc in moves:
                            if local[r][c] in ("@", "G"):
                                break
                            r, c = r + dr, c + dc
                            if local[r][c] in ("#", "X"):
                                deaths += 1
                                break
                    assert deaths / 64 == row["label_probs"]
            state_sets[split] = {json.dumps(row["state"], sort_keys=True, ensure_ascii=False) for row in rows}
            group_sets[split] = {row["meta"]["source_group_id"] for row in rows}
            total += len(rows)
        for a, b in itertools.combinations(state_sets, 2):
            assert not group_sets[a] & group_sets[b], (folder, a, b, "source group overlap")
            # Official local windows can legitimately repeat geometry on different held-out maps.
            if "official" not in folder.parts:
                assert not state_sets[a] & state_sets[b], (folder, a, b, "same model input")
        config = folder / "luce.yaml"
        if config.exists():
            LuceConfig.load(str(config)).validate()
        results[str(folder.relative_to(ROOT))] = {"validated_records": total, "source_group_overlap": 0,
                                                   "hashes_match": True, "target_distributions_valid": True}
    source = ROOT / "github_issues/raw/issues.jsonl"
    issues = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(issues) == 6122 and len({row["id"] for row in issues}) == 6122
    coverage = json.loads((ROOT / "github_issues/label_coverage.json").read_text())
    assert coverage["kind_labeled"] + coverage["kind_unresolved"] == len(issues)
    assert coverage["priority_labeled"] + coverage["priority_unresolved"] == len(issues)
    assert coverage["duplicate_marker_true"] + coverage["duplicate_marker_false"] == len(issues)
    tickets = [json.loads(line) for line in (ROOT / "rule_tickets/train.jsonl").read_text().splitlines()]
    assert len(tickets) == 3000
    assert Counter(row["meta"]["question_name"] for row in tickets) == {"queue": 1000, "priority": 1000, "angry": 1000}
    # The secret is never part of collected files; do not print any matching content.
    secret_files = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()
                    and b"sk-or-v1-" in path.read_bytes()]
    assert not secret_files, "API credential pattern found in collected artifacts"
    report = {"status": "passed", "collections": results,
              "total_validated_records": sum(r["validated_records"] for r in results.values()),
              "github_source_issues": len(issues), "api_key_written": False,
              "training_executed": False, "luce_source_modified": False}
    (ROOT / "validation_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
