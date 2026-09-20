"""Download public sources for the four-task data collection; never train models."""
from __future__ import annotations

import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1] / "data" / "four_tasks"
REVISIONS = {
    "phishing": ("AreLit/PhishNChips", "89afcc39610084298c4679159cb2e27d9ffffa46"),
    "maze": ("C-Tianyu/NanoJev-Data", "87061eb91e8fc687e9b046454afdcc5551e3eff7"),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(4):
            try:
                response = requests.get(url, timeout=(15, 120))
                response.raise_for_status()
                temporary = path.with_name(path.name + ".part")
                temporary.write_bytes(response.content)
                temporary.replace(path)
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    contents = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "url": url, "bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest()}


def collect_hf(task):
    repo, revision = REVISIONS[task]
    base = ROOT / task / "raw"
    url = f"https://huggingface.co/api/datasets/{repo}/revision/{revision}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    metadata = response.json()
    assert metadata["sha"] == revision
    write_json(base / "repository.json", metadata)
    names = [entry["rfilename"] for entry in metadata["siblings"]]
    if task == "phishing":
        wanted = [name for name in names if name in {
            "README.md", "SOURCE_LICENSES.md", "V5.2_RELEASE_MANIFEST.md", "core_emails.csv",
            "prompt_strategies.json", "real_phishing_validation.csv",
        }]
    else:
        wanted = [name for name in names if name in {"README.md", "manifest.json", "games_v4/README.md", "games_v4/manifest.json"}
                  or name.startswith(("games_v4/data/", "benchmark/", "stage2/"))]
    def fetch(name):
        item = download(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}", base / name)
        print(task, name, item["bytes"], flush=True)
        return item
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, wanted))
    write_json(ROOT / task / "source_manifest.json", {
        "repository": repo, "revision": revision, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "files": files,
    })


def gh_json(endpoint, path):
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(5):
        process = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True)
        if process.returncode == 0:
            result = json.loads(process.stdout)
            write_json(path, result)
            return result
        if attempt == 4:
            raise RuntimeError(process.stderr[:500])
        time.sleep(8 * (attempt + 1))


def collect_github():
    base = ROOT / "github_issues" / "raw"
    repo = "kubernetes/kubernetes"
    write_json(base / "repository.json", gh_json("repos/" + repo, base / "repository.json"))
    all_items, queries = [], []
    # Disjoint complete calendar months avoid GitHub Search's 1,000-result cap.
    for year, month in ((year, month) for year in (2023, 2024, 2025) for month in range(1, 13)):
        last = calendar.monthrange(year, month)[1]
        query = f"repo:{repo} is:issue created:{year}-{month:02d}-01..{year}-{month:02d}-{last}"
        endpoint = "search/issues?q=" + quote(query, safe="") + "&sort=created&order=asc&per_page=100"
        first = gh_json(endpoint + "&page=1", base / f"{year}-{month:02d}-page-01.json")
        total = first["total_count"]
        assert total <= 1000, (query, total, "partition this interval before continuing")
        assert not first.get("incomplete_results"), query
        monthly = list(first["items"])
        for page in range(2, (total + 99) // 100 + 1):
            response = gh_json(endpoint + f"&page={page}", base / f"{year}-{month:02d}-page-{page:02d}.json")
            assert not response.get("incomplete_results"), (query, page)
            monthly.extend(response["items"])
        assert len(monthly) == total, (query, total, len(monthly))
        all_items.extend(monthly)
        queries.append({"query": query, "count": total})
        print("github", year, month, total, "cumulative", len(all_items), flush=True)
        time.sleep(2)
    # Repeated acquisition IDs indicate an API pagination problem, not textual duplicates.
    assert len({row["id"] for row in all_items}) == len(all_items)
    assert all("pull_request" not in row for row in all_items)
    path = base / "issues.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_items))
    write_json(ROOT / "github_issues" / "source_manifest.json", {
        "repository": repo, "source_url": "https://github.com/" + repo,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "queries": queries,
        "issues": len(all_items), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selection": "All issues created during 2023-2025, all open/closed states; no class balancing or content deduplication.",
        "labels": "Repository labels at collection time; labels may have changed since issue creation.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["phishing", "maze", "github"])
    args = parser.parse_args()
    collect_github() if args.task == "github" else collect_hf(args.task)
