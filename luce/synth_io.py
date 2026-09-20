"""JSONL input and output for synthetic generation, labeling, and relabeling.

Input states keep their JSON types. Decision records are retained separately so
callers can replace selected labels while carrying unrelated records through.
This module does not call a model or invent candidates for imported inputs.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .config import LuceConfig, QuestionSpec


def state_key(state: Any) -> str:
    """Return the canonical JSON representation used to group equal states."""
    return json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _error(path: str, line: int, message: str) -> ValueError:
    return ValueError(f"{path}:{line}: {message}")


def _invalid_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _read_jsonl(path: str):
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{path}: cannot read JSONL: {exc}") from exc
    with handle:
        for line, text in enumerate(handle, 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text, parse_constant=_invalid_constant)
            except (ValueError, TypeError) as exc:
                raise _error(path, line, f"invalid JSON: {exc}") from exc
            yield line, row


def _record_question(row: Dict[str, Any], questions: Mapping[str, QuestionSpec], path: str, line: int) -> QuestionSpec:
    meta = row.get("meta") or {}
    top_name, meta_name = row.get("question_name"), meta.get("question_name")
    if top_name is not None and meta_name is not None and top_name != meta_name:
        raise _error(path, line, "question_name disagrees with meta.question_name")
    explicit = meta_name if meta_name is not None else top_name
    if explicit is not None:
        if not isinstance(explicit, str) or explicit not in questions:
            raise _error(path, line, f"unknown question_name {explicit!r}")
        q = questions[explicit]
        if "type" in row and row["type"] != q.type:
            raise _error(path, line, f"question {explicit!r} has type {row['type']!r}, expected {q.type!r}")
        # A stable question name deliberately permits a revised policy/prompt.
        return q
    matches = [q for q in questions.values() if row.get("type") == q.type and row.get("question") == q.prompt]
    if len(matches) != 1:
        detail = "ambiguous" if matches else "unknown"
        raise _error(path, line, f"{detail} type/question pair; provide a valid question_name or meta.question_name")
    return matches[0]


def _candidate_options(value: Any, q: QuestionSpec, path: str, line: int) -> Dict[str, str]:
    if not isinstance(value, dict) or len(value) < 2 or any(
        not isinstance(key, str) or not isinstance(description, str) for key, description in value.items()
    ):
        raise _error(path, line, f"question {q.name!r}: options must be a mapping of at least two string keys to string descriptions")
    if q.type != "choice":
        raise _error(path, line, f"question {q.name!r}: options are only valid for choice")
    if not getattr(q, "dynamic_options", False) and value != q.options:
        raise _error(path, line, f"question {q.name!r}: input options differ from the configured fixed options")
    return copy.deepcopy(value)


def _identity(row: Dict[str, Any]) -> Any:
    """An explicit input identity can distinguish identical states/candidates."""
    for name in ("record_id", "state_id", "id"):
        if name in row and row[name] is not None:
            return row[name]
    meta = row.get("meta") or {}
    for name in ("record_id", "state_id", "id"):
        if name in meta and meta[name] is not None:
            return meta[name]
    return None


def read_inputs(path: str, cfg: LuceConfig, selected_questions: Sequence[QuestionSpec]) -> List[Dict[str, Any]]:
    """Read imported states and resolve the selected questions per input group.

    Accepted rows are JSON strings, raw state objects, ``{"state": ...}``
    envelopes, and Luce decision records. Envelopes can supply per-question
    candidates as ``options: {question_name: {key: description}}``. Flat options
    are also accepted when exactly one selected question has dynamic options.

    The result contains ``state``, ``questions`` (name to QuestionSpec),
    ``records`` (the original decision rows), and ``meta`` for each group.
    Unselected-only decision groups have empty ``questions`` for passthrough.
    Every raw input line is a separate observation. Decision rows for different
    questions share a group by state and optional record_id/state_id/id. Repeated
    occurrences of the same question start separate groups: the nth occurrence
    of each question belongs to the nth observation for that state and identity.
    """
    path = os.fspath(path)
    known = {q.name: q for q in cfg.questions}
    selected = {q.name: q for q in selected_questions}
    if len(selected) != len(selected_questions):
        raise ValueError(f"{path}: selected questions contain duplicate names")
    for name in selected:
        if name not in known:
            raise ValueError(f"{path}: selected question {name!r} is not configured")
    dynamic = [q for q in selected.values() if q.type == "choice" and getattr(q, "dynamic_options", False)]
    groups: List[Dict[str, Any]] = []
    record_groups: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    occurrences: Dict[Tuple[str, str, str], int] = {}

    for line, row in _read_jsonl(path):
        if not isinstance(row, (str, dict)):
            raise _error(path, line, "input must be a string or JSON object")
        wrapped = isinstance(row, dict) and "state" in row
        state = row["state"] if wrapped else row
        if not isinstance(state, (str, dict)):
            raise _error(path, line, "state must be a string or JSON object")
        envelope = row if wrapped else {}
        meta = envelope.get("meta", {})
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise _error(path, line, "meta must be a JSON object")
        is_record = wrapped and (
            any(name in row for name in ("type", "question", "question_name")) or "question_name" in meta
        )
        q = _record_question(row, known, path, line) if is_record else None
        identity = _identity(envelope)
        group = None
        if q is not None:
            base = (state_key(state), state_key(identity))
            occurrence_key = (*base, q.name)
            occurrence = occurrences.get(occurrence_key, 0)
            occurrences[occurrence_key] = occurrence + 1
            key = (*base, occurrence)
            group = record_groups.get(key)
        if group is None:
            group = {
                "state": copy.deepcopy(state), "records": [], "meta": copy.deepcopy(meta),
                "_options": {}, "_question_names": set(), "_raw": False, "_line": line,
            }
            groups.append(group)
            if q is not None:
                record_groups[key] = group
        for name, value in meta.items():
            group["meta"].setdefault(name, copy.deepcopy(value))
        if identity is not None and not any(group["meta"].get(name) is not None for name in ("record_id", "state_id", "id")):
            group["meta"]["record_id"] = copy.deepcopy(identity)

        def add_options(question: QuestionSpec, value: Any) -> None:
            options = _candidate_options(value, question, path, line)
            previous = group["_options"].get(question.name)
            if previous is not None and previous != options:
                raise _error(path, line, f"conflicting options for question {question.name!r} on the same input; use distinct record_id values for separate inputs")
            group["_options"][question.name] = options

        if q is not None:
            group["records"].append(copy.deepcopy(row))
            group["_question_names"].add(q.name)
            if q.type == "choice" and "options" in row:
                add_options(q, row["options"])
        else:
            group["_raw"] = True
            if wrapped and "options" in row:
                options = row["options"]
                if not isinstance(options, dict):
                    raise _error(path, line, "options must be a JSON object")
                if options and all(isinstance(value, dict) for value in options.values()):
                    for name, candidates in options.items():
                        if name not in known:
                            raise _error(path, line, f"options supplied for unknown question {name!r}")
                        add_options(known[name], candidates)
                elif len(dynamic) == 1:
                    add_options(dynamic[0], options)
                else:
                    raise _error(path, line, "flat options require exactly one selected dynamic choice question; otherwise use options: {question_name: {...}}")

    result: List[Dict[str, Any]] = []
    for group in groups:
        relevant = group["_raw"] or bool(group["_question_names"].intersection(selected))
        resolved: Dict[str, QuestionSpec] = {}
        if relevant:
            for name, q in selected.items():
                options = group["_options"].get(name)
                if q.type == "choice" and getattr(q, "dynamic_options", False) and options is None:
                    raise _error(path, group["_line"], f"question {name!r}: dynamic options are required for imported inputs; labeling does not generate candidates")
                changes = {"options": copy.deepcopy(options if options is not None else q.options), "levels": list(q.levels)}
                resolved[name] = replace(q, **changes)
        result.append({"state": group["state"], "questions": resolved, "records": group["records"], "meta": group["meta"]})
    return result


def read_output(out: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read existing train/validation records; a missing split is empty."""
    def read_split(name: str) -> List[Dict[str, Any]]:
        path = os.path.join(out, name)
        if not os.path.exists(path):
            return []
        records = []
        for line, row in _read_jsonl(path):
            if not isinstance(row, dict):
                raise _error(path, line, "output record must be a JSON object")
            records.append(row)
        return records

    return read_split("train.jsonl"), read_split("val.jsonl")


def write_output(out: str, train_rows: Sequence[Dict[str, Any]], val_rows: Sequence[Dict[str, Any]], report: Dict[str, Any]) -> None:
    """Serialize all three outputs before replacing any existing destination.

    Replacements are atomic per file. If serialization fails, existing output
    files remain untouched and temporary files are removed.
    """
    os.makedirs(out, exist_ok=True)
    staged: List[Tuple[str, str]] = []
    try:
        for name, value in (("train.jsonl", train_rows), ("val.jsonl", val_rows), ("synth_report.json", report)):
            fd, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=out)
            staged.append((temporary, os.path.join(out, name)))
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                if name.endswith(".jsonl"):
                    for row in value:
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                else:
                    json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            if os.path.exists(temporary):
                os.unlink(temporary)
