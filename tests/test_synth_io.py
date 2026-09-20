"""Offline tests for importing decision inputs and preserving dataset outputs."""

import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from luce.config import LuceConfig, QuestionSpec, TaskSpec
from luce.synth_io import read_inputs, read_output, state_key, write_output


class SynthInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "inputs.jsonl")
        self.flag = QuestionSpec("flag", "noul", "Is this urgent?")
        self.category = QuestionSpec("category", "choice", "Which category?", options={"a": "Account", "b": "Billing"})
        self.score = QuestionSpec("score", "score", "How urgent?", levels=["Low", "High"])
        self.cfg = LuceConfig(task=TaskSpec(name="support"), questions=[self.flag, self.category, self.score])

    def write(self, rows):
        with open(self.path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return self.path

    def dynamic(self, name="action", prompt="Which action?"):
        q = QuestionSpec(name, "choice", prompt, dynamic_options=True)
        self.cfg.questions.append(q)
        return q

    def record(self, q, state, **extra):
        return {"state": state, **q.to_record_fields(), "meta": {"question_name": q.name}, **extra}

    def test_raw_and_wrapped_states_preserve_nested_json_types_and_metadata(self):
        state = {"hp": 20, "ready": False, "items": [{"count": 2}], "missing": None}
        rows = ["an input", state, {"state": {"hp": 1}, "meta": {"source": "simulator", "episode": 4}}]
        groups = read_inputs(self.write(rows), self.cfg, [self.flag])
        self.assertEqual([g["state"] for g in groups], ["an input", state, {"hp": 1}])
        self.assertEqual(groups[-1]["meta"], {"source": "simulator", "episode": 4})
        self.assertTrue(all(g["records"] == [] for g in groups))
        self.assertTrue(all(list(g["questions"]) == ["flag"] for g in groups))
        self.assertIsNot(groups[0]["questions"]["flag"], self.flag)

    def test_full_rows_group_by_state_and_preserve_unselected_records(self):
        state = {"text": "refund", "amount": 12}
        flag_row = self.record(self.flag, state, label=True, source="human")
        category_row = self.record(self.category, {"amount": 12, "text": "refund"}, label="b")
        category_row["meta"]["reviewer"] = "ops"
        groups = read_inputs(self.write([flag_row, category_row]), self.cfg, [self.flag])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["records"], [flag_row, category_row])
        self.assertEqual(list(groups[0]["questions"]), ["flag"])

    def test_repeated_raw_inputs_each_remain_an_observation(self):
        for row in ("same", {"hp": 20}, {"state": "same"}, {"state": "same", "record_id": "repeated"}):
            with self.subTest(row=row):
                groups = read_inputs(self.write([row, row]), self.cfg, [self.flag])
                self.assertEqual(len(groups), 2)
                self.assertEqual([list(g["questions"]) for g in groups], [["flag"], ["flag"]])
                self.assertEqual(groups[0]["state"], groups[1]["state"])

    def test_repeated_decision_occurrences_keep_each_question_row(self):
        first = self.record(self.flag, "same", label=True)
        second = self.record(self.flag, "same", label=False)
        first_category = self.record(self.category, "same", label="a")
        second_category = self.record(self.category, "same", label="b")
        for rows in ([first, first_category, second, second_category], [first, second, first_category, second_category]):
            with self.subTest(rows=rows):
                groups = read_inputs(self.write(rows), self.cfg, [self.flag, self.category])
                self.assertEqual(len(groups), 2)
                self.assertEqual(groups[0]["records"], [first, first_category])
                self.assertEqual(groups[1]["records"], [second, second_category])
                self.assertEqual(sum(len(g["records"]) for g in groups), len(rows))

    def test_identical_decision_rows_and_raw_rows_are_not_combined(self):
        row = self.record(self.flag, "same", label=True, record_id="one")
        groups = read_inputs(self.write([row, row, {"state": "same", "record_id": "one"}]), self.cfg, [self.flag])
        self.assertEqual(len(groups), 3)
        self.assertEqual([g["records"] for g in groups], [[row], [row], []])

    def test_unselected_only_records_are_passthrough_without_dynamic_candidates(self):
        dynamic = self.dynamic()
        row = self.record(self.flag, "unrelated", label=False)
        groups = read_inputs(self.write([row]), self.cfg, [dynamic])
        self.assertEqual(groups[0]["questions"], {})
        self.assertEqual(groups[0]["records"], [row])

    def test_named_records_allow_policy_rewording_but_type_mismatch_fails(self):
        row = self.record(self.flag, "input", label=False)
        row["question"] = "Previous policy wording"
        groups = read_inputs(self.write([row]), self.cfg, [self.flag])
        self.assertEqual(groups[0]["questions"]["flag"].prompt, self.flag.prompt)
        self.assertEqual(groups[0]["records"][0]["question"], "Previous policy wording")
        row["type"] = "score"
        with self.assertRaisesRegex(ValueError, r"inputs.jsonl:1:.*expected 'noul'"):
            read_inputs(self.write([row]), self.cfg, [self.flag])

    def test_top_question_name_and_anonymous_records_resolve(self):
        named = {"state": "one", "question_name": "flag", "type": "noul", "question": "old"}
        anonymous = {"state": "two", **self.score.to_record_fields(), "label": 1}
        groups = read_inputs(self.write([named, anonymous]), self.cfg, [self.flag, self.score])
        self.assertEqual(len(groups), 2)
        self.assertEqual(list(groups[0]["questions"]), ["flag", "score"])
        self.assertEqual(groups[1]["records"], [anonymous])

    def test_question_identity_errors_include_path_and_line(self):
        row = self.record(self.flag, "input")
        row["question_name"] = "category"
        with self.assertRaisesRegex(ValueError, r"inputs.jsonl:2:.*disagrees"):
            read_inputs(self.write(["first", row]), self.cfg, [self.flag])
        row.pop("question_name")
        row["meta"]["question_name"] = "unknown"
        with self.assertRaisesRegex(ValueError, r"inputs.jsonl:1:.*unknown question_name"):
            read_inputs(self.write([row]), self.cfg, [self.flag])
        row.pop("meta")
        self.cfg.questions.append(QuestionSpec("another_flag", "noul", self.flag.prompt))
        with self.assertRaisesRegex(ValueError, r"inputs.jsonl:1:.*ambiguous"):
            read_inputs(self.write([row]), self.cfg, [self.flag])

    def test_dynamic_candidates_from_records_envelopes_and_flat_options(self):
        dynamic = self.dynamic()
        candidates = {"jump": "Jump over wall", "hide": "Hide behind wall"}
        full = self.record(dynamic, "full", options=candidates)
        rows = [full, {"state": "bundled", "options": {"action": candidates}}, {"state": "flat", "options": candidates}]
        groups = read_inputs(self.write(rows), self.cfg, [dynamic, self.flag])
        self.assertEqual(len(groups), 3)
        for group in groups:
            self.assertEqual(group["questions"]["action"].options, candidates)
        self.assertEqual(dynamic.options, {})
        self.assertEqual(groups[0]["records"], [full])

    def test_each_dynamic_question_needs_its_own_imported_candidates(self):
        action = self.dynamic()
        route = self.dynamic("route", "Which route?")
        row = {"state": "input", "options": {"action": {"a": "A", "b": "B"}, "route": {"c": "C", "d": "D"}}}
        group = read_inputs(self.write([row]), self.cfg, [action, route])[0]
        self.assertEqual(group["questions"]["route"].options, {"c": "C", "d": "D"})
        del row["options"]["route"]
        with self.assertRaisesRegex(ValueError, r"inputs.jsonl:1:.*'route'.*dynamic options are required"):
            read_inputs(self.write([row]), self.cfg, [action, route])
        row["options"] = {"a": "A", "b": "B"}
        with self.assertRaisesRegex(ValueError, "flat options require exactly one"):
            read_inputs(self.write([row]), self.cfg, [action, route])

    def test_fixed_candidates_must_match_and_are_not_mutated(self):
        original = copy.deepcopy(self.category.options)
        row = self.record(self.category, "input", label="a")
        group = read_inputs(self.write([row]), self.cfg, [self.category])[0]
        self.assertEqual(group["questions"]["category"].options, original)
        group["questions"]["category"].options["a"] = "mutated copy"
        self.assertEqual(self.category.options, original)
        row["options"]["a"] = "Different meaning"
        with self.assertRaisesRegex(ValueError, "input options differ"):
            read_inputs(self.write([row]), self.cfg, [self.category])

    def test_repeated_dynamic_decisions_keep_each_occurrences_candidates(self):
        dynamic = self.dynamic()
        rows = [self.record(dynamic, "same", options={"a": "A", "b": "B"}), self.record(dynamic, "same", options={"a": "C", "b": "D"})]
        groups = read_inputs(self.write(rows), self.cfg, [dynamic])
        self.assertEqual(len(groups), 2)
        self.assertEqual([g["questions"]["action"].options for g in groups], [r["options"] for r in rows])
        rows[0]["record_id"] = "first"
        rows[1]["record_id"] = "second"
        groups = read_inputs(self.write(rows), self.cfg, [dynamic])
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[1]["records"], [rows[1]])

    def test_explicit_record_ids_keep_different_question_observations_separate(self):
        rows = [self.record(self.flag, "same", record_id="first"), self.record(self.category, "same", record_id="second")]
        groups = read_inputs(self.write(rows), self.cfg, [self.flag, self.category])
        self.assertEqual([g["records"] for g in groups], [[rows[0]], [rows[1]]])

    def test_import_errors_report_actual_line(self):
        for text, message in [("\nnot json\n", "invalid JSON"), ("\n42\n", "input must be"), ('\n{"state":null}\n', "state must be")]:
            with self.subTest(text=text):
                with open(self.path, "w", encoding="utf-8") as handle:
                    handle.write(text)
                with self.assertRaisesRegex(ValueError, "inputs.jsonl:2:.*" + message):
                    read_inputs(self.path, self.cfg, [self.flag])

    def test_wrapper_record_ids_survive_in_group_metadata(self):
        rows = [{"state": "same", "record_id": 1}, {"state": "same", "record_id": 2}]
        groups = read_inputs(self.write(rows), self.cfg, [self.flag])
        self.assertEqual(len(groups), 2)
        self.assertEqual([g["meta"]["record_id"] for g in groups], [1, 2])

    def test_state_key_canonicalizes_object_order_without_stringifying_values(self):
        self.assertEqual(state_key({"a": 1, "b": False}), state_key({"b": False, "a": 1}))
        self.assertNotEqual(state_key({"a": 1}), state_key({"a": "1"}))


class SynthOutputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.out = os.path.join(self.directory.name, "dataset")

    def test_roundtrip_existing_splits_and_report(self):
        self.assertEqual(read_output(self.out), ([], []))
        train, val = [{"state": {"hp": 2}, "label": False}], [{"state": "한국어", "label": 1}]
        write_output(self.out, train, val, {"records": 2})
        self.assertEqual(read_output(self.out), (train, val))
        with open(os.path.join(self.out, "synth_report.json"), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {"records": 2})
        self.assertEqual(sorted(os.listdir(self.out)), ["synth_report.json", "train.jsonl", "val.jsonl"])

    def test_serialization_failure_does_not_replace_any_existing_output(self):
        train, val = [{"state": "old train"}], [{"state": "old val"}]
        write_output(self.out, train, val, {"version": 1})
        with self.assertRaises(TypeError):
            write_output(self.out, [{"state": "new train"}], [], {"cannot_serialize": object()})
        self.assertEqual(read_output(self.out), (train, val))
        with open(os.path.join(self.out, "synth_report.json"), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {"version": 1})
        self.assertEqual(len(os.listdir(self.out)), 3)

    def test_all_outputs_are_staged_before_first_replacement(self):
        replace_file = os.replace
        observed = []

        def inspect_staging(source, destination):
            if not observed:
                temporary = [name for name in os.listdir(self.out) if name.endswith(".tmp")]
                self.assertEqual(len(temporary), 3)
                for name in temporary:
                    with open(os.path.join(self.out, name), encoding="utf-8") as handle:
                        self.assertTrue(handle.read().strip())
            observed.append(destination)
            return replace_file(source, destination)

        with patch("luce.synth_io.os.replace", side_effect=inspect_staging):
            write_output(self.out, [{"state": "train"}], [{"state": "val"}], {"done": True})
        self.assertEqual(len(observed), 3)


if __name__ == "__main__":
    unittest.main()
