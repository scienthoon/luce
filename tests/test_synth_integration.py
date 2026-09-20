"""Offline regressions for selected quotas and append split membership."""

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from luce.config import Endpoint, LuceConfig
from luce.synth import run_synth


class SynthIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.teacher = Endpoint("http://localhost:1/v1", "offline-teacher")
        self.prompts = []

    def config(self, fraction=0):
        return LuceConfig.from_dict({
            "questions": {
                "route": {"type": "choice", "prompt": "Which queue?",
                          "options": {"billing": "Payments", "shipping": "Delivery"}},
                "angry": {"type": "noul", "prompt": "Is the customer angry?"},
            },
            "synth": {"val_fraction": fraction, "personas": 0},
        })

    def write_rows(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return str(path)

    def record(self, state, name, **extra):
        question = self.config().question(name)
        return {"state": state, **question.to_record_fields(), "meta": {"question_name": name}, **extra}

    def run_import(self, cfg, source, out, **overrides):
        def annotate(endpoint, prompts, **kwargs):
            self.prompts.extend(prompts)
            return [json.dumps({"route": "billing", "angry": True}) for _ in prompts]

        with ExitStack() as stack:
            stack.enter_context(patch("luce.synth.chat", side_effect=AssertionError("unexpected persona generation")))
            stack.enter_context(patch("luce.synth.chat_many", side_effect=annotate))
            stack.enter_context(redirect_stdout(io.StringIO()))
            run_synth(cfg, self.teacher, None, None, str(out), input_path=source, **overrides)

    def read_split(self, out, split):
        path = out / f"{split}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_append_existing_pair_retains_original_and_adds_new_label(self):
        out = self.root / "existing"
        original = self.record("same event", "route", label="shipping")
        self.write_rows(out / "train.jsonl", [original])
        source = self.write_rows(self.root / "input.jsonl", [self.record("same event", "route")])

        self.run_import(self.config(), source, out, mode="append", questions=["route"], counts={"route": 1})

        rows = self.read_split(out, "train")
        self.assertEqual(len(rows), 2)
        self.assertIn(original, rows)
        self.assertEqual([row["state"] for row in rows], ["same event", "same event"])
        self.assertEqual(sorted(row["label"] for row in rows), ["billing", "shipping"])
        self.assertEqual(self.read_split(out, "val"), [])
        self.assertEqual(len(self.prompts), 1)

    def test_zero_count_dynamic_question_needs_no_candidates_or_labels(self):
        cfg = self.config()
        cfg.question("route").dynamic_options = True
        cfg.question("route").options = {}
        source = self.write_rows(self.root / "input.jsonl", ["actual user message without candidates"])
        out = self.root / "labeled"

        self.run_import(cfg, source, out, mode="label", questions=["route", "angry"],
                        counts={"route": 0, "angry": 1})

        rows = self.read_split(out, "train") + self.read_split(out, "val")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["meta"]["question_name"], "angry")
        self.assertTrue(rows[0]["label"])
        self.assertNotIn("Which queue?", json.dumps(self.prompts))
        self.assertIn("Is the customer angry?", json.dumps(self.prompts))

    def test_append_new_question_keeps_existing_train_state_split(self):
        # 0.5 covers the requested single-row case; 0.75 rounds to one validation
        # target and verifies that available existing split membership wins.
        for fraction in (0.5, 0.75):
            with self.subTest(val_fraction=fraction):
                out = self.root / str(fraction)
                original = self.record("shared event", "route", label="shipping")
                self.write_rows(out / "train.jsonl", [original])
                source = self.write_rows(self.root / f"input-{fraction}.jsonl",
                                         [self.record("shared event", "angry")])

                self.run_import(self.config(fraction), source, out, mode="append", questions=["angry"],
                                counts={"angry": 1})

                train = self.read_split(out, "train")
                self.assertIn(original, train)
                self.assertEqual(len(train), 2)
                added = next(row for row in train if row["meta"]["question_name"] == "angry")
                self.assertEqual(added["state"], original["state"])
                self.assertTrue(added["label"])
                self.assertEqual(self.read_split(out, "val"), [])


if __name__ == "__main__":
    unittest.main()
