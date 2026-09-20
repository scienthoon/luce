"""Offline configuration and CLI contract tests for synthesis workflows."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from luce.cli import build_parser, main
from luce.config import LuceConfig


def config_data():
    return {
        "questions": {
            "queue": {"type": "choice", "prompt": "Which queue?", "options": {"a": "A", "b": "B"}},
            "risk": {"type": "score", "prompt": "Risk?", "levels": ["low", "high"]},
            "angry": {"type": "noul", "prompt": "Angry?"},
            "link": {"type": "choice", "prompt": "Which link?", "dynamic_options": True,
                     "options_prompt": "Generate candidates for this state."},
        },
        "synth": {
            "teacher": "http://localhost:1/v1|offline-teacher", "mode": "label", "input": "events.jsonl",
            "questions": ["queue", "angry"], "types": ["choice", "noul"],
            "counts": {"queue": 0, "angry": 25}, "max_rounds": 4,
            "grid": {"language": {"ko": 70, "en": 30}, "channel": ["chat", "email"]},
            "answer_ratios": {"queue": {"a": 1, "b": 1}, "risk": {0: 1, 1: 2},
                              "angry": {True: 1, False: 9}},
        },
    }


class SynthConfigTests(unittest.TestCase):
    def test_yaml_roundtrip_preserves_generation_plan(self):
        cfg = LuceConfig.from_dict(config_data())
        cfg.validate()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "luce.yaml")
            cfg.save(path)
            restored = LuceConfig.load(path)
        self.assertEqual(restored.to_dict(), cfg.to_dict())
        self.assertEqual(restored.synth.grid["language"], {"ko": 70, "en": 30})
        self.assertEqual(restored.synth.grid["channel"], ["chat", "email"])
        self.assertEqual(restored.synth.answer_ratios["angry"], {"true": 1, "false": 9})
        self.assertEqual(restored.synth.answer_ratios["risk"], {"0": 1, "1": 2})
        self.assertTrue(restored.question("link").dynamic_options)
        self.assertEqual(restored.question("link").options, {})
        self.assertEqual(restored.question("link").options_prompt, "Generate candidates for this state.")

    def test_no_implicit_selection_or_balancing(self):
        data = config_data()
        data.pop("synth")
        cfg = LuceConfig.from_dict(data)
        cfg.validate()
        self.assertEqual(cfg.synth.mode, "new")
        self.assertEqual(cfg.synth.questions, [])
        self.assertEqual(cfg.synth.types, [])
        self.assertEqual(cfg.synth.counts, {})
        self.assertEqual(cfg.synth.answer_ratios, {})
        self.assertIsNone(cfg.synth.teacher)

    def test_invalid_generation_plans_are_rejected(self):
        cases = [
            ("questions", ["missing"]), ("questions", "queue"), ("types", ["text"]),
            ("counts", {"queue": -1}), ("counts", {"queue": .5}), ("counts", {"queue": True}),
            ("counts", {"missing": 2}), ("max_rounds", 0), ("max_rounds", 1.5),
            ("grid", {"language": {"ko": -1, "en": 2}}),
            ("grid", {"language": {"ko": float("nan")}}),
            ("grid", {"language": {"ko": float("inf")}}),
            ("grid", {"language": {"ko": 0}}),
            ("answer_ratios", {"queue": {"a": 1}}),
            ("answer_ratios", {"risk": {"low": 1, "high": 1}}),
            ("answer_ratios", {"angry": {"true": 0, "false": 0}}),
            ("answer_ratios", {"link": {"a": 1, "b": 1}}),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                data = config_data()
                data["synth"][key] = value
                with self.assertRaises(ValueError):
                    LuceConfig.from_dict(data).validate()

    def test_dynamic_candidates_are_choice_only(self):
        data = config_data()
        data["questions"]["angry"]["dynamic_options"] = True
        with self.assertRaisesRegex(ValueError, "only supported for choice"):
            LuceConfig.from_dict(data).validate()


class SynthCliTests(unittest.TestCase):
    def invoke(self, flags):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "luce.yaml")
            LuceConfig.from_dict(config_data()).save(path)
            original_argv = list(sys.argv)
            with patch("luce.synth.run_synth") as run:
                main(["--config", path, "synth", *flags])
            self.assertEqual(sys.argv, original_argv)
        return run.call_args

    def test_cli_forwards_explicit_overrides(self):
        call = self.invoke([
            "--questions", "queue,angry", "--types", "choice,noul", "--counts", "queue=0,angry=20",
            "--mode", "relabel", "--input", "actual.jsonl", "--n", "0", "--max-rounds", "6",
            "--grid-ratios", '{"language":{"ko":3,"en":1}}',
            "--answer-ratios", '{"angry":{"true":2,"false":8}}', "--dry-run", "--out", "data/actual",
        ])
        self.assertEqual(call.kwargs["questions"], ["queue", "angry"])
        self.assertEqual(call.kwargs["types"], ["choice", "noul"])
        self.assertEqual(call.kwargs["counts"], {"queue": 0, "angry": 20})
        self.assertEqual(call.kwargs["mode"], "relabel")
        self.assertEqual(call.kwargs["input_path"], "actual.jsonl")
        self.assertEqual(call.kwargs["grid_ratios"], {"language": {"ko": 3, "en": 1}})
        self.assertEqual(call.kwargs["answer_ratios"], {"angry": {"true": 2, "false": 8}})
        self.assertEqual(call.kwargs["max_rounds"], 6)
        self.assertEqual(call.kwargs["n"], 0)
        self.assertTrue(call.kwargs["dry_run"])
        self.assertEqual(call.kwargs["out"], "data/actual")
        self.assertEqual(call.kwargs["teacher"].model, "offline-teacher")

    def test_omitted_flags_preserve_yaml_settings_for_engine(self):
        call = self.invoke([])
        for field in ("questions", "types", "counts", "mode", "input_path", "grid_ratios", "answer_ratios", "max_rounds", "n"):
            self.assertIsNone(call.kwargs[field], field)
        self.assertEqual(call.args[0].synth.counts, {"queue": 0, "angry": 25})
        self.assertEqual(call.args[0].synth.mode, "label")

    def test_cli_rejects_malformed_overrides(self):
        cases = [
            ["--questions", "queue,"], ["--types", "text"], ["--counts", "queue=1,queue=2"],
            ["--counts", "queue=-1"], ["--counts", "queue=1.5"], ["--counts", "queue"],
            ["--max-rounds", "0"], ["--grid-ratios", "[]"],
            ["--grid-ratios", '{"language":{"ko":0}}'],
            ["--answer-ratios", '{"angry":{"true":NaN,"false":1}}'],
        ]
        for flags in cases:
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    build_parser().parse_args(["synth", *flags])
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
