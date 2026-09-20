"""Torch-free unit tests: luce.yaml parsing, teacher spec, mode:auto, data schema. Run: python -m pytest tests -q (or python -m unittest)."""

import json
import os
import tempfile
import unittest

from luce.config import Endpoint, LuceConfig, MAX_LABEL_OPTIONS, decide_mode
from luce.data import example_from_record, permute_example
import random


class EndpointTests(unittest.TestCase):
    def test_parse_pipe(self):
        e = Endpoint.parse("http://localhost:11434/v1|qwen2.5:7b")
        self.assertEqual((e.url, e.model), ("http://localhost:11434/v1", "qwen2.5:7b"))

    def test_parse_mapping(self):
        e = Endpoint.parse({"url": "https://api.openai.com/v1/", "model": "gpt-5", "votes": 5})
        self.assertEqual(e.url, "https://api.openai.com/v1")
        self.assertEqual(e.votes, 5)

    def test_parse_rejects_garbage(self):
        with self.assertRaises(ValueError):
            Endpoint.parse("gpt-5")


class ConfigRoundTrip(unittest.TestCase):
    def test_yaml_roundtrip_and_validate(self):
        cfg = LuceConfig.from_dict({
            "task": {"name": "t", "description": "d", "state": {"fields": ["a", "b"]}},
            "questions": {"q": {"type": "choice", "prompt": "Which?", "options": {"x": "X", "y": "Y"}},
                          "s": {"type": "score", "prompt": "How?", "levels": ["lo", "hi"]},
                          "n": {"type": "noul", "prompt": "It is."}},
            "synth": {"teacher": "http://localhost:11434/v1|qwen2.5:7b", "grid": {"tone": ["calm", "angry"]}},
        })
        cfg.validate()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "luce.yaml")
            cfg.save(path)
            back = LuceConfig.load(path)
        self.assertEqual([q.name for q in back.questions], ["q", "s", "n"])
        self.assertEqual(back.synth.teacher.model, "qwen2.5:7b")
        self.assertEqual(back.max_options(), 2)

    def test_validate_rejects_bad_question(self):
        cfg = LuceConfig.from_dict({"questions": {"q": {"type": "choice", "prompt": "Which?", "options": {"only": "one"}}}})
        with self.assertRaises(ValueError):
            cfg.validate()


class BackboneOptionTests(unittest.TestCase):
    def test_yaml_backbone_options_roundtrip_and_flags(self):
        import tempfile, os
        from luce.config import LuceConfig, backbone_flags
        Q = {"q": {"type": "choice", "prompt": "Which?", "options": {"a": "A", "b": "B"}}}
        cfg = LuceConfig.from_dict({
            "task": {"name": "t"}, "questions": Q,
            "model": {"backbone": "ByteDance/Ouro-2.6B", "mode": "label", "trust_remote_code": True,
                      "backbone_overrides": {"total_ut_steps": 4}, "label_overflow": "text"},
        })
        cfg.validate()
        self.assertEqual(backbone_flags(cfg.model), ["--trust-remote-code", "--backbone-override", "total_ut_steps=4", "--label-overflow", "text"])
        self.assertEqual(backbone_flags(cfg.model, include_label_overflow=False), ["--trust-remote-code", "--backbone-override", "total_ut_steps=4"])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "luce.yaml"); cfg.save(path)
            again = LuceConfig.load(path)
        self.assertTrue(again.model.trust_remote_code)
        self.assertEqual(again.model.backbone_overrides, {"total_ut_steps": 4})
        self.assertEqual(again.model.label_overflow, "text")
        # defaults: plain backbones add no flags
        plain = LuceConfig.from_dict({"task": {"name": "t"}, "questions": Q})
        self.assertEqual(backbone_flags(plain.model), [])
        self.assertEqual(plain.model.backbone, "auto")
        bad = LuceConfig.from_dict({"task": {"name": "t"}, "questions": Q, "model": {"label_overflow": "nope"}})
        with self.assertRaises(ValueError):
            bad.validate()

    def test_backbone_ladder(self):
        import json, os, tempfile
        from luce.config import BACKBONE_LADDER, DEFAULT_BACKBONE, decide_backbone
        def write(n, closed=True):
            d = tempfile.mkdtemp(); path = os.path.join(d, "train.jsonl")
            with open(path, "w") as f:
                for i in range(n):
                    opts = ["a", "b", "c"] if closed else ["a", "b", f"c{i}"]
                    f.write(json.dumps({"type": "choice", "state": {"x": str(i)}, "question": "q", "options": opts, "label": "a"}) + "\n")
            return path
        self.assertEqual(decide_backbone("auto", []).backbone, DEFAULT_BACKBONE)
        # E15: 4B led at every label count on the synthetic task, so auto is 4B regardless of size
        for n, closed in ((300, True), (1500, True), (6000, True), (6000, False)):
            self.assertEqual(decide_backbone("auto", [write(n, closed=closed)]).backbone, DEFAULT_BACKBONE)
        self.assertEqual(BACKBONE_LADDER[0][1], DEFAULT_BACKBONE)
        self.assertEqual(decide_backbone("Qwen/Qwen2.5-3B", [write(6000)]).backbone, "Qwen/Qwen2.5-3B")
        d = decide_backbone("auto", [write(1500)]).describe()
        self.assertIn("->", d); self.assertIn("E15", d)


class AutoModeTests(unittest.TestCase):
    def _write(self, rows):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.close()
        return f.name

    def test_no_data_label_or_isolated(self):
        self.assertEqual(decide_mode("auto", [], config_max_options=4).mode, "label")
        self.assertEqual(decide_mode("auto", [], config_max_options=MAX_LABEL_OPTIONS + 1).mode, "isolated")

    def test_closed_set_is_bi(self):
        path = self._write([{"type": "choice", "question": "Q", "options": {"a": "A", "b": "B"}, "label": "a"}] * 5)
        d = decide_mode("auto", [path])
        self.assertEqual(d.mode, "bi"); self.assertTrue(d.closed_set)

    def test_open_candidates_is_isolated(self):
        rows = [{"type": "choice", "question": "Which option is correct?", "options": {"A": f"opt{i}", "B": f"other{i}"}, "label": "A"} for i in range(5)]
        d = decide_mode("auto", [self._write(rows)])
        self.assertEqual(d.mode, "isolated"); self.assertFalse(d.closed_set)

    def test_explicit_mode_wins(self):
        self.assertEqual(decide_mode("label", [], 100).mode, "label")


class SubsampleOptionsTests(unittest.TestCase):
    def test_keeps_gold_and_renormalizes(self):
        import random
        from luce.data import Example, subsample_options
        ex = Example(query_text="q", option_texts=[f"Option: o{i}" for i in range(10)], option_keys=[f"k{i}" for i in range(10)],
                     target=[0.0] * 7 + [0.6, 0.3, 0.1], qtype="choice")
        sub = subsample_options(ex, 4, random.Random(0))
        self.assertEqual(sub.num_options, 4)
        self.assertIn("k7", sub.option_keys)                       # gold kept
        self.assertAlmostEqual(sum(sub.target), 1.0, places=6)     # renormalised
        self.assertEqual(sub.target[sub.option_keys.index("k7")], max(sub.target))
        self.assertEqual(sub.meta.get("subsampled_from"), 10)
        self.assertIs(subsample_options(ex, 10, random.Random(0)), ex)   # nothing to do
        self.assertIs(subsample_options(ex, 0, random.Random(0)), ex)    # disabled
        score = Example(query_text="q", option_texts=["a", "b", "c", "d", "e"], option_keys=list("abcde"), target=[0, 0, 1, 0, 0], qtype="score")
        self.assertIs(subsample_options(score, 2, random.Random(0)), score)  # score/noul untouched


class DataSchemaTests(unittest.TestCase):
    def test_example_from_record_types(self):
        c = example_from_record({"state": "s", "type": "choice", "question": "q", "options": {"a": "A", "b": "B"}, "label": "b"})
        self.assertEqual(c.target, [0.0, 1.0]); self.assertEqual(c.option_descriptions, ["A", "B"])
        s = example_from_record({"state": "s", "type": "score", "question": "q", "levels": ["lo", "mid", "hi"], "label": 2})
        self.assertEqual(s.hard_label_index, 2)
        n = example_from_record({"state": "s", "type": "noul", "question": "q", "label_probs": 0.8})
        self.assertAlmostEqual(n.target[0], 0.8)

    def test_permute_keeps_label_with_option(self):
        c = example_from_record({"state": "s", "type": "choice", "question": "q", "options": {"a": "A", "b": "B", "c": "C"}, "label": "b"})
        p = permute_example(c, random.Random(3))
        self.assertEqual(p.option_keys[p.hard_label_index], "b")


if __name__ == "__main__":
    unittest.main()
