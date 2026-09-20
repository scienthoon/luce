"""Repeated observations survive annotation, relabeling, and append."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from luce.config import Endpoint, LuceConfig
from luce.synth import run_synth


class RepeatedInputTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cfg = LuceConfig.from_dict({
            "questions": {
                "angry": {"type": "noul", "prompt": "Is the customer angry?"},
                "urgent": {"type": "noul", "prompt": "Is this urgent?"},
            },
            "synth": {"personas": 0, "val_fraction": 0},
        })
        self.teacher = Endpoint("http://localhost:1/v1", "offline-teacher")
        self.prompts = []

    def write_rows(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return str(path)

    def record(self, name, label):
        return {"state": "same observed event", **self.cfg.question(name).to_record_fields(),
                "label": label, "meta": {"question_name": name}}

    def run_input(self, rows, mode="label", **kwargs):
        source = self.write_rows(self.root / "input.jsonl", rows)

        def annotate(endpoint, prompts, **options):
            self.prompts.extend(prompts)
            return [json.dumps({"angry": True, "urgent": True}) for _ in prompts]

        with patch("luce.synth.chat_many", side_effect=annotate), redirect_stdout(io.StringIO()):
            run_synth(self.cfg, self.teacher, None, None, str(self.root / "output"),
                      input_path=source, mode=mode, votes=1, **kwargs)
        return self.read_split("train") + self.read_split("val")

    def read_split(self, split):
        return [json.loads(line) for line in (self.root / "output" / f"{split}.jsonl").read_text().splitlines()]

    def test_repeated_raw_rows_each_receive_all_selected_labels(self):
        rows = self.run_input(["same observed event"] * 3)
        self.assertEqual(len(self.prompts), 3)
        self.assertEqual(len(rows), 6)
        for name in ("angry", "urgent"):
            self.assertEqual(sum(row["meta"]["question_name"] == name for row in rows), 3)

    def test_relabel_updates_each_repeated_question_occurrence(self):
        source = [self.record("angry", False)] * 2 + [self.record("urgent", False)] * 2
        rows = self.run_input(source, mode="relabel")
        self.assertEqual(len(self.prompts), 2)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["label"] is True for row in rows))

    def test_append_retains_repetitions_already_in_both_splits(self):
        original = self.record("angry", False)
        for split in ("train", "val"):
            self.write_rows(self.root / "output" / f"{split}.jsonl", [original])
        rows = self.run_input(["same observed event"] * 2, mode="append", questions=["angry"])
        self.assertEqual(len(self.prompts), 2)
        self.assertEqual(len(rows), 4)
        self.assertEqual(self.read_split("val"), [original])
        self.assertEqual(self.read_split("train")[0], original)
        self.assertEqual(sum(row["label"] is True for row in rows), 2)


if __name__ == "__main__":
    unittest.main()
