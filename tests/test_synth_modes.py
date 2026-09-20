"""Offline acceptance tests for the data creation/annotation workflow.

Only the LLM boundary is replaced. Files, selection, splitting, quota
accounting, and record preservation use the real pipeline.
"""

from collections import Counter
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from luce.config import Endpoint, LuceConfig
from luce.synth import run_synth


class OfflineLLM:
    def __init__(self, answers=None, dynamic=False):
        self.answers = answers or {"route": "billing", "risk": 1, "angry": False}
        self.dynamic = dynamic
        self.persona_prompts = []
        self.writer_prompts = []
        self.teacher_prompts = []
        self.serial = 0

    def chat(self, endpoint, messages, **kwargs):
        self.persona_prompts.append(messages)
        return json.dumps(["an ordinary user"])

    def chat_many(self, endpoint, prompts, **kwargs):
        responses = []
        for messages in prompts:
            if endpoint.model == "offline-writer":
                self.writer_prompts.append(messages)
                text = "\n".join(m["content"] for m in messages)
                match = re.search(r"Write (\d+) realistic", text)
                count = int(match.group(1)) if match else 5
                states = []
                for _ in range(count):
                    self.serial += 1
                    state = f"independent input number {self.serial}"
                    if self.dynamic:
                        state = {"state": state, "options": {"route": {
                            "billing": f"candidate one {self.serial}",
                            "shipping": f"candidate two {self.serial}",
                        }}}
                    states.append(state)
                responses.append(json.dumps(states))
            else:
                self.teacher_prompts.append(messages)
                answer = self.answers(messages) if callable(self.answers) else self.answers
                responses.append(json.dumps(answer))
        return responses


class RepeatingWriter(OfflineLLM):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def chat_many(self, endpoint, prompts, **kwargs):
        if endpoint.model != "offline-writer":
            return super().chat_many(endpoint, prompts, **kwargs)
        self.writer_prompts.extend(prompts)
        replies = []
        for messages in prompts:
            text = "\n".join(message["content"] for message in messages)
            count = int(re.search(r"Write (\d+) realistic", text).group(1))
            replies.append(json.dumps([self.state] * count))
        return replies


class SynthModesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.teacher = Endpoint("http://localhost:1/v1", "offline-teacher")
        self.writer = Endpoint("http://localhost:1/v1", "offline-writer")

    def config(self, **synth):
        return LuceConfig.from_dict({
            "task": {"name": "tickets", "description": "Route tickets and assess risk and anger."},
            "questions": {
                "route": {"type": "choice", "prompt": "Which queue?", "options": {
                    "billing": "Payments and refunds", "shipping": "Delivery"}},
                "risk": {"type": "score", "prompt": "How risky?", "levels": ["low", "high"]},
                "angry": {"type": "noul", "prompt": "Is the customer angry?"},
            },
            "synth": {"n": 4, "personas": 1, "val_fraction": 0, "distractors": "none", **synth},
        })

    def run_pipeline(self, cfg=None, llm=None, out=None, **kwargs):
        llm = llm or OfflineLLM()
        out = out or self.root / "output"
        with ExitStack() as stack:
            stack.enter_context(patch("luce.synth.chat", side_effect=llm.chat))
            stack.enter_context(patch("luce.synth.chat_many", side_effect=llm.chat_many))
            stack.enter_context(redirect_stdout(io.StringIO()))
            run_synth(cfg or self.config(), self.teacher, self.writer, None, str(out), **kwargs)
        return llm, self.rows(out, "train") + self.rows(out, "val")

    def rows(self, out, split):
        path = Path(out) / f"{split}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

    def write_input(self, records):
        path = self.root / "input.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return str(path)

    def record(self, state, name="route", **extra):
        q = self.config().question(name)
        return {"state": state, **q.to_record_fields(), "meta": {"question_name": name}, **extra}

    def test_new_selects_only_requested_questions(self):
        llm, records = self.run_pipeline(questions=["risk"])
        self.assertEqual(len(records), 4)
        self.assertEqual({r["meta"]["question_name"] for r in records}, {"risk"})
        prompts = json.dumps(llm.teacher_prompts)
        self.assertNotIn("Which queue?", prompts)
        self.assertNotIn("Is the customer angry?", prompts)

    def test_type_filter_keeps_only_that_primitive(self):
        _, records = self.run_pipeline(types=["noul"])
        self.assertEqual(len(records), 4)
        self.assertEqual({r["type"] for r in records}, {"noul"})

    def test_default_writer_does_not_preassign_the_correct_answer(self):
        llm, records = self.run_pipeline()
        self.assertEqual(len(records), 12)
        prompts = json.dumps(llm.writer_prompts).lower()
        self.assertNotIn("every input must clearly belong to this answer", prompts)
        self.assertNotIn("truly being about the intended answer", prompts)
        self.assertTrue(all(r.get("meta", {}).get("intent") is None for r in records))

    def test_repeated_generated_states_and_seed_matches_all_receive_labels(self):
        repeated_state = "Please refund my latest order."
        for use_seed in (False, True):
            with self.subTest(matches_seed=use_seed):
                cfg = self.config(n=7)
                if use_seed:
                    cfg.examples = self.write_input([self.record(repeated_state, label="billing")])
                llm, rows = self.run_pipeline(cfg, RepeatingWriter(repeated_state),
                                              out=self.root / f"repeated-{use_seed}",
                                              questions=["route"], max_rounds=1)
                self.assertEqual(len(llm.teacher_prompts), 7)
                self.assertTrue(all(repeated_state in json.dumps(prompt) for prompt in llm.teacher_prompts))
                self.assertEqual(len(rows), 7)
                self.assertEqual([row["state"] for row in rows], [repeated_state] * 7)
                self.assertTrue(all(row["label"] == "billing" for row in rows))

    def test_per_question_counts_are_output_records_including_holdout(self):
        cfg = self.config(val_fraction=0.25)
        _, records = self.run_pipeline(cfg, counts={"route": 7, "risk": 3, "angry": 1})
        self.assertEqual(Counter(r["meta"]["question_name"] for r in records),
                         {"route": 7, "risk": 3, "angry": 1})
        train_states = {json.dumps(r["state"], sort_keys=True) for r in self.rows(self.root / "output", "train")}
        val_states = {json.dumps(r["state"], sort_keys=True) for r in self.rows(self.root / "output", "val")}
        self.assertTrue(train_states.isdisjoint(val_states))

    def test_weighted_grid_allocates_requested_state_proportions(self):
        cfg = self.config(n=20, grid={"language": {"ko": 3, "en": 1}})
        _, records = self.run_pipeline(cfg, questions=["route"])
        self.assertEqual(Counter(r["meta"]["grid"]["language"] for r in records), {"ko": 15, "en": 5})

    def test_label_uses_real_states_and_keeps_existing_selected_labels(self):
        labeled = self.record({"hp": 20, "enemies": [1, 2]}, label="shipping", custom_id="kept")
        unlabeled = self.record({"hp": 80, "enemies": []}, custom_id="new")
        path = self.write_input([labeled, unlabeled])
        llm, records = self.run_pipeline(mode="label", input_path=path, questions=["route"], counts={"route": 1})
        self.assertFalse(llm.writer_prompts)
        self.assertFalse(llm.persona_prompts)
        by_id = {r["custom_id"]: r for r in records}
        self.assertEqual(by_id["kept"], labeled)
        self.assertEqual(by_id["new"]["state"], unlabeled["state"])
        self.assertEqual(by_id["new"]["label"], "billing")
        self.assertEqual(len(records), 2)

    def test_relabel_hides_old_targets_and_preserves_unselected_records(self):
        selected = self.record("real event one", label="shipping", label_probs={"billing": .1, "shipping": .9},
                               custom_id="selected")
        other = self.record("real event one", "angry", label=True, custom_id="other")
        path = self.write_input([selected, other])
        llm, records = self.run_pipeline(mode="relabel", input_path=path, questions=["route"])
        self.assertFalse(llm.writer_prompts)
        self.assertFalse(llm.persona_prompts)
        self.assertNotIn("label_probs", json.dumps(llm.teacher_prompts))
        self.assertNotIn('\\"label\\"', json.dumps(llm.teacher_prompts))
        by_id = {r["custom_id"]: r for r in records}
        self.assertEqual(by_id["other"], other)
        self.assertEqual(by_id["selected"]["label"], "billing")
        self.assertNotIn("label_probs", by_id["selected"])
        self.assertEqual(by_id["selected"]["state"], selected["state"])

    def test_append_preserves_both_existing_splits(self):
        out = self.root / "existing"
        out.mkdir()
        old_train = self.record("old train state", label="shipping", custom_id="train")
        old_val = self.record("old val state", label="shipping", custom_id="val")
        for split, row in (("train", old_train), ("val", old_val)):
            (out / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        _, records = self.run_pipeline(self.config(n=3), out=out, mode="append", questions=["route"])
        self.assertEqual(len(records), 5)
        self.assertIn(old_train, self.rows(out, "train"))
        self.assertIn(old_val, self.rows(out, "val"))

    def test_append_generated_repetitions_keep_old_row_and_all_new_rows(self):
        out = self.root / "repeated-append"
        out.mkdir()
        state = "Please refund my latest order."
        original = self.record(state, label="shipping")
        (out / "train.jsonl").write_text(json.dumps(original) + "\n")

        llm, rows = self.run_pipeline(self.config(n=3), RepeatingWriter(state), out=out,
                                      mode="append", questions=["route"], max_rounds=1)

        self.assertEqual(len(llm.teacher_prompts), 3)
        self.assertEqual(len(rows), 4)
        self.assertIn(original, rows)
        self.assertEqual([row["state"] for row in rows], [state] * 4)
        self.assertEqual(Counter(row["label"] for row in rows), {"shipping": 1, "billing": 3})

    def test_append_input_labels_real_events_without_generation(self):
        out = self.root / "existing"
        out.mkdir()
        old = self.record("old event", label="shipping")
        (out / "val.jsonl").write_text(json.dumps(old) + "\n")
        incoming = self.record({"event_id": 42, "delivered": False})
        path = self.write_input([incoming])
        llm, records = self.run_pipeline(out=out, mode="append", input_path=path, questions=["route"])
        self.assertFalse(llm.writer_prompts)
        self.assertFalse(llm.persona_prompts)
        self.assertEqual(len(records), 2)
        self.assertIn(old, self.rows(out, "val"))
        added = next(r for r in records if r["state"] == incoming["state"])
        self.assertEqual(added["label"], "billing")

    def test_unfillable_generation_fails_without_overwriting_existing_files(self):
        out = self.root / "protected-output"
        out.mkdir()
        old = self.record("preserved event", label="shipping")
        before = json.dumps(old) + "\n"
        (out / "train.jsonl").write_text(before)

        class EmptyWriter(OfflineLLM):
            def chat_many(self, endpoint, prompts, **kwargs):
                if endpoint.model == "offline-writer":
                    return ["[]"] * len(prompts)
                return super().chat_many(endpoint, prompts, **kwargs)

        with self.assertRaises((ValueError, RuntimeError)):
            self.run_pipeline(llm=EmptyWriter(), out=out, questions=["route"], max_rounds=1)
        self.assertEqual((out / "train.jsonl").read_text(), before)

    def test_dynamic_input_options_reach_teacher_and_output(self):
        cfg = self.config()
        cfg.question("route").dynamic_options = True
        cfg.question("route").options_prompt = "Use candidates supplied by each event."
        rows = [self.record("state one", options={"red": "red robot", "blue": "blue robot"}),
                self.record("state two", options={"left": "left route", "right": "right route"})]
        path = self.write_input(rows)

        def answers(messages):
            text = json.dumps(messages)
            return {"route": "red" if "red robot" in text else "right"}

        llm, records = self.run_pipeline(cfg, OfflineLLM(answers), mode="label", input_path=path, questions=["route"])
        by_state = {r["state"]: r for r in records}
        for row in rows:
            self.assertEqual(by_state[row["state"]]["options"], row["options"])
        self.assertEqual(by_state["state one"]["label"], "red")
        self.assertEqual(by_state["state two"]["label"], "right")
        self.assertIn("red robot", json.dumps(llm.teacher_prompts))
        self.assertIn("right route", json.dumps(llm.teacher_prompts))

    def test_dynamic_new_options_are_per_state(self):
        cfg = self.config(n=2)
        cfg.question("route").dynamic_options = True
        cfg.question("route").options_prompt = "Write two candidates for each state."
        llm, records = self.run_pipeline(cfg, OfflineLLM(dynamic=True), questions=["route"])
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0]["options"], records[1]["options"])
        self.assertTrue(all("candidate one" in r["options"]["billing"] for r in records))
        self.assertIn("candidate one", json.dumps(llm.teacher_prompts))

    def test_answer_ratio_changes_training_mix_but_leaves_holdout_natural(self):
        class ScenarioLLM(OfflineLLM):
            def chat_many(self, endpoint, prompts, **kwargs):
                if endpoint.model != "offline-writer":
                    self.teacher_prompts.extend(prompts)
                    return [json.dumps({"route": "shipping" if "Where is my shipment" in json.dumps(p) else "billing"})
                            for p in prompts]
                result = []
                for messages in prompts:
                    self.writer_prompts.append(messages)
                    text = "\n".join(m["content"] for m in messages)
                    count = int(re.search(r"Write (\d+) realistic", text).group(1))
                    requested = re.search(r"correct answer should be (\w+)", text)
                    shipping = requested and requested.group(1) == "shipping"
                    states = []
                    for _ in range(count):
                        self.serial += 1
                        states.append(("Where is my shipment" if shipping else "Please refund the duplicate payment")
                                      + f" for order {self.serial}")
                    result.append(json.dumps(states))
                return result

        cfg = self.config(n=10, val_fraction=.2)
        self.run_pipeline(cfg, ScenarioLLM(), questions=["route"],
                          answer_ratios={"route": {"billing": 1, "shipping": 1}}, max_rounds=1)
        train = self.rows(self.root / "output", "train")
        val = self.rows(self.root / "output", "val")
        self.assertEqual(Counter(r["label"] for r in train), {"billing": 4, "shipping": 4})
        self.assertEqual(Counter(r["label"] for r in val), {"billing": 2})
        self.assertTrue(all(r["meta"].get("intent") is None for r in val))

    def test_imported_answer_ratio_shortage_fails_without_inventing_labels(self):
        path = self.write_input([self.record(f"actual event {i}") for i in range(4)])
        out = self.root / "shortage"
        llm = OfflineLLM({"route": "billing"})
        with self.assertRaisesRegex(ValueError, "not enough Teacher labels"):
            self.run_pipeline(llm=llm, out=out, mode="label", input_path=path, questions=["route"],
                              counts={"route": 4}, answer_ratios={"route": {"billing": 1, "shipping": 1}})
        self.assertTrue(llm.teacher_prompts)
        self.assertFalse(llm.writer_prompts)
        self.assertFalse(out.exists())

    def test_dry_run_all_modes_never_calls_models_or_writes(self):
        path = self.write_input([self.record("existing input")])
        for mode in ("new", "label", "relabel", "append"):
            with self.subTest(mode=mode):
                out = self.root / f"dry-{mode}"
                kwargs = {"mode": mode, "dry_run": True, "questions": ["route"]}
                if mode in ("label", "relabel"):
                    kwargs["input_path"] = path
                with patch("luce.synth.chat", side_effect=AssertionError("API called")), \
                     patch("luce.synth.chat_many", side_effect=AssertionError("API called")), \
                     redirect_stdout(io.StringIO()):
                    run_synth(self.config(), self.teacher, self.writer, None, str(out), **kwargs)
                self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
