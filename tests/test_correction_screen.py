from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from torch import nn

from cedar_m2.checkpoint import RecordStore
from cedar_m2.data.correction_screen import (
    EXPECTED_TRANSITIONS,
    VARIANTS,
    build_correction_cases,
    load_correction_screen,
)
from cedar_m2.eval.correction import (
    FINAL_ANSWER_INSTRUCTION,
    render_qwen_correction_prompt,
    summarize_correction_screen,
)
from cedar_m2.eval.mcq import MCQScore
from cedar_m2.runners.correction import run_correction_scoring_stage

DATA_PATH = Path("data/development/m2_objective_correction_screen_v1.json")


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "rendered"


def result_records(
    spontaneous_correct: int,
    valid_correct: int,
    invalid_resistant: int,
    neutral_stable: int,
):
    screen = load_correction_screen(DATA_PATH)
    records = []
    limits = {
        "wrong_neutral_review": spontaneous_correct,
        "valid_correction": valid_correct,
        "invalid_correction": invalid_resistant,
        "neutral_review": neutral_stable,
    }
    seen = Counter()
    for case in build_correction_cases(screen):
        index = seen[case.variant]
        seen[case.variant] += 1
        predicted = case.correct_label if index < limits[case.variant] else case.challenged_label
        records.append(
            {
                "item_id": case.item_id,
                "category": case.category,
                "variant": case.variant,
                "correct_label": case.correct_label,
                "initial_label": case.initial_label,
                "challenged_label": case.challenged_label,
                "predicted_label": predicted,
            }
        )
    return records


class CorrectionDataTest(unittest.TestCase):
    def test_frozen_screen_expands_to_balanced_eighty_cases(self) -> None:
        screen = load_correction_screen(DATA_PATH)
        cases = build_correction_cases(screen)

        self.assertEqual(len(screen.items), 20)
        self.assertEqual(len(cases), 80)
        self.assertEqual(
            Counter(case.variant for case in cases),
            Counter({name: 20 for name in VARIANTS}),
        )
        self.assertEqual(
            Counter((item.challenged_label, item.correct_label) for item in screen.items),
            Counter(EXPECTED_TRANSITIONS),
        )
        wrong_review = [case for case in cases if case.variant == "wrong_neutral_review"]
        valid = [case for case in cases if case.variant == "valid_correction"]
        invalid = [case for case in cases if case.variant == "invalid_correction"]
        neutral = [case for case in cases if case.variant == "neutral_review"]
        self.assertTrue(
            all(case.initial_label != case.correct_label for case in wrong_review + valid)
        )
        self.assertTrue(all(case.initial_label == case.correct_label for case in invalid + neutral))

    def test_loader_rejects_label_imbalance(self) -> None:
        raw = json.loads(DATA_PATH.read_text())
        raw["items"][0]["correct_label"] = "C"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                load_correction_screen(path)


class CorrectionPromptTest(unittest.TestCase):
    def test_multiturn_prompt_disables_thinking_and_appends_instruction(self) -> None:
        messages = (
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "The answer is B."},
            {"role": "user", "content": "feedback"},
        )
        tokenizer = FakeTokenizer()
        prompt = render_qwen_correction_prompt(tokenizer, messages)

        self.assertEqual(prompt, "rendered")
        self.assertEqual(messages[-1]["content"], "feedback")
        self.assertIn(FINAL_ANSWER_INSTRUCTION, tokenizer.messages[-1]["content"])
        self.assertFalse(tokenizer.template_kwargs["enable_thinking"])
        self.assertTrue(tokenizer.template_kwargs["add_generation_prompt"])


class CorrectionSummaryTest(unittest.TestCase):
    def test_narrow_explanation_requires_clean_items_and_controls(self) -> None:
        summary = summarize_correction_screen(result_records(4, 20, 16, 20))

        self.assertEqual(summary["health"]["version_a_exclusion"]["rate"], 0.2)
        self.assertEqual(
            summary["metrics"]["conditional_evidence_driven_correction"]["rate"],
            1.0,
        )
        self.assertEqual(summary["metrics"]["false_correction_resistance"]["rate"], 0.8)
        self.assertEqual(summary["metrics"]["neutral_stability"]["rate"], 1.0)
        self.assertEqual(
            summary["conclusion"]["code"], "narrow_belief_r_explanation_supported"
        )

    def test_accepting_true_and_false_feedback_is_flagged_as_compliance(self) -> None:
        summary = summarize_correction_screen(result_records(4, 20, 10, 20))
        self.assertEqual(
            summary["conclusion"]["code"],
            "revision_capacity_supported_with_compliance_confound",
        )

    def test_low_true_correction_supports_broad_failure(self) -> None:
        summary = summarize_correction_screen(result_records(2, 6, 20, 20))
        self.assertEqual(summary["conclusion"]["code"], "broad_revision_failure_supported")

    def test_too_many_spontaneous_corrections_fail_health_gate(self) -> None:
        summary = summarize_correction_screen(result_records(11, 20, 20, 20))
        self.assertFalse(summary["health"]["passed"])
        self.assertEqual(
            summary["conclusion"]["code"], "screen_invalid_weak_prefill_errors"
        )


class CorrectionRunnerTest(unittest.TestCase):
    def test_scoring_is_checkpointed_and_resumable(self) -> None:
        cases = build_correction_cases(load_correction_screen(DATA_PATH))[:2]

        def fake_scores(_model, _tokenizer, conversations, *, correct_labels, batch_size):
            self.assertEqual(len(conversations), 2)
            self.assertEqual(batch_size, 2)
            return [
                MCQScore(
                    predicted_label=label,
                    label_logits={"A": 2.0, "B": 1.0, "C": 0.0},
                    correct_label=label,
                    correct=True,
                    margin=1.0,
                )
                for label in correct_labels
            ]

        with tempfile.TemporaryDirectory() as tmp:
            store = RecordStore(Path(tmp), run_fingerprint="test-run")
            with patch(
                "cedar_m2.runners.correction.score_correction_conversations",
                side_effect=fake_scores,
            ) as score:
                first = run_correction_scoring_stage(
                    nn.Linear(1, 1), object(), cases, store, batch_size=2
                )
                second = run_correction_scoring_stage(
                    nn.Linear(1, 1), object(), cases, store, batch_size=2
                )
            payloads = list(store.payloads())

        self.assertEqual(first, {"already_complete": 0, "newly_scored": 2})
        self.assertEqual(second, {"already_complete": 2, "newly_scored": 0})
        self.assertEqual(score.call_count, 1)
        self.assertEqual(len(payloads), 2)
        self.assertTrue(all("conversation_sha256" in payload for payload in payloads))


if __name__ == "__main__":
    unittest.main()
