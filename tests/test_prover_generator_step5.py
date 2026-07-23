from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cedar_m2.checkpoint import atomic_write_json, file_sha256
from scripts import analyze_prover_generator_step5 as analysis
from scripts import audit_prover_generator_step5 as audit
from scripts import run_prover_generator_step5_uncapped as runner


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self) -> None:
        self.enable_thinking: bool | None = None

    def encode(self, text: str, **_kwargs):
        if text == runner.ASSISTANT_PREFIX:
            return [90, 91]
        if text in runner.LABEL_TOKEN_IDS:
            return [runner.LABEL_TOKEN_IDS[text]]
        return [ord(character) for character in text]

    def decode(self, ids, **_kwargs):
        reverse = {value: key for key, value in runner.LABEL_TOKEN_IDS.items()}
        if len(ids) == 1 and ids[0] in reverse:
            return reverse[ids[0]]
        return "".join(chr(value) for value in ids if value > 0)

    def apply_chat_template(self, _messages, **kwargs):
        self.enable_thinking = kwargs["enable_thinking"]
        return "thinking-prompt" if self.enable_thinking else "nonthinking-prompt"


class ProverGeneratorStep5Tests(unittest.TestCase):
    def test_frozen_inputs_and_smoke_selection(self) -> None:
        rows = runner.validate_inputs()
        self.assertEqual(len(rows), 288)
        self.assertEqual({arm: 48 for arm in runner.ARMS}, {
            arm: sum(row["arm"] == arm for row in rows) for arm in runner.ARMS
        })
        smoke = runner.select_rows(rows, "smoke")
        self.assertEqual(len(smoke), 24)
        self.assertEqual(set(runner.SMOKE_BASE_IDS), {row["base_id"] for row in smoke})
        self.assertEqual(set(runner.LABELS), {row["target_answer"] for row in smoke})
        self.assertEqual({"easy", "medium"}, {row["variant"] for row in smoke})
        self.assertEqual(
            {"early", "middle", "late"},
            {row["error_position"] for row in smoke if row["arm"] == "F"},
        )

    def test_thinking_toggle_is_the_rendering_difference(self) -> None:
        tokenizer = FakeTokenizer()
        row = runner.validate_inputs()[0]
        thinking = runner.render_prompt(tokenizer, row, runner.MODES["thinking"])
        self.assertTrue(tokenizer.enable_thinking)
        nonthinking = runner.render_prompt(tokenizer, row, runner.MODES["nonthinking"])
        self.assertFalse(tokenizer.enable_thinking)
        self.assertNotEqual(thinking, nonthinking)

    def test_strict_parser_mode_integrity_and_outcome_classes(self) -> None:
        response = '<think>Reason carefully.</think>\n{"answer":"D"}'
        answer, status, reasoning, sections = runner.strict_answer(
            response, truncated=False
        )
        self.assertEqual((answer, status), ("D", "passed"))
        self.assertEqual(reasoning, "Reason carefully.")
        self.assertTrue(runner.mode_integrity(runner.MODES["thinking"], sections)["passed"])
        non_answer, non_status, _, non_sections = runner.strict_answer(
            '{"answer":"A"}', truncated=False
        )
        self.assertEqual((non_answer, non_status), ("A", "passed"))
        self.assertTrue(
            runner.mode_integrity(runner.MODES["nonthinking"], non_sections)["passed"]
        )
        row = {"initial_answer": "A", "target_answer": "D"}
        self.assertEqual(runner.classify(row, "A"), "initial")
        self.assertEqual(runner.classify(row, "D"), "induced_target")
        self.assertEqual(runner.classify(row, "B"), "other_wrong")
        self.assertEqual(runner.classify(row, None), "invalid_or_abstain")

    def test_answer_prefix_capture_and_four_option_diagnostics(self) -> None:
        tokenizer = FakeTokenizer()
        row = {"initial_answer": "D", "target_answer": "A"}
        capture = runner.AnswerPrefixLogitCapture(tokenizer, row, prompt_tokens=3)
        scores = torch.full((1, 100), -10.0)
        scores[0, 32:36] = torch.tensor([4.0, 3.0, 2.0, 1.0])
        untouched = capture(torch.tensor([[7, 8, 9, 90, 91]]), scores)
        self.assertIs(untouched, scores)
        self.assertEqual(list(capture.by_response_index), [2])
        diagnostics, index, candidates = runner.select_captured_diagnostics(
            [90, 91, 32], capture.by_response_index, "A"
        )
        self.assertEqual(index, 2)
        self.assertEqual(candidates[0]["generated_label"], "A")
        self.assertAlmostEqual(diagnostics["target_minus_initial"], 3.0)
        record = {
            "initial_answer": "D",
            "target_answer": "A",
            "chosen_option": "A",
            "answer_logits": diagnostics,
        }
        self.assertTrue(audit.diagnostics_valid(record))

    def test_run_order_blocks_skips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "thinking smoke"):
                runner.validate_run_order("smoke", "nonthinking", root)
            with self.assertRaisesRegex(ValueError, "combined smoke audit"):
                runner.validate_run_order("formal", "thinking", root)

            run_dir = root / runner.run_name("smoke", "thinking")
            run_dir.mkdir(parents=True)
            results = run_dir / "results.jsonl"
            runner.atomic_write_jsonl(results, [{"id": str(i)} for i in range(24)])
            atomic_write_json(
                run_dir / "runtime.json",
                {
                    "complete_results": 24,
                    "content_retries": 0,
                    "results_sha256": file_sha256(results),
                },
            )
            runner.validate_run_order("smoke", "nonthinking", root)
            atomic_write_json(root / "SMOKE_AUDIT.json", {"status": "passed"})
            runner.validate_run_order("formal", "thinking", root)
            with self.assertRaisesRegex(ValueError, "288 thinking"):
                runner.validate_run_order("formal", "nonthinking", root)

    def test_analysis_keeps_continuous_and_discrete_outputs(self) -> None:
        rows = []
        for index, outcome in enumerate(
            ("initial", "induced_target", "other_wrong", "invalid_or_abstain")
        ):
            logits = torch.full((100,), -10.0)
            logits[32:36] = torch.tensor([1.0, 2.0, 3.0, 4.0])
            diagnostics = runner.choice_diagnostics(
                logits,
                {"initial_answer": "A", "target_answer": "D"},
                FakeTokenizer(),
            )
            rows.append(
                {
                    "base_id": f"base-{index}",
                    "arm": "S",
                    "outcome_class": outcome,
                    "gold_correct": outcome == "initial",
                    "final_answer_parse_status": (
                        "invalid_json" if outcome == "invalid_or_abstain" else "passed"
                    ),
                    "answer_logits": diagnostics,
                    "target_answer": "D",
                    "initial_answer": "A",
                    "response_tokens": 10,
                    "reasoning_token_count": 2,
                    "other_wrong_indicator": outcome == "other_wrong",
                    "invalid_or_abstain_indicator": outcome == "invalid_or_abstain",
                    "synchronized_other_error": outcome
                    in {"other_wrong", "invalid_or_abstain"},
                }
            )
        summary = analysis.arm_summary(rows)
        self.assertEqual(summary["rows"], 4)
        self.assertEqual(summary["answer_margin_count"], 4)
        self.assertEqual(summary["outcome_rates"]["induced_target"], 0.25)
        self.assertIsNotNone(summary["target_minus_initial_mean"])


if __name__ == "__main__":
    unittest.main()
