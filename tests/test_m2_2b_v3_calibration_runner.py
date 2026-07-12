from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.run_m2_2b_v3_calibration import (
    ARMS,
    atomic_write_results,
    load_calibration_rows,
    load_existing_results,
    render_qwen_calibration_prompt,
)

DATA = Path(
    "datasets/m2_2b_formal/v3_calibration/m2_2b_v3_calibration72_arms.jsonl"
)
S_DATA = Path(
    "datasets/m2_2b_formal/v3_s_calibration/m2_2b_v3_s_calibration36.jsonl"
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "rendered"


class CalibrationV3RunnerTest(unittest.TestCase):
    def test_frozen_data_has_twelve_complete_six_arm_units(self) -> None:
        rows = load_calibration_rows(DATA)

        self.assertEqual(len(rows), 72)
        self.assertEqual(Counter(row["arm"] for row in rows), Counter({a: 12 for a in ARMS}))
        self.assertEqual(len({row["base_id"] for row in rows}), 12)

    def test_prompt_uses_exact_four_message_role_order_and_fixed_assistant(self) -> None:
        row = load_calibration_rows(DATA)[0]
        tokenizer = FakeTokenizer()

        prompt = render_qwen_calibration_prompt(tokenizer, row)

        self.assertEqual(prompt, "rendered")
        self.assertEqual(
            [message["role"] for message in tokenizer.messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(tokenizer.messages[2]["content"], row["initial_assistant"])
        self.assertFalse(tokenizer.kwargs["enable_thinking"])
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])

    def test_social_strength_profile_has_three_complete_global_levels(self) -> None:
        rows = load_calibration_rows(S_DATA, profile="s_strength")

        self.assertEqual(len(rows), 36)
        self.assertEqual(
            Counter(row["social_pressure_strength"] for row in rows),
            Counter(
                {
                    "S1_tentative": 12,
                    "S2_confident_preference": 12,
                    "S3_coercive": 12,
                }
            ),
        )
        self.assertEqual(len({row["base_id"] for row in rows}), 12)

    def test_results_are_ordered_resumable_and_strictly_validated(self) -> None:
        rows = load_calibration_rows(DATA)
        second = {"id": rows[1]["id"], "seed": 0, "response": '{"answer":"A"}'}
        first = {"id": rows[0]["id"], "seed": 0, "response": '{"answer":"B"}'}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            atomic_write_results(path, rows, {second["id"]: second, first["id"]: first})
            lines = [json.loads(line) for line in path.read_text().splitlines()]
            loaded = load_existing_results(path, valid_ids={row["id"] for row in rows})

            self.assertEqual([row["id"] for row in lines], [first["id"], second["id"]])
            self.assertEqual(loaded, {first["id"]: first, second["id"]: second})

            path.write_text(json.dumps({**first, "seed": 1}) + "\n")
            with self.assertRaisesRegex(ValueError, "Malformed result"):
                load_existing_results(path, valid_ids={row["id"] for row in rows})


if __name__ == "__main__":
    unittest.main()
