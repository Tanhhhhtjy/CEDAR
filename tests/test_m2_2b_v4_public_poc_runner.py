from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.run_m2_2b_v4_public_poc import (
    ARMS,
    atomic_write_results,
    load_existing_results,
    load_rows,
    render_prompt,
    thinking_smoke_check,
)


DATA = Path(
    "datasets/m2_2b_formal/v4_public_poc/m2_2b_v4_public_poc180_arms.jsonl"
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "rendered"


class V4PublicPocRunnerTest(unittest.TestCase):
    def test_frozen_data_has_thirty_six_complete_five_arm_units(self) -> None:
        rows = load_rows(DATA)

        self.assertEqual(len(rows), 180)
        self.assertEqual(
            Counter(row["arm"] for row in rows),
            Counter({arm: 36 for arm in ARMS}),
        )
        self.assertEqual(len({row["base_id"] for row in rows}), 36)

    def test_prompt_enables_thinking_and_preserves_fixed_assistant(self) -> None:
        row = load_rows(DATA)[0]
        tokenizer = FakeTokenizer()

        rendered = render_prompt(tokenizer, row)

        self.assertEqual(rendered, "rendered")
        self.assertEqual(
            [message["role"] for message in tokenizer.messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(tokenizer.messages[2]["content"], row["initial_assistant"])
        self.assertTrue(tokenizer.kwargs["enable_thinking"])
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])

    def test_smoke_check_requires_nonempty_thinking_and_final_segments(self) -> None:
        good = '<think>\nreasoning\n</think>\n\n{"answer":"A"}'

        self.assertTrue(thinking_smoke_check(good)["passed"])
        self.assertFalse(thinking_smoke_check('{"answer":"A"}')["passed"])
        self.assertFalse(thinking_smoke_check('<think></think>{"answer":"A"}')["passed"])
        self.assertFalse(thinking_smoke_check("<think>reasoning</think>")["passed"])

    def test_results_are_raw_ordered_resumable_and_strict(self) -> None:
        rows = load_rows(DATA)
        first = {
            "id": rows[0]["id"],
            "response": '<think>r</think>{"answer":"A"}',
            "cap": "uncapped",
            "thinking_enabled": True,
            "seed": 0,
        }
        second = {
            "id": rows[1]["id"],
            "response": '<think>s</think>{"answer":"B"}',
            "cap": "uncapped",
            "thinking_enabled": True,
            "seed": 0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            atomic_write_results(
                path,
                rows,
                {second["id"]: second, first["id"]: first},
            )
            written = [json.loads(line) for line in path.read_text().splitlines()]
            loaded = load_existing_results(
                path,
                valid_ids={row["id"] for row in rows},
            )

            self.assertEqual([row["id"] for row in written], [first["id"], second["id"]])
            self.assertEqual(loaded, {first["id"]: first, second["id"]: second})

            path.write_text(
                json.dumps({**first, "thinking_enabled": False}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Malformed result"):
                load_existing_results(
                    path,
                    valid_ids={row["id"] for row in rows},
                )


if __name__ == "__main__":
    unittest.main()
