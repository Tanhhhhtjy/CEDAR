from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from cedar_m2.checkpoint import RecordStore
from cedar_m2.interventions.capping import LayerCap
from scripts.run_m2_2b_v4_neutral_pair import (
    RUN_NAMES,
    aggregate_cap_stats,
    batch_key,
    direction_sha256,
    fixed_batches,
    load_rows,
    materialize_outputs,
    render_prompt,
)

DATA = Path(
    "datasets/m2_2b_formal/v4_public_poc/"
    "m2_2b_v4_public_poc180_neutral_prompt_arms.jsonl"
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "rendered"


class NeutralPairRunnerTest(unittest.TestCase):
    def test_frozen_rows_form_ninety_ordered_batches(self) -> None:
        rows = load_rows(DATA)
        batches = fixed_batches(rows, 2)

        self.assertEqual(len(rows), 180)
        self.assertEqual(len(batches), 90)
        self.assertTrue(all(len(batch) == 2 for batch in batches))
        self.assertEqual(
            [row["id"] for batch in batches for row in batch],
            [row["id"] for row in rows],
        )

    def test_prompt_keeps_fixed_assistant_and_enables_thinking(self) -> None:
        row = load_rows(DATA)[0]
        tokenizer = FakeTokenizer()

        self.assertEqual(render_prompt(tokenizer, row), "rendered")
        self.assertEqual(
            [message["role"] for message in tokenizer.messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(tokenizer.messages[2]["content"], row["initial_assistant"])
        self.assertTrue(tokenizer.kwargs["enable_thinking"])
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])

    def test_materialization_is_atomic_ordered_and_condition_strict(self) -> None:
        rows = load_rows(DATA)[:4]
        batches = fixed_batches(rows, 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RecordStore(root / "batches", run_fingerprint="test")
            for index, _batch in enumerate(reversed(batches)):
                actual_index = len(batches) - index - 1
                actual_batch = batches[actual_index]
                payload = {
                    "batch_index": actual_index,
                    "condition": "p75",
                    "ids": [row["id"] for row in actual_batch],
                    "results": [
                        {
                            "id": row["id"],
                            "response": '<think>r</think>{"answer":"A"}',
                            "cap": "p75",
                            "thinking_enabled": True,
                            "seed": 0,
                        }
                        for row in actual_batch
                    ],
                    "cap_stats": None,
                }
                store.write(batch_key(actual_index, actual_batch, "p75"), payload)

            results, payloads = materialize_outputs(
                store,
                batches,
                "p75",
                root / "results.jsonl",
                root / "runtime.jsonl",
            )

            self.assertEqual([row["id"] for row in results], [row["id"] for row in rows])
            self.assertEqual([payload["batch_index"] for payload in payloads], [0, 1])
            written = [
                json.loads(line)
                for line in (root / "results.jsonl").read_text().splitlines()
            ]
            self.assertEqual(written, results)

    def test_aggregates_p75_layer_position_stats(self) -> None:
        payloads = [
            {
                "cap_stats": {
                    "definition": "test",
                    "by_layer": {
                        "46": {
                            "total_positions": 10,
                            "triggered_positions": 2,
                            "projection_correction_sum": 5.0,
                        }
                    },
                }
            },
            {
                "cap_stats": {
                    "definition": "test",
                    "by_layer": {
                        "46": {
                            "total_positions": 30,
                            "triggered_positions": 6,
                            "projection_correction_sum": 7.0,
                        }
                    },
                }
            },
        ]

        summary = aggregate_cap_stats(payloads)

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["batches"], 2)
        self.assertEqual(summary["pooled"]["total_positions"], 40)
        self.assertEqual(summary["pooled"]["triggered_positions"], 8)
        self.assertAlmostEqual(summary["pooled"]["trigger_rate"], 0.2)
        self.assertAlmostEqual(summary["pooled"]["mean_projection_correction"], 0.3)

    def test_output_directories_are_frozen_for_only_two_conditions(self) -> None:
        self.assertEqual(set(RUN_NAMES), {"uncapped", "p75"})
        self.assertEqual(
            RUN_NAMES["uncapped"],
            "qwen3_32b_neutral_thinking_uncapped_greedy_seed0_development",
        )
        self.assertEqual(
            RUN_NAMES["p75"],
            "qwen3_32b_neutral_thinking_p75_greedy_seed0_development",
        )

    def test_hashes_official_bfloat16_direction_without_dtype_conversion(self) -> None:
        cap = LayerCap(
            layer=46,
            direction=torch.tensor([1.0, -2.0], dtype=torch.bfloat16),
            threshold=0.0,
        )

        first = direction_sha256(cap)
        second = direction_sha256(cap)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
