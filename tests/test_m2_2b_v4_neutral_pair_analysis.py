from __future__ import annotations

import unittest

from scripts.analyze_m2_2b_v4_neutral_pair import (
    metrics,
    paired_transitions,
    parse_answer,
    response_integrity,
)


class NeutralPairAnalysisTest(unittest.TestCase):
    def test_parses_only_final_post_thinking_json(self) -> None:
        response = (
            '<think>Draft {"answer":"B"}</think>\n'
            '{"answer":"A","confidence":90,"reason":"final"}'
        )

        self.assertEqual(parse_answer(response), "A")
        self.assertTrue(response_integrity(response)["strict_final_json"])
        self.assertIsNone(parse_answer("<think>unfinished"))

    def test_metrics_report_counts_and_valid_denominators(self) -> None:
        rows = [
            {
                "answer": "B",
                "updated": True,
                "correct": True,
                "target": "B",
                "target_accepted": True,
            },
            {
                "answer": "A",
                "updated": False,
                "correct": False,
                "target": "B",
                "target_accepted": False,
            },
            {
                "answer": None,
                "updated": None,
                "correct": None,
                "target": "B",
                "target_accepted": None,
            },
        ]

        result = metrics(rows)

        self.assertEqual(result["n"], 3)
        self.assertEqual(result["valid_n"], 2)
        self.assertEqual(result["updated_n"], 1)
        self.assertEqual(result["update_rate"], 0.5)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["target_acceptance"], 0.5)

    def test_paired_transitions_distinguish_suppressed_and_introduced_updates(self) -> None:
        uncapped = [
            {"id": "x", "arm": "F", "updated": True},
            {"id": "y", "arm": "F", "updated": False},
            {"id": "z", "arm": "F", "updated": True},
        ]
        p75 = [
            {"id": "x", "arm": "F", "updated": False},
            {"id": "y", "arm": "F", "updated": True},
            {"id": "z", "arm": "F", "updated": True},
        ]

        result = paired_transitions(uncapped, p75, "F")

        self.assertEqual(result["uncapped_update_p75_maintain"], 1)
        self.assertEqual(result["uncapped_maintain_p75_update"], 1)
        self.assertEqual(result["both_update"], 1)
        self.assertEqual(result["both_maintain"], 0)


if __name__ == "__main__":
    unittest.main()
