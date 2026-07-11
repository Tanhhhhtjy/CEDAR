from __future__ import annotations

import unittest

from cedar_m2.eval.belief import summarize_belief


def score_record(
    sample_id: str,
    target: str,
    condition: str,
    step: str,
    *,
    correct: bool,
    margin: float,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "target": target,
        "condition": condition,
        "step": step,
        "correct": correct,
        "margin": margin,
    }


class BeliefSummaryTest(unittest.TestCase):
    def test_metrics_bootstrap_and_pilot_gate(self) -> None:
        records: list[dict[str, object]] = []
        for sample_id in ("bu-1", "bu-2"):
            records.extend(
                [
                    score_record(sample_id, "BU", "uncapped", "time_t", correct=True, margin=2),
                    score_record(sample_id, "BU", "uncapped", "time_t1", correct=True, margin=2),
                    score_record(sample_id, "BU", "p25", "time_t", correct=True, margin=1),
                    score_record(sample_id, "BU", "p25", "time_t1", correct=False, margin=-1),
                ]
            )
        for index, capped_correct in enumerate((True, False), start=1):
            sample_id = f"bm-{index}"
            records.extend(
                [
                    score_record(sample_id, "BM", "uncapped", "time_t", correct=True, margin=2),
                    score_record(sample_id, "BM", "uncapped", "time_t1", correct=True, margin=2),
                    score_record(sample_id, "BM", "p25", "time_t", correct=True, margin=1),
                    score_record(
                        sample_id,
                        "BM",
                        "p25",
                        "time_t1",
                        correct=capped_correct,
                        margin=1 if capped_correct else -1,
                    ),
                ]
            )

        summary = summarize_belief(records, ("uncapped", "p25"), bootstrap_samples=200)

        self.assertEqual(summary["items"], 4)
        self.assertEqual(summary["target_counts"], {"BU": 2, "BM": 2})
        self.assertEqual(summary["conditions"]["uncapped"]["breu"], 1.0)
        self.assertEqual(summary["conditions"]["p25"]["time_t1_accuracy"]["BU"], 0.0)
        self.assertEqual(summary["comparisons_to_uncapped"]["p25"]["delta_bu"], 1.0)
        self.assertEqual(summary["comparisons_to_uncapped"]["p25"]["delta_bm"], 0.5)
        self.assertEqual(
            summary["comparisons_to_uncapped"]["p25"]["delta_bu_one_sided_95ci_lower"],
            1.0,
        )
        self.assertTrue(summary["pilot_gate"]["cost_established"])

    def test_development_gate_rejects_average_without_bu_headroom(self) -> None:
        records = [
            score_record("one", "BU", "uncapped", "time_t", correct=True, margin=1),
            score_record("one", "BU", "uncapped", "time_t1", correct=False, margin=-1),
            score_record("two", "BM", "uncapped", "time_t", correct=True, margin=1),
            score_record("two", "BM", "uncapped", "time_t1", correct=True, margin=1),
        ]
        summary = summarize_belief(records, ("uncapped",), bootstrap_samples=10)

        self.assertEqual(
            summary["conditions"]["uncapped"]["trajectory_accuracy"]["overall"], 0.5
        )
        self.assertFalse(summary["development_gate"]["primary_delta_bu_is_identifiable"])
        self.assertFalse(summary["development_gate"]["prompt_valid"])

    def test_development_gate_passes_with_trajectory_and_bu_headroom(self) -> None:
        records = [
            score_record("one", "BU", "uncapped", "time_t", correct=True, margin=1),
            score_record("one", "BU", "uncapped", "time_t1", correct=True, margin=1),
            score_record("two", "BM", "uncapped", "time_t", correct=True, margin=1),
            score_record("two", "BM", "uncapped", "time_t1", correct=True, margin=1),
        ]
        summary = summarize_belief(records, ("uncapped",), bootstrap_samples=10)

        self.assertTrue(summary["development_gate"]["primary_delta_bu_is_identifiable"])
        self.assertTrue(summary["development_gate"]["prompt_valid"])

    def test_rejects_incomplete_duplicate_or_nonfinite_scores(self) -> None:
        records = [
            score_record("one", "BU", "uncapped", "time_t", correct=True, margin=1),
            score_record("one", "BU", "uncapped", "time_t1", correct=True, margin=1),
        ]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize_belief(records[:-1], ("uncapped",), bootstrap_samples=10)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_belief([*records, records[0]], ("uncapped",), bootstrap_samples=10)

        invalid = [dict(record) for record in records]
        invalid[0]["margin"] = float("nan")
        with self.assertRaisesRegex(ValueError, "numeric margin"):
            summarize_belief(invalid, ("uncapped",), bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
