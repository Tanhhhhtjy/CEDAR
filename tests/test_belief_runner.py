from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torch import nn

from cedar_m2.checkpoint import RecordStore
from cedar_m2.data.belief_r import BeliefPair, BeliefRow, BeliefSplits
from cedar_m2.eval.mcq import MCQScore
from cedar_m2.runners.belief import (
    conditions_for_split,
    expected_score_count,
    pairs_for_split,
    run_belief_scoring_stage,
)


def make_pair(index: int, target: str) -> BeliefPair:
    common = {
        "modus": "ponens",
        "relation_type": "If-Event-Then-Event",
        "atomic_idx": str(index),
        "dataset_id": f"{index}-strong",
    }
    time_t = BeliefRow(
        question=f"initial {index}",
        ground_truth="A",
        step="time_t",
        agreement_level="",
        **common,
    )
    time_t1 = BeliefRow(
        question=f"updated {index}",
        ground_truth="C" if target == "BU" else "A",
        step="time_t1",
        agreement_level="5",
        **common,
    )
    return BeliefPair(time_t=time_t, time_t1=time_t1)


class BeliefRunnerTest(unittest.TestCase):
    def test_split_conditions_and_score_counts_are_frozen(self) -> None:
        dev, bu, bm, extension = (
            make_pair(1, "BU"),
            make_pair(2, "BU"),
            make_pair(3, "BM"),
            make_pair(4, "BU"),
        )
        splits = BeliefSplits((dev,), (bu,), (bm,), (extension,))

        self.assertEqual(pairs_for_split(splits, "pilot"), (bu, bm))
        self.assertEqual(conditions_for_split("development"), ("uncapped",))
        self.assertEqual(
            conditions_for_split("pilot"), ("uncapped", "p1", "p25", "p50", "p75")
        )
        self.assertEqual(conditions_for_split("extension"), ("uncapped", "p25"))
        self.assertEqual(expected_score_count(200, conditions_for_split("pilot")), 2000)

    def test_scoring_checkpoints_each_item_and_resumes_without_recomputing(self) -> None:
        pairs = [make_pair(1, "BU"), make_pair(2, "BM")]

        def fake_scores(_model, _tokenizer, questions, *, correct_labels, batch_size):
            self.assertEqual(batch_size, 1)
            return [
                MCQScore(
                    predicted_label=label,
                    label_logits={"A": 2.0, "B": 0.0, "C": 1.0},
                    correct_label=label,
                    correct=True,
                    margin=1.0,
                )
                for _question, label in zip(questions, correct_labels, strict=True)
            ]

        with tempfile.TemporaryDirectory() as tmp:
            store = RecordStore(Path(tmp), run_fingerprint="test-run")
            with patch(
                "cedar_m2.runners.belief.score_mcq_prompts", side_effect=fake_scores
            ) as score:
                first = run_belief_scoring_stage(
                    nn.Linear(1, 1),
                    object(),
                    pairs,
                    ("uncapped",),
                    {},
                    store,
                    batch_size=1,
                )
                calls_after_first = score.call_count
                second = run_belief_scoring_stage(
                    nn.Linear(1, 1),
                    object(),
                    pairs,
                    ("uncapped",),
                    {},
                    store,
                    batch_size=1,
                )

            payloads = list(store.payloads())

        self.assertEqual(first, {"already_complete": 0, "newly_scored": 4})
        self.assertEqual(second, {"already_complete": 4, "newly_scored": 0})
        self.assertEqual(score.call_count, calls_after_first)
        self.assertEqual(len(payloads), 4)
        self.assertEqual({payload["target"] for payload in payloads}, {"BU", "BM"})
        self.assertTrue(all("question_sha256" in payload for payload in payloads))

    def test_capped_condition_requires_its_pinned_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RecordStore(Path(tmp), run_fingerprint="test-run")
            with self.assertRaisesRegex(ValueError, "Missing capping assets"):
                run_belief_scoring_stage(
                    nn.Linear(1, 1),
                    object(),
                    [make_pair(1, "BU")],
                    ("p25",),
                    {},
                    store,
                    batch_size=1,
                )


if __name__ == "__main__":
    unittest.main()
