from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from cedar_m2.data.belief_r import (
    BeliefPair,
    BeliefRow,
    build_protocol_splits,
    load_protocol_splits,
    pair_belief_rows,
    read_belief_rows,
)


def make_pair(index: int, *, target: str, modus: str) -> BeliefPair:
    ground_truth_t = "A" if modus == "ponens" else "B"
    ground_truth_t1 = "C" if target == "BU" else ground_truth_t
    dataset_id = f"{index}-strong"
    common = {
        "modus": modus,
        "relation_type": ("If-Event-Then-Event" if index % 3 else "If-Event-Then-MentalState"),
        "atomic_idx": str(index),
        "dataset_id": dataset_id,
    }
    time_t = BeliefRow(
        question=f"time t question {index}",
        ground_truth=ground_truth_t,
        step="time_t",
        agreement_level="",
        **common,
    )
    time_t1 = BeliefRow(
        question=f"time t1 question {index}",
        ground_truth=ground_truth_t1,
        step="time_t1",
        agreement_level=str(4 + index % 2),
        **common,
    )
    return BeliefPair(time_t=time_t, time_t1=time_t1)


class BeliefCsvTest(unittest.TestCase):
    def test_read_and_pair_multiline_questions(self) -> None:
        fieldnames = [
            "questions",
            "ground_truth",
            "step",
            "modus",
            "types_of_relation",
            "agreement_lv",
            "atomic_idx",
            "dataset_id",
            "a",
            "b",
            "c",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "belief.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "questions": "premise one\npremise two",
                        "ground_truth": "a",
                        "step": "time_t",
                        "modus": "ponens",
                        "types_of_relation": "If-Event-Then-Event",
                        "agreement_lv": "",
                        "atomic_idx": "0",
                        "dataset_id": "0-strong",
                    }
                )
                writer.writerow(
                    {
                        "questions": "premise one\nnew evidence",
                        "ground_truth": "c",
                        "step": "time_t1",
                        "modus": "ponens",
                        "types_of_relation": "If-Event-Then-Event",
                        "agreement_lv": "5",
                        "atomic_idx": "0",
                        "dataset_id": "0-strong",
                    }
                )

            rows = read_belief_rows(path)
            pairs = pair_belief_rows(rows)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].target, "BU")
            self.assertIn("\n", pairs[0].time_t.question)


class ProtocolSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs = []
        for index in range(320):
            modus = "ponens" if index % 2 == 0 else "tollens"
            self.pairs.append(make_pair(index, target="BU", modus=modus))
        for index in range(1000, 1180):
            modus = "ponens" if index % 2 == 0 else "tollens"
            self.pairs.append(make_pair(index, target="BM", modus=modus))

    def test_protocol_counts_are_disjoint_and_deterministic(self) -> None:
        first = build_protocol_splits(self.pairs)
        second = build_protocol_splits(self.pairs)

        self.assertEqual(len(first.development), 50)
        self.assertEqual(len(first.pilot_bu), 100)
        self.assertEqual(len(first.pilot_bm), 100)
        self.assertEqual(len(first.extension_bu), 100)
        self.assertEqual(first.all_sample_ids(), second.all_sample_ids())
        self.assertEqual(len(first.all_sample_ids()), len(set(first.all_sample_ids())))

    def test_pilot_bu_preserves_balanced_modus(self) -> None:
        splits = build_protocol_splits(self.pairs)
        counts = Counter(item.time_t1.modus for item in splits.pilot_bu)
        self.assertLessEqual(abs(counts["ponens"] - counts["tollens"]), 1)

    def _manifest(self) -> dict[str, object]:
        splits = build_protocol_splits(self.pairs)
        manifest = splits.manifest()
        manifest["seed"] = 20260711
        manifest["counts"] = {
            "paired_items": len(self.pairs),
            "development": 50,
            "pilot_bu": 100,
            "pilot_bm": 100,
            "extension_bu": 100,
        }
        return manifest

    def test_frozen_manifest_is_loaded_only_when_it_matches_selection(self) -> None:
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = load_protocol_splits(path, self.pairs)

        self.assertEqual(
            loaded.all_sample_ids(), build_protocol_splits(self.pairs).all_sample_ids()
        )

    def test_frozen_manifest_rejects_metadata_and_order_changes(self) -> None:
        manifest = self._manifest()
        manifest["pilot_bu"][0]["target"] = "BM"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                load_protocol_splits(path, self.pairs)

        manifest = self._manifest()
        manifest["development"][0], manifest["development"][1] = (
            manifest["development"][1],
            manifest["development"][0],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reordered.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "preregistered selection"):
                load_protocol_splits(path, self.pairs)


if __name__ == "__main__":
    unittest.main()
