from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cedar_m2.data.belief_r import BeliefPair, BeliefRow
from cedar_m2.data.formal import read_formal_items, validate_formal_items


def make_pair() -> BeliefPair:
    common = {
        "modus": "ponens",
        "relation_type": "If-Event-Then-Event",
        "atomic_idx": "7",
        "dataset_id": "7-strong",
    }
    time_t = BeliefRow(
        question="Initial question",
        ground_truth="A",
        step="time_t",
        agreement_level="",
        options=("Alpha", "Beta", "Gamma"),
        **common,
    )
    time_t1 = BeliefRow(
        question="Updated question",
        ground_truth="C",
        step="time_t1",
        agreement_level="5",
        options=("Alpha", "Beta", "Gamma"),
        **common,
    )
    return BeliefPair(time_t=time_t, time_t1=time_t1)


def formal_record() -> dict[str, object]:
    return {
        "id": "7-strong",
        "question": "Updated question",
        "options": {"a": "Alpha", "b": "Beta", "c": "Gamma"},
        "ground_truth": "c",
        "should_update": True,
        "modus": "ponens",
        "agreement_lv": 5,
    }


class FormalDataTest(unittest.TestCase):
    def _read(self, records: list[dict[str, object]]):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "formal.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            return read_formal_items(path)

    def test_formal_payload_matches_pinned_pair_exactly(self) -> None:
        pair = make_pair()
        items = self._read([formal_record()])
        selected = validate_formal_items(items, [pair])

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].sample_id, pair.sample_id)
        self.assertEqual(selected[0].ground_truth, "C")

        altered = replace(selected[0], question="Changed question")
        with self.assertRaisesRegex(ValueError, "differs from pinned Belief-R"):
            validate_formal_items([altered], [pair])

    def test_rejects_inconsistent_update_label_and_duplicate_ids(self) -> None:
        record = formal_record()
        record["should_update"] = False
        with self.assertRaisesRegex(ValueError, "Inconsistent should_update"):
            self._read([record])

        with self.assertRaisesRegex(ValueError, "must be unique"):
            self._read([formal_record(), formal_record()])


if __name__ == "__main__":
    unittest.main()
