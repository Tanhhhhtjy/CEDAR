from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from cedar_m2.data.m2_2b_v4_neutral import (
    ALLOWED_CHANGED_FIELDS,
    N_PROMPT,
    R_PROMPT,
    SYSTEM,
    changed_fields,
    neutralize_rows,
    split_vf_followup,
    unchanged_projection,
)

SOURCE = Path(
    "datasets/m2_2b_formal/v4_public_poc/m2_2b_v4_public_poc180_arms.jsonl"
)


def load_rows() -> list[dict]:
    return [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]


class NeutralPromptTransformTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = load_rows()
        cls.neutral = neutralize_rows(cls.source)

    def test_preserves_structure_and_all_non_prompt_fields(self) -> None:
        self.assertEqual(len(self.neutral), 180)
        self.assertEqual(
            Counter(row["arm"] for row in self.neutral),
            Counter({arm: 36 for arm in ("N", "R", "V", "F", "S")}),
        )
        self.assertEqual(
            [row["id"] for row in self.source],
            [row["id"] for row in self.neutral],
        )
        for source, neutral in zip(self.source, self.neutral, strict=True):
            self.assertLessEqual(set(changed_fields(source, neutral)), ALLOWED_CHANGED_FIELDS)
            self.assertEqual(unchanged_projection(source), unchanged_projection(neutral))

    def test_uses_frozen_system_n_and_r_prompts(self) -> None:
        self.assertTrue(all(row["system"] == SYSTEM for row in self.neutral))
        self.assertTrue(
            all(row["followup_user"] == N_PROMPT for row in self.neutral if row["arm"] == "N")
        )
        self.assertTrue(
            all(row["followup_user"] == R_PROMPT for row in self.neutral if row["arm"] == "R")
        )

    def test_vf_inner_observation_and_candidate_are_byte_preserved(self) -> None:
        checked = 0
        for source, neutral in zip(self.source, self.neutral, strict=True):
            if source["arm"] not in {"V", "F"}:
                continue
            self.assertEqual(
                split_vf_followup(source["followup_user"]),
                split_vf_followup(neutral["followup_user"]),
            )
            checked += 1
        self.assertEqual(checked, 72)

    def test_s_uses_frozen_target_label_and_option_text(self) -> None:
        for row in self.neutral:
            if row["arm"] != "S":
                continue
            target = row["induced_target_answer_id"]
            self.assertIn(
                f"answer should be {target} ({row['options'][target]})",
                row["followup_user"],
            )
            self.assertIn("Please reconsider carefully.", row["followup_user"])


if __name__ == "__main__":
    unittest.main()
