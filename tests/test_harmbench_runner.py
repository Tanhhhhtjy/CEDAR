from __future__ import annotations

import unittest

from cedar_m2.data.harmbench import HarmBenchBehavior
from cedar_m2.runners.harmbench import (
    CONDITIONS,
    build_generation_batches,
    expected_generation_count,
    generation_key,
)


def make_behavior(index: int) -> HarmBenchBehavior:
    return HarmBenchBehavior(
        behavior_id=f"id-{index}",
        behavior=f"request {index}",
        semantic_category="harmful",
        tags=(),
        context="",
    )


class HarmBenchRunnerTest(unittest.TestCase):
    def test_batches_have_stable_paired_seeds_and_expected_count(self) -> None:
        behaviors = [make_behavior(index) for index in range(5)]
        batches = build_generation_batches(behaviors, batch_size=2)
        self.assertEqual(sum(len(batch.behaviors) for batch in batches), 60)
        self.assertEqual(expected_generation_count(5), 60)

        first_seed_by_condition = {
            condition: next(
                batch.batch_seed
                for batch in batches
                if batch.condition == condition
                and batch.decoding == "sampled"
                and batch.seed == 20260711
                and batch.batch_index == 0
            )
            for condition in CONDITIONS
        }
        self.assertEqual(len(set(first_seed_by_condition.values())), 1)

    def test_generation_key_is_condition_specific(self) -> None:
        baseline = generation_key("one", "behavior_uncapped", "sampled", 7)
        persona = generation_key("one", "persona_uncapped", "sampled", 7)
        self.assertNotEqual(baseline, persona)


if __name__ == "__main__":
    unittest.main()
