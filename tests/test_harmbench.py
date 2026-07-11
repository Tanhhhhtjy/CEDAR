from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from cedar_m2.data.harmbench import load_harmbench_split, read_harmbench_behaviors
from cedar_m2.eval.harmbench import (
    HARMBENCH_CLASSIFIER_PROMPT,
    parse_classifier_label,
    render_classifier_prompt,
    summarize_harmbench,
    summarize_harmbench_greedy,
)
from cedar_m2.personas import load_persona_config


class HarmBenchDataTest(unittest.TestCase):
    def test_reads_only_standard_rows(self) -> None:
        fields = [
            "Behavior",
            "FunctionalCategory",
            "SemanticCategory",
            "Tags",
            "ContextString",
            "BehaviorID",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "behaviors.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "Behavior": "standard request",
                        "FunctionalCategory": "standard",
                        "SemanticCategory": "harmful",
                        "Tags": "tag_a, tag_b",
                        "ContextString": "",
                        "BehaviorID": "standard-id",
                    }
                )
                writer.writerow(
                    {
                        "Behavior": "contextual request",
                        "FunctionalCategory": "contextual",
                        "SemanticCategory": "harmful",
                        "Tags": "context",
                        "ContextString": "context",
                        "BehaviorID": "context-id",
                    }
                )
            behaviors = read_harmbench_behaviors(path)
        self.assertEqual(len(behaviors), 1)
        self.assertEqual(behaviors[0].behavior_id, "standard-id")
        self.assertEqual(behaviors[0].tags, ("tag_a", "tag_b"))

    def test_test_split_is_sealed_before_checkout_access(self) -> None:
        with self.assertRaisesRegex(PermissionError, "sealed"):
            load_harmbench_split("/path/that/must/not/be/read", "test")


class PersonaConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.config = load_persona_config(root / "configs/pilot/harmbench_personas.yaml")

    def test_all_categories_render_without_changing_behavior(self) -> None:
        behavior = "EXACT {behavior} TEXT"
        self.assertEqual(len(self.config.personas), 6)
        for category in self.config.personas:
            messages = self.config.render_messages(category, behavior)
            self.assertEqual(
                [item["role"] for item in messages],
                ["system", "user", "assistant", "user"],
            )
            self.assertIn(behavior, messages[-1]["content"])
            self.assertIn(self.config.personas[category].name, messages[0]["content"])

    def test_development_template_cannot_open_test(self) -> None:
        with self.assertRaisesRegex(PermissionError, "requires 'frozen'"):
            self.config.require_frozen()


class HarmBenchEvaluationTest(unittest.TestCase):
    def test_classifier_prompt_and_label_parser(self) -> None:
        prompt = render_classifier_prompt("a behavior", "a generation")
        self.assertIn("[BEHAVIOR]:\na behavior", prompt)
        self.assertTrue(prompt.endswith("Answer: [/INST]"))
        self.assertEqual(parse_classifier_label(" Yes "), 1)
        self.assertEqual(parse_classifier_label("no."), 0)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            parse_classifier_label("maybe")
        self.assertEqual(
            hashlib.sha256(HARMBENCH_CLASSIFIER_PROMPT.encode()).hexdigest(),
            "788f4f6aa1491c433c4da76c9140cfc30966cea3ff3875c4d0fcb336d92f60e0",
        )

    def test_sampled_summary_applies_preregistered_gates(self) -> None:
        records = []
        for index in range(6):
            behavior_id = f"behavior-{index}"
            category = "harmful" if index % 2 else "illegal"
            for seed in (1, 2, 3):
                for condition, label in (
                    ("behavior_uncapped", 0),
                    ("persona_uncapped", 1),
                    ("persona_p25", 0),
                ):
                    records.append(
                        {
                            "behavior_id": behavior_id,
                            "semantic_category": category,
                            "condition": condition,
                            "decoding": "sampled",
                            "seed": seed,
                            "label": label,
                        }
                    )
        summary = summarize_harmbench(records, bootstrap_samples=200, bootstrap_seed=7)
        self.assertEqual(summary["asr"]["persona_uncapped"], 1.0)
        self.assertEqual(summary["absolute_reduction"], 1.0)
        self.assertTrue(summary["gates"]["attack_effective"])
        self.assertTrue(summary["gates"]["m2_1_pass"])

    def test_greedy_summary_is_separate(self) -> None:
        records = []
        for condition, label in (
            ("behavior_uncapped", 0),
            ("persona_uncapped", 1),
            ("persona_p25", 0),
        ):
            records.append(
                {
                    "behavior_id": "one",
                    "semantic_category": "illegal",
                    "condition": condition,
                    "decoding": "greedy",
                    "seed": None,
                    "label": label,
                }
            )
        summary = summarize_harmbench_greedy(records)
        self.assertEqual(summary["absolute_reduction"], 1.0)


if __name__ == "__main__":
    unittest.main()
