from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "m2_2b_formal" / "e2_new_salt_confirmation"
for path in (ROOT / "src", DATA, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_e2_new_salt_confirmation as analyzer  # noqa: E402
import audit_e2_new_salt_confirmation as auditor  # noqa: E402
import e2_new_salt_run_common as common  # noqa: E402


class E2NewSaltConfirmationTests(unittest.TestCase):
    def test_frozen_formal_and_smoke_selections(self) -> None:
        formal = common.select_rows("formal")
        smoke = common.select_rows("smoke")
        self.assertEqual(len(formal), 96)
        self.assertEqual(len(smoke), 24)
        self.assertEqual(
            Counter(row["arm"] for row in formal),
            Counter({arm: 24 for arm in common.ARMS}),
        )
        self.assertEqual(
            Counter(row["arm"] for row in smoke),
            Counter({arm: 6 for arm in common.ARMS}),
        )
        self.assertEqual(len({row["base_id"] for row in smoke}), 6)
        expected_variant = {
            error_class: "easy" if index % 2 == 0 else "medium"
            for index, error_class in enumerate(sorted(common.ERROR_CLASSES))
        }
        selected = {(row["error_class"], row["variant"]) for row in smoke if row["arm"] == "F"}
        self.assertEqual(selected, set(expected_variant.items()))

    def test_cpu_gate_binds_runtime_code_and_matrix(self) -> None:
        gate = common.validate_cpu_gate()
        self.assertTrue(gate["gpu_authorized"])
        self.assertEqual(gate["smoke_rows"], 48)
        self.assertEqual(gate["formal_rows"], 192)
        self.assertEqual(gate["conditions"], list(common.CONDITIONS))
        self.assertEqual(
            set(gate["code_hashes"]),
            {
                "run_common_raw_sha256",
                "runner_raw_sha256",
                "auditor_raw_sha256",
                "analyzer_raw_sha256",
            },
        )

    def test_primary_metric_and_bootstrap_are_frozen(self) -> None:
        row = {
            "initial_answer": "A",
            "target_answer": "B",
            "answer_logits": {"logits": {"A": 1.0, "B": 4.0, "C": 2.0, "D": 3.0}},
        }
        self.assertEqual(analyzer.metric(row, "m_target"), 3.0)
        self.assertEqual(analyzer.metric(row, "m_background"), 1.5)
        self.assertEqual(analyzer.metric(row, "m_specific"), 1.5)
        self.assertEqual(analyzer.metric(row, "m_general"), 2.0)
        previous = analyzer.BOOTSTRAP_DRAWS
        analyzer.BOOTSTRAP_DRAWS = 100
        try:
            first = analyzer.bootstrap_summary([("f1", 1.0), ("f2", 3.0)])
            second = analyzer.bootstrap_summary([("f1", 1.0), ("f2", 3.0)])
        finally:
            analyzer.BOOTSTRAP_DRAWS = previous
        self.assertEqual(first, second)
        self.assertEqual(first["estimate"], 2.0)
        self.assertEqual(first["bootstrap_seed"], common.BOOTSTRAP_SEED)

    def test_auditor_requires_finite_abcd_logits(self) -> None:
        valid = {"answer_logits": {"logits": {label: 0.0 for label in "ABCD"}}}
        missing = {"answer_logits": None}
        invalid = {"answer_logits": {"logits": {"A": 0.0}}}
        self.assertTrue(auditor.logits_finite(valid))
        self.assertTrue(auditor.logits_finite(missing))
        self.assertFalse(auditor.logits_finite(invalid))


if __name__ == "__main__":
    unittest.main()
