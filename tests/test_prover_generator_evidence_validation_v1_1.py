from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "m2_2b_formal" / "prover_generator_evidence_validation_v1_1"
for path in (DATA, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_prover_generator_evidence_validation_v1_1 as analyzer
import evidence_validation_common as common
import validate_evidence_validation_v1_1 as validator


class EvidenceValidationV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = common.select_validation_rows()

    def test_calibration_selects_evidence_and_excludes_social(self) -> None:
        decision = common.validate_module_selection()
        self.assertEqual(decision["evidence_module"]["status"], "selected_for_internal_validation")
        self.assertEqual(decision["evidence_module"]["v_minus_f"], 13 / 24)
        self.assertEqual(decision["evidence_module"]["family_directions"], [9, 0, 3])
        self.assertEqual(decision["social_module"]["status"], "excluded_from_formal_prover_evaluation")
        self.assertEqual(decision["social_module"]["ao_target_rate"], 17 / 24)

    def test_validation_view_is_balanced_and_unseen(self) -> None:
        self.assertEqual(len(self.rows), 96)
        self.assertEqual(Counter(row["arm"] for row in self.rows), Counter({arm: 24 for arm in common.EVIDENCE_ARMS}))
        self.assertEqual(len({row["family_id"] for row in self.rows}), 12)
        self.assertEqual(len({row["base_id"] for row in self.rows}), 24)
        bases = {row["base_id"]: row for row in self.rows}
        self.assertEqual(set(Counter(row["position_transition"] for row in bases.values()).values()), {2})
        self.assertEqual(set(Counter((row["position_transition"], row["variant"]) for row in bases.values()).values()), {1})

    def test_paired_order_is_complete_and_adjacent(self) -> None:
        cells = common.ordered_run_cells(self.rows)
        self.assertEqual(len(cells), 192)
        for index in range(0, len(cells), 2):
            left, right = cells[index], cells[index + 1]
            self.assertEqual(left[0]["id"], right[0]["id"])
            self.assertEqual((left[1], right[1]), ("uncapped", "p25"))

    def test_pending_release_passes_automatic_checks_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "EVIDENCE_ONLY_AMENDMENT.json",
                "GATE_SPEC.json",
                "HUMAN_REVIEW_CARRY_FORWARD.json",
            ):
                (root / name).write_bytes((DATA / name).read_bytes())
            (root / "validation_input_fingerprints.json").write_text(
                json.dumps({"status": "pending", "gpu_used": False}), encoding="utf-8"
            )
            result = validator.validate(root)
        self.assertTrue(result["passed"])
        self.assertFalse(result["release_ready"])
        self.assertFalse(result["release_checks"]["fixed_qwen_and_p25_asset_preflight"])

    def test_valid_preflight_authorizes_only_evidence_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "EVIDENCE_ONLY_AMENDMENT.json",
                "GATE_SPEC.json",
                "HUMAN_REVIEW_CARRY_FORWARD.json",
            ):
                (root / name).write_bytes((DATA / name).read_bytes())
            preflight = {
                "status": "passed",
                "gpu_used": False,
                "model": {"repo": common.MODEL_REPO, "revision": common.MODEL_REVISION},
                "answer_label_token_ids": {"A": [32], "B": [33], "C": [34], "D": [35]},
                "rendered_prompts_sha256": "0" * 64,
                "tokenized_input_ids_sha256": "1" * 64,
                "selection": {
                    "rows": 96,
                    "families": 12,
                    "bases": 24,
                    "ids_sha256": common.canonical_sha256([row["id"] for row in self.rows]),
                    "messages_sha256": common.canonical_sha256(
                        [{"id": row["id"], "messages": row["messages"]} for row in self.rows]
                    ),
                },
                "official_p25": {
                    "repo": common.ASSET_REPO,
                    "revision": common.ASSET_REVISION,
                    "axis_sha256": common.EXPECTED_AXIS_SHA256,
                    "capping_config_sha256": common.EXPECTED_CAP_CONFIG_SHA256,
                    "experiment": common.CAP_EXPERIMENT,
                    "layers": list(common.CAP_LAYERS),
                },
            }
            (root / "validation_input_fingerprints.json").write_text(json.dumps(preflight), encoding="utf-8")
            result = validator.validate(root, require_release_gate=True)
            gate = json.loads((root / "EVIDENCE_VALIDATION_GATE.json").read_text(encoding="utf-8"))
        self.assertTrue(result["release_ready"])
        self.assertTrue(gate["may_run_paired_validation"])
        self.assertEqual(gate["p25_authorized_scope"], "evidence_only_internal_validation")
        self.assertFalse(gate["p1_authorized"])
        self.assertFalse(gate["social_formal_prover_authorized"])

    def test_truncation_bounds_are_intention_to_treat(self) -> None:
        rows = [
            {"arm": "V", "condition": "uncapped", "outcome_class": "induced_target", "truncated": False},
            {"arm": "V", "condition": "uncapped", "outcome_class": "invalid_or_abstain", "truncated": True},
            {"arm": "V", "condition": "p25", "outcome_class": "initial", "truncated": False},
            {"arm": "V", "condition": "p25", "outcome_class": "invalid_or_abstain", "truncated": True},
        ]
        self.assertEqual(analyzer.rate_bounds(rows, "V", "uncapped"), [0.5, 1.0])
        self.assertEqual(analyzer.rate_bounds(rows, "V", "p25"), [0.0, 0.5])
        self.assertEqual(analyzer.subtract_bounds([0.0, 0.5], [0.5, 1.0]), [-1.0, 0.0])

    def test_analyzer_accepts_any_outcome_direction(self) -> None:
        synthetic = []
        for row in self.rows:
            for condition in common.CONDITIONS:
                if row["arm"] in {"N", "R"}:
                    outcome = "initial"
                elif row["arm"] == "V":
                    outcome = "induced_target" if condition == "uncapped" or row["variant"] == "medium" else "initial"
                else:
                    outcome = "induced_target" if condition == "uncapped" else "initial"
                synthetic.append(
                    {
                        **{key: row[key] for key in ("id", "base_id", "family_id", "variant", "arm", "initial_answer", "target_answer", "position_transition")},
                        "condition": condition,
                        "outcome_class": outcome,
                        "truncated": False,
                        "final_answer_parse_status": "passed",
                        "answer_logits": {"target_minus_initial": 1.0 if outcome == "induced_target" else -1.0, "option_mass": 1.0, "full_vocab_entropy": 0.0},
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "FORMAL_AUDIT.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
            run = root / "formal-paired-uncapped-p25-seed0"
            run.mkdir()
            (run / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in synthetic), encoding="utf-8")
            result = analyzer.analyze(root)
        self.assertEqual(result["status"], "complete")
        self.assertLess(result["target_following_cap_effects"]["F"]["estimate"], 0)
        self.assertFalse(result["p25_authorized_for_additional_development_retesting"])


if __name__ == "__main__":
    unittest.main()
