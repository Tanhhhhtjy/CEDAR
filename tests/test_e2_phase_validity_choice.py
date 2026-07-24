from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "m2_2b_formal" / "e2_phase_validity_choice"
for path in (ROOT / "src", DATA, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_e2_phase_validity_choice as analyzer  # noqa: E402
import audit_e2_phase_validity_choice as auditor  # noqa: E402
import e2_common as common  # noqa: E402
import run_e2_phase_validity_choice as runner  # noqa: E402

from cedar_m2.interventions.capping import LayerCap  # noqa: E402
from cedar_m2.interventions.phase_capping import PhaseActivationCapper  # noqa: E402


class IdentityBlock(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 0


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList([IdentityBlock()]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model.layers[0](value)


class E2PhaseValidityChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.answers = common.read_jsonl(DATA / "answer_inputs96.jsonl")
        cls.validity = common.read_jsonl(DATA / "validity_inputs48.jsonl")
        cls.mapping = common.read_json(DATA / "validity_mapping.json")
        cls.preflight = common.read_json(DATA / "tokenizer_preflight.json")

    def test_input_matrix_and_label_mapping(self) -> None:
        self.assertEqual(len(self.answers), 96)
        self.assertEqual(len(self.validity), 48)
        self.assertEqual(
            Counter(row["source_arm"] for row in self.validity), Counter({"V": 24, "F": 24})
        )
        self.assertEqual(
            Counter(row["gold_label"] for row in self.validity), Counter({"A": 24, "B": 24})
        )
        sufficient = self.mapping["sufficient_label_by_base"]
        self.assertEqual(Counter(sufficient.values()), Counter({"A": 12, "B": 12}))
        self.assertEqual(self.mapping["observed_balance"]["balanced_position_transitions"], 10)
        self.assertFalse(
            self.mapping["observed_balance"]["joint_exact_transition_balance_feasible"]
        )

    def test_validity_wrapper_preserves_a_then_b_presentation(self) -> None:
        for row in self.validity:
            visible = row["messages"][1]["content"]
            question = visible.index("The proposed target")
            self.assertLess(visible.index("A) ", question), visible.index("B) ", question))
            self.assertIn("BEGIN QUOTED UPDATE MATERIAL", visible)
            self.assertIn("logically suffice both", visible)
            self.assertNotIn("source_arm", visible)
            self.assertNotIn("gold_semantic", visible)

    def test_cpu_tokenizer_preflight(self) -> None:
        self.assertEqual(self.preflight["status"], "passed")
        self.assertFalse(self.preflight["gpu_used"])
        for group in ("answer_labels", "validity_labels"):
            for item in self.preflight[group]["labels"].values():
                self.assertTrue(item["single_contextual_token"])

    def test_formal_matrix_has_576_cells(self) -> None:
        rows = [("answer", row) for row in self.answers] + [
            ("validity", row) for row in self.validity
        ]
        matrix = runner.cells(rows, "formal")
        self.assertEqual(len(matrix), 576)
        self.assertEqual(
            Counter(condition for _, _, condition in matrix),
            Counter({condition: 144 for condition in common.FORMAL_CONDITIONS}),
        )

    def test_strict_validity_parser(self) -> None:
        response = '<think>check</think>\n{"judgment":"A"}'
        judgment, status, reasoning, sections = runner.strict_validity(response, truncated=False)
        self.assertEqual((judgment, status, reasoning), ("A", "passed", "check"))
        self.assertTrue(sections["ordered_single_pair"])
        self.assertEqual(
            runner.strict_validity("```json\n{}\n```", truncated=False)[1], "invalid_json"
        )

    def test_prefill_only_caps_prefill_and_not_decode(self) -> None:
        model = DummyModel()
        cap = LayerCap(layer=0, direction=torch.tensor([1.0, 0.0]), threshold=0.0)
        with PhaseActivationCapper(model, {0: cap}, enabled_phases=("prefill",)) as capper:
            prefill = model(torch.tensor([[[2.0, 1.0], [3.0, 1.0]]]))
            decode = model(torch.tensor([[[4.0, 1.0]]]))
        self.assertTrue(torch.equal(prefill[..., 0], torch.zeros_like(prefill[..., 0])))
        self.assertEqual(float(decode[0, 0, 0]), 4.0)
        telemetry = capper.telemetry()["by_layer"]["0"]
        self.assertEqual(telemetry["prefill"]["applied_positions"], 2)
        self.assertEqual(telemetry["decode"]["applied_positions"], 0)
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)

    def test_decode_only_and_monitor_are_nonleaking(self) -> None:
        model = DummyModel()
        cap = LayerCap(layer=0, direction=torch.tensor([1.0, 0.0]), threshold=0.0)
        with PhaseActivationCapper(model, {0: cap}, enabled_phases=("decode",)) as capper:
            prefill = model(torch.tensor([[[2.0, 1.0], [3.0, 1.0]]]))
            decode = model(torch.tensor([[[4.0, 1.0]]]))
        self.assertEqual(float(prefill[0, 0, 0]), 2.0)
        self.assertEqual(float(decode[0, 0, 0]), 0.0)
        self.assertEqual(capper.telemetry()["by_layer"]["0"]["decode"]["triggered_positions"], 1)

        original = torch.tensor([[[5.0, 1.0], [6.0, 1.0]]])
        with PhaseActivationCapper(
            model,
            {0: cap},
            enabled_phases=("prefill", "decode"),
            apply_changes=False,
        ) as monitor:
            observed = model(original)
        self.assertTrue(torch.equal(observed, original))
        self.assertEqual(monitor.telemetry()["by_layer"]["0"]["prefill"]["triggered_positions"], 2)

    def test_human_review_is_single_reviewer_and_complete(self) -> None:
        review = json.loads((DATA / "human_review.json").read_text(encoding="utf-8"))
        self.assertEqual(review["status"], "passed")
        self.assertEqual(review["reviewer_count_required"], 1)
        self.assertEqual(review["reviewed_items"], 48)
        self.assertEqual(review["failed_items"], 0)

    def test_smoke_auditor_enforces_noop_equivalence_and_runtime_hash(self) -> None:
        def phase_stats(enabled: set[str]) -> dict[str, object]:
            return {
                "enabled_phases": sorted(enabled),
                "apply_changes": enabled != {"prefill", "decode"},
                "by_layer": {
                    str(layer): {
                        phase: {
                            "forward_calls": 1,
                            "observed_positions": 2 if phase == "prefill" else 1,
                            "applied_positions": (2 if phase == "prefill" else 1)
                            if phase in enabled
                            else 0,
                            "triggered_positions": 1 if phase in enabled else 0,
                            "projection_correction_sum": 0.5 if phase in enabled else 0.0,
                        }
                        for phase in ("prefill", "decode")
                    }
                    for layer in common.CAP_LAYERS
                },
            }

        def smoke_row(assay: str, item_id: str, condition: str) -> dict[str, object]:
            row: dict[str, object] = {
                "assay": assay,
                "id": item_id,
                "condition": condition,
                "content_retry_count": 0,
                "answer_repair_performed": False,
                "hook_counts_before": {str(layer): 0 for layer in common.CAP_LAYERS},
                "hook_counts_after": {str(layer): 0 for layer in common.CAP_LAYERS},
                "prompt_sha256": f"prompt-{assay}-{item_id}",
                "prompt_token_ids_sha256": f"prompt-tokens-{assay}-{item_id}",
                "response_token_ids_sha256": f"response-{assay}-{item_id}",
                "final_answer_parse_status": "passed",
                "truncated": False,
                "selected_answer_logit_index": 10,
                "answer_logit_capture_indices": [10],
                "answer_logit_candidates": [{"response_index": 10, "generated_label": "A"}],
                "gold_correct": True,
                "answer_logits": {
                    "logits": {"A": 2.0, "B": 1.0},
                    "option_argmax": "A",
                    "option_mass": 0.9,
                },
                "phase_telemetry": None,
                "full_cap_stats": None,
            }
            if assay == "answer":
                row.update({"chosen_option": "A", "outcome_class": "initial"})
            else:
                row.update({"chosen_label": "A", "chosen_semantic": "sufficient"})
            if condition in {"monitor_only", "full_p25"}:
                row["phase_telemetry"] = phase_stats({"prefill", "decode"})
                assert isinstance(row["phase_telemetry"], dict)
                row["phase_telemetry"]["apply_changes"] = False
            elif condition == "prefill_only":
                row["phase_telemetry"] = phase_stats({"prefill"})
            elif condition == "decode_only":
                row["phase_telemetry"] = phase_stats({"decode"})
            if condition == "full_p25":
                row["full_cap_stats"] = {
                    "by_layer": {
                        str(layer): {
                            "total_positions": 3,
                            "triggered_positions": 2,
                            "projection_correction_sum": 1.0,
                        }
                        for layer in common.CAP_LAYERS
                    }
                }
            return row

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "smoke-all-seed0"
            run.mkdir(parents=True)
            manifest = common.read_json(DATA / "manifest.json")
            rows = [
                smoke_row(assay, item_id, condition)
                for assay in ("answer", "validity")
                for item_id in manifest["smoke_ids"][assay]
                for condition in common.SMOKE_CONDITIONS
            ]
            results = run / "results.jsonl"
            results.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            runtime = {
                "complete_results": len(rows),
                "results_sha256": common.file_sha256(results),
                "truncations": [],
                "parse_failures": [],
                "missing_logits": [],
            }
            (run / "runtime.json").write_text(json.dumps(runtime) + "\n", encoding="utf-8")
            result = auditor.audit(root, scope="smoke", assay="all")
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["smoke_noop_equivalence"]["passed"])

            rows[1]["response_token_ids_sha256"] = "monitor-changed-output"
            results.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            runtime["results_sha256"] = common.file_sha256(results)
            (run / "runtime.json").write_text(json.dumps(runtime) + "\n", encoding="utf-8")
            changed = auditor.audit(root, scope="smoke", assay="all")
            self.assertEqual(changed["status"], "failed")
            self.assertTrue(
                any(failure.startswith("smoke_noop_field:") for failure in changed["failures"])
            )

            runtime["results_sha256"] = "0" * 64
            (run / "runtime.json").write_text(json.dumps(runtime) + "\n", encoding="utf-8")
            bad_hash = auditor.audit(root, scope="smoke", assay="all")
            self.assertIn("runtime_results_hash_mismatch", bad_hash["failures"])

    def test_analyzer_accepts_complete_synthetic_matrix(self) -> None:
        rows = []
        for row in self.answers:
            for condition in common.FORMAL_CONDITIONS:
                target = row["arm"] == "V" or (
                    row["arm"] == "F" and condition in {"full_p25", "decode_only"}
                )
                initial = row["initial_answer"]
                induced = row["target_answer"]
                logits = {label: -2.0 for label in ("A", "B", "C", "D")}
                logits[initial] = 0.0
                logits[induced] = 1.0 if target else -1.0
                rows.append(
                    {
                        "assay": "answer",
                        "id": row["id"],
                        "base_id": row["base_id"],
                        "family_id": row["family_id"],
                        "variant": row["variant"],
                        "position_transition": row["position_transition"],
                        "arm": row["arm"],
                        "condition": condition,
                        "initial_answer": initial,
                        "target_answer": induced,
                        "outcome_class": "induced_target" if target else "initial",
                        "answer_logits": {
                            "logits": logits,
                            "target_minus_initial": logits[induced] - logits[initial],
                        },
                        "final_answer_parse_status": "passed",
                    }
                )
        for row in self.validity:
            for condition in common.FORMAL_CONDITIONS:
                rows.append(
                    {
                        "assay": "validity",
                        "id": row["id"],
                        "base_id": row["base_id"],
                        "family_id": row["family_id"],
                        "variant": row["variant"],
                        "position_transition": row["position_transition"],
                        "source_arm": row["source_arm"],
                        "sufficient_label": row["sufficient_label"],
                        "condition": condition,
                        "chosen_semantic": row["gold_semantic"],
                        "final_answer_parse_status": "passed",
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "E2A_AUDIT.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
            run = root / "formal-all-seed0"
            run.mkdir()
            (run / "results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            previous = analyzer.BOOTSTRAP_DRAWS
            analyzer.BOOTSTRAP_DRAWS = 200
            try:
                result = analyzer.analyze(root)
            finally:
                analyzer.BOOTSTRAP_DRAWS = previous
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["rows"], 576)
        self.assertIn(result["route_gate"][0], {"T", "E", "C", "F"})


if __name__ == "__main__":
    unittest.main()
