from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "m2_2b_formal" / "prover_generator_development_v1_1"
SRC = ROOT / "src"
for path in (DATA, SRC, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import development_v1_1_common as common
import finalize_social_template
import validate_prover_generator_development_v1_1 as validator


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("v1_1_runner", ROOT / "scripts" / "run_prover_generator_v1_1_uncapped.py")
analyzer = load_script("v1_1_analyzer", ROOT / "scripts" / "analyze_prover_generator_v1_1.py")
tokenizer_preflight = load_script(
    "v1_1_tokenizer_preflight",
    DATA / "run_qwen_tokenizer_preflight.py",
)


class DevelopmentV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bases = common.read_jsonl(DATA / "measurement_bases48.jsonl")
        cls.arms = common.read_jsonl(DATA / "development_candidates336.jsonl")

    def test_parent_release_is_hash_frozen(self) -> None:
        common.validate_parent_files()

    def test_parent_text_hash_is_lf_crlf_portable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lf = root / "lf.json"
            crlf = root / "crlf.json"
            changed = root / "changed.json"
            lf.write_bytes(b'{"value":1}\n')
            crlf.write_bytes(b'{"value":1}\r\n')
            changed.write_bytes(b'{"value":2}\n')
            self.assertEqual(
                common.canonical_text_sha256(lf),
                common.canonical_text_sha256(crlf),
            )
            self.assertNotEqual(
                common.canonical_text_sha256(lf),
                common.canonical_text_sha256(changed),
            )

    def test_hf_home_cache_root_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hub = root / "hub"
            hub.mkdir()
            self.assertEqual(tokenizer_preflight.resolve_transformers_cache_dir(root), hub)
            self.assertEqual(tokenizer_preflight.resolve_transformers_cache_dir(hub), hub)
            self.assertIsNone(tokenizer_preflight.resolve_transformers_cache_dir(None))

    def test_candidate_matrix_and_position_balance(self) -> None:
        self.assertEqual(len(self.bases), 48)
        self.assertEqual(len(self.arms), 336)
        self.assertEqual(Counter(row["arm"] for row in self.arms), Counter({arm: 48 for arm in common.CANDIDATE_ARMS}))
        self.assertEqual(set(Counter(row["position_transition"] for row in self.bases).values()), {4})
        split_variant = Counter((row["development_split"], row["position_transition"], row["variant"]) for row in self.bases)
        self.assertTrue(all(value == 1 for value in split_variant.values()))
        self.assertEqual(len(split_variant), 48)

    def test_measurement_templates_are_neutral_and_matched(self) -> None:
        for base in self.bases:
            combined = base["english_v_text"] + base["english_f_text"]
            self.assertNotIn("Reviewer", combined)
            self.assertNotIn("Morgan", combined)
            self.assertNotIn("audit channel", combined)
            self.assertEqual(common.whitespace_token_count(base["english_v_text"]), common.whitespace_token_count(base["english_f_text"]))
            social = list(base["social_templates"].values())
            self.assertEqual(len({common.whitespace_token_count(text) for text in social}), 1)
            self.assertEqual(len({common.canonical_token_count(text) for text in social}), 1)

    def test_independent_automatic_preflight_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "measurement_bases48.jsonl",
                "development_candidates336.jsonl",
                "construction_audit48.jsonl",
                "family_split.json",
                "GATE_SPEC.json",
                "build_report.json",
                "deterministic_rebuild_report.json",
                "tokenizer_preflight.json",
                "HUMAN_NATURALIZATION_AUDIT.json",
                "ZERO_GPU_GATE.json",
                "manifest.json",
            ):
                (root / name).write_bytes((DATA / name).read_bytes())
            result = validator.validate(root)
            self.assertTrue(result["passed"])
            self.assertFalse(result["release_ready"])

    def test_runner_calibration_selection(self) -> None:
        evidence = runner.select_rows(self.arms, "formal", "evidence")
        social = runner.select_rows(self.arms, "formal", "social-candidates")
        self.assertEqual(len(evidence), 96)
        self.assertEqual(len(social), 96)
        self.assertEqual(set(row["development_split"] for row in evidence + social), {"template_calibration"})
        self.assertEqual(Counter(row["arm"] for row in evidence), Counter({arm: 24 for arm in common.EVIDENCE_ARMS}))
        self.assertEqual(Counter(row["arm"] for row in social), Counter({arm: 24 for arm in common.SOCIAL_CANDIDATE_ARMS}))

    def test_global_social_finalizer_never_selects_by_item(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            audit = temp / "analysis.json"
            audit.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "selected_social_template": "S1",
                        "itemwise_template_selection": False,
                        "selection_rule": "weakest_globally_passing_template_s1_wins_ties",
                    }
                ),
                encoding="utf-8",
            )
            result = finalize_social_template.finalize(DATA, audit, temp / "release")
            self.assertEqual(result["selected_social_template"], "S1")
            self.assertEqual(result["rows"], 288)
            self.assertEqual(result["arm_counts"], {arm: 48 for arm in sorted(common.FINAL_ARMS)})

    def test_calibration_analyzer_selects_weakest_global_template(self) -> None:
        evidence_rows = runner.select_rows(self.arms, "formal", "evidence")
        social_rows = runner.select_rows(self.arms, "formal", "social-candidates")

        def result(row: dict, target: bool, margin: float) -> dict:
            return {
                **{key: row[key] for key in ("id", "base_id", "family_id", "variant", "arm", "initial_answer", "target_answer", "gold_answer", "position_transition")},
                "outcome_class": "induced_target" if target else "initial",
                "chosen_option": row["target_answer"] if target else row["initial_answer"],
                "response": json.dumps({"answer": row["target_answer"] if target else row["initial_answer"]}),
                "final_answer_parse_status": "passed",
                "truncated": False,
                "answer_logits": {"target_minus_initial": margin, "option_mass": 1.0, "full_vocab_entropy": 0.1},
            }

        evidence = [result(row, row["arm"] == "V", 2.0 if row["arm"] == "V" else -2.0) for row in evidence_rows]
        social = [result(row, row["arm"] in {"S1", "S2"}, 1.0 if row["arm"] in {"S1", "S2"} else -1.0) for row in social_rows]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "TEMPLATE_CALIBRATION_FORMAL_AUDIT.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
            for track, rows in (("evidence", evidence), ("social-candidates", social)):
                run_dir = root / analyzer.run_name(track)
                run_dir.mkdir(parents=True)
                (run_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            old_data_root = analyzer.DATA_ROOT
            try:
                analyzer.DATA_ROOT = DATA
                analysis = analyzer.analyze(root)
            finally:
                analyzer.DATA_ROOT = old_data_root
            self.assertEqual(analysis["status"], "passed")
            self.assertEqual(analysis["selected_social_template"], "S1")
            self.assertFalse(analysis["p25_authorized"])


if __name__ == "__main__":
    unittest.main()
