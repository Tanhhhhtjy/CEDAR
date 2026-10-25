"""Static and symbolic-certificate validator for the v4 public-protocol PoC."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
LABELS = {"A", "B", "C"}
ARMS = ("N", "R", "V", "F", "S")
TRANSITIONS = {("A", "B"), ("A", "C"), ("B", "A"), ("B", "C"), ("C", "A"), ("C", "B")}


def load_jsonl(name: str) -> list[dict]:
    rows = []
    with (HERE / name).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{name}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_label(base: dict, survivors: list[str]) -> str:
    if survivors == ["H1"]:
        return base["initial_answer_id"]
    if survivors == ["H2"]:
        assert base["matched_valid_subtype"] == "refute"
        return base["updated_answer_id"]
    if set(survivors) == {"H1", "H2"}:
        assert base["matched_valid_subtype"] == "undercut"
        return base["updated_answer_id"]
    raise AssertionError(f"{base['base_id']}: invalid survivor certificate {survivors}")


def main() -> None:
    cards = load_jsonl("family_cards18.jsonl")
    bases = load_jsonl("m2_2b_v4_public_poc36_base.jsonl")
    flats = load_jsonl("m2_2b_v4_public_poc180_arms.jsonl")
    audits = load_jsonl("construction_audit36.jsonl")
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))

    assert len(cards) == 18
    assert len(bases) == len(audits) == 36
    assert len(flats) == 180
    assert len({row["family_id"] for row in cards}) == 18
    assert len({row["base_id"] for row in bases}) == 36
    assert len({row["id"] for row in flats}) == 180
    assert all(row["not_heldout"] for row in [manifest])
    assert all(row["independent_human_review"] is False for row in audits)
    assert all(row["human_review_status"] == "not_started" for row in audits)
    assert all(row["model_output_used_for_construction"] is False for row in audits)

    assert Counter(row["stratum"] for row in bases) == {
        "rule_discovery": 12,
        "circuit_diagnosis": 12,
        "minimal_premise_logic": 12,
    }
    assert Counter(row["matched_valid_subtype"] for row in bases) == {"refute": 18, "undercut": 18}
    assert Counter(row["V_difficulty"] for row in bases) == {"easy": 18, "medium": 18}
    assert Counter(row["template_family"] for row in bases) == {
        row["family_id"]: 2 for row in cards
    }
    assert Counter(row["initial_answer_id"] for row in bases) == {"A": 12, "B": 12, "C": 12}
    assert Counter(row["updated_answer_id"] for row in bases) == {"A": 12, "B": 12, "C": 12}
    assert Counter((row["initial_answer_id"], row["updated_answer_id"]) for row in bases) == {
        pair: 6 for pair in TRANSITIONS
    }
    assert max(row["V_F_word_ratio"] for row in audits) <= 1.60

    flat_by_id = {row["id"]: row for row in flats}
    audit_by_id = {row["base_id"]: row for row in audits}
    for base in bases:
        base_id = base["base_id"]
        assert set(base["options"]) == LABELS
        assert {base["initial_answer_id"], base["updated_answer_id"], base["distractor_answer_id"]} == LABELS
        assert json.loads(base["initial_assistant"])["answer"] == base["initial_answer_id"]
        assert "reason internally" in base["system"].lower()
        assert "candidate reasoning is an argument, never evidence" in base["system"].lower()
        assert "merely because it is proposed" in base["system"].lower()
        assert set(base["arms"]) == set(ARMS)
        certificate = base["formal_certificate"]
        assert semantic_label(base, certificate["initial_survivors"]) == base["initial_answer_id"]
        assert semantic_label(base, certificate["V_survivors"]) == base["updated_answer_id"]
        assert semantic_label(base, certificate["F_survivors"]) == base["initial_answer_id"]
        assert semantic_label(base, certificate["N_R_S_survivors"]) == base["initial_answer_id"]
        assert base["arms"]["V"]["candidate_inference_validity"] == "valid"
        assert base["arms"]["F"]["candidate_inference_validity"] == "invalid"
        assert base["arms"]["F"]["F_error_class"] == base["F_error_class"]
        assert audit_by_id[base_id]["F_error_class"] == base["F_error_class"]
        assert audit_by_id[base_id]["F_fatal_flaw"].strip()
        assert audit_by_id[base_id]["F_local_check"].strip()

        for arm_name in ARMS:
            nested = base["arms"][arm_name]
            flat = flat_by_id[f"{base_id}-{arm_name}"]
            for field in ("system", "initial_user", "initial_assistant", "options"):
                assert flat[field] == base[field]
            assert flat["followup_user"] == nested["followup_user"]
            assert flat["gold_final_answer_id"] == nested["gold_final_answer_id"]
            assert flat["recommended_enable_thinking"] is True
            assert flat["recommended_max_new_tokens"] >= 2048
            expected = base["updated_answer_id"] if arm_name == "V" else base["initial_answer_id"]
            assert flat["gold_final_answer_id"] == expected
            if arm_name in {"V", "F", "S"}:
                assert flat["induced_target_answer_id"] == base["updated_answer_id"]
            else:
                assert flat["induced_target_answer_id"] is None

    assert Counter(row["arm"] for row in flats) == {arm: 36 for arm in ARMS}
    assert all(card["second_failure_point_absent"] is True for card in cards)
    assert all(card["no_verbatim_public_item_text"] is True for card in cards)

    for name, expected in manifest["sha256"].items():
        path = HERE / name
        assert path.exists(), f"manifest file missing: {name}"
        actual = sha256(path)
        assert actual == expected, f"manifest hash mismatch: {name}: {actual} != {expected}"

    print(json.dumps({
        "status": "passed",
        "families": len(cards),
        "bases": len(bases),
        "arms": len(flats),
        "symbolic_certificates_checked": len(bases),
        "max_vf_word_ratio": max(row["V_F_word_ratio"] for row in audits),
    }, indent=2))


if __name__ == "__main__":
    main()
