"""Build the bounded CEDAR proof-generator feasibility comparison.

The two routes share semantics and templates but differ in construction order:
`provergen_style` installs the requested proof/edit skeleton directly, whereas
`reviseqa_style` searches one-record neighbors of the same valid initial world.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from formal_horn import (
    MASTER_SEED,
    SCHEMA_VERSION,
    apply_edit,
    canonical_jsonl_sha256,
    canonical_sha256,
    decisive_literal,
    edit_count,
    evaluate_fixed_point,
    fact,
    file_sha256,
    label_from_evaluation,
    literal,
    literal_key,
    locate_solver,
    opposite,
    proof_count,
    rule,
    run_mace4_countermodel,
    shortest_proof,
    stable_seed,
    validate_with_both,
    write_json,
    write_jsonl,
)

HERE = Path(__file__).resolve().parent
ROUTES = ("provergen_style", "reviseqa_style")
VARIANTS = ("easy", "medium")
MAX_CANDIDATES = 100

SCHEMA_CONTRACT = {
    "schema_version": SCHEMA_VERSION,
    "logic": "finite_function_free_horn_with_explicit_negation_open_world",
    "labels": ["true", "false", "unknown", "invalid"],
    "routes": list(ROUTES),
    "arms": ["N", "R", "V", "F"],
    "v_edit_operations": ["add", "delete", "replace"],
    "master_seed": MASTER_SEED,
    "max_candidates_per_specification": MAX_CANDIDATES,
}

FAMILY_BLUEPRINTS = [
    {
        "family_id": "PGF-01",
        "family_title": "signed_seed_record_replacement",
        "structure_signature": (
            "one signed seed record selects between disjoint positive and negative proof chains"
        ),
        "transition": "true_to_false",
        "easy_depth": [2, 3],
        "medium_depth": [4, 6],
        "allowed_edit_type": "fact_replacement",
        "fallacy_error_class": "cross_entity_transfer",
        "forbidden_shortcuts": ["direct query fact", "second seed for target entity"],
        "development_overlap_notes": (
            "formal signed-Horn proof switch; no arithmetic, label lookup, or V7 CSP reuse"
        ),
    },
    {
        "family_id": "PGF-02",
        "family_title": "terminal_rule_polarity_replacement",
        "structure_signature": (
            "a shared upstream proof reaches one terminal rule whose complete replacement "
            "changes conclusion polarity"
        ),
        "transition": "false_to_true",
        "easy_depth": [2, 3],
        "medium_depth": [4, 6],
        "allowed_edit_type": "rule_replacement",
        "fallacy_error_class": "converse_error",
        "forbidden_shortcuts": ["direct query fact", "parallel terminal rule"],
        "development_overlap_notes": (
            "proof-rule replacement rather than V7 functional-dependency direction or key closure"
        ),
    },
    {
        "family_id": "PGF-03",
        "family_title": "critical_fact_withdrawal",
        "structure_signature": (
            "a unique proof depends on one root fact; withdrawing that record leaves both signs "
            "underived"
        ),
        "transition": "true_to_unknown",
        "easy_depth": [2, 3],
        "medium_depth": [4, 6],
        "allowed_edit_type": "fact_deletion",
        "fallacy_error_class": "unsupported_bridge",
        "forbidden_shortcuts": ["negative query support", "alternate positive root"],
        "development_overlap_notes": (
            "open-world proof withdrawal, not a V7 record correction with a replacement answer"
        ),
    },
    {
        "family_id": "PGF-04",
        "family_title": "missing_conjunct_insertion",
        "structure_signature": (
            "two conjuncts jointly unlock a proof; exactly one missing conjunct is added"
        ),
        "transition": "unknown_to_true",
        "easy_depth": [2, 3],
        "medium_depth": [4, 6],
        "allowed_edit_type": "fact_addition",
        "fallacy_error_class": "conjunction_weakening",
        "forbidden_shortcuts": ["single-conjunct rule", "direct query fact"],
        "development_overlap_notes": (
            "Horn conjunction entailment, distinct from V7 weighted-quorum arithmetic and veto"
        ),
    },
    {
        "family_id": "PGF-05",
        "family_title": "critical_rule_withdrawal",
        "structure_signature": (
            "a unique negative proof is undercut by deleting one complete rule record"
        ),
        "transition": "false_to_unknown",
        "easy_depth": [2, 3],
        "medium_depth": [4, 6],
        "allowed_edit_type": "rule_deletion",
        "fallacy_error_class": "scope_transfer",
        "forbidden_shortcuts": ["positive query support", "alternate negative rule"],
        "development_overlap_notes": (
            "formal rule withdrawal, not V7 rule-direction reversal or dependency replacement"
        ),
    },
    {
        "family_id": "PGF-06",
        "family_title": "missing_directed_edge_completion",
        "structure_signature": (
            "recursive directed reachability becomes complete after one missing local edge is added"
        ),
        "transition": "unknown_to_false",
        "easy_depth": [2, 3],
        "medium_depth": [4, 6],
        "allowed_edit_type": "fact_addition",
        "fallacy_error_class": "converse_error",
        "forbidden_shortcuts": ["reverse-edge equivalence", "alternate complete path"],
        "development_overlap_notes": (
            "generator stress family for recursive Horn depth; not admitted to V7 held-out"
        ),
    },
    {
        "family_id": "PGF-07",
        "family_title": "shared_gate_two_support_paths",
        "structure_signature": (
            "two distinct positive proofs share one gate; replacing the gate disables both and "
            "enables a negative proof"
        ),
        "transition": "true_to_false",
        "easy_depth": [2, 3],
        "medium_depth": [4, 6],
        "allowed_edit_type": "fact_replacement",
        "fallacy_error_class": "support_path_generalization",
        "forbidden_shortcuts": ["ungated positive path", "second negative gate"],
        "development_overlap_notes": (
            "explicit proof multiplicity control, not V7 global matching or path-existence item"
        ),
    },
    {
        "family_id": "PGF-08",
        "family_title": "local_edge_global_status_replacement",
        "structure_signature": (
            "one local relational record feeds disjoint propagation chains to opposite global "
            "query signs"
        ),
        "transition": "false_to_true",
        "easy_depth": [2, 3],
        "medium_depth": [4, 6],
        "allowed_edit_type": "fact_replacement",
        "fallacy_error_class": "scope_transfer",
        "forbidden_shortcuts": ["direct global status fact", "cross-subsystem propagation"],
        "development_overlap_notes": (
            "relational local-to-global Horn propagation, not V7 critical-path max-plus reasoning"
        ),
    },
]

SPEC_PARAMETERS = {
    "PGF-01": {"easy": (2, 3), "medium": (4, 8)},
    "PGF-02": {"easy": (3, 3), "medium": (5, 8)},
    "PGF-03": {"easy": (2, 3), "medium": (4, 8)},
    "PGF-04": {"easy": (2, 3), "medium": (5, 8)},
    "PGF-05": {"easy": (3, 3), "medium": (6, 8)},
    "PGF-06": {"easy": (3, 3), "medium": (5, 8)},
    "PGF-07": {"easy": (2, 3), "medium": (4, 8)},
    "PGF-08": {"easy": (3, 3), "medium": (5, 8)},
}


def common_specs() -> list[dict[str, Any]]:
    blueprints = {item["family_id"]: item for item in FAMILY_BLUEPRINTS}
    specs: list[dict[str, Any]] = []
    for family_id in sorted(blueprints):
        for variant in VARIANTS:
            depth, distractors = SPEC_PARAMETERS[family_id][variant]
            blueprint = blueprints[family_id]
            specs.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "spec_id": f"{family_id}-{variant[0].upper()}",
                    "family_id": family_id,
                    "variant": variant,
                    "structure_signature": blueprint["structure_signature"],
                    "transition": blueprint["transition"],
                    "allowed_edit_type": blueprint["allowed_edit_type"],
                    "fallacy_error_class": blueprint["fallacy_error_class"],
                    "target_proof_depth": depth,
                    "distractor_count": distractors,
                    "state_reconstruction_steps": 0 if variant == "easy" else 1,
                    "candidate_budget": MAX_CANDIDATES,
                    "shared_route_contract": True,
                }
            )
    return specs


def _chain(
    records: list[dict[str, Any]],
    *,
    prefix: str,
    start: dict[str, Any],
    target: dict[str, Any],
    depth: int,
    medium: bool,
) -> None:
    if depth < 1:
        raise ValueError("chain depth must be positive")
    current = start
    for index in range(1, depth + 1):
        head = target if index == depth else literal(f"{prefix}_s{index}", *start["arguments"])
        role = "state_reconstruction" if medium and index == max(1, depth // 2) else "structural"
        records.append(rule(f"r_{prefix}_{index}", [current], head, role=role))
        current = head


def _fill_distractors(
    records: list[dict[str, Any]], constants: set[str], requested: int, prefix: str
) -> None:
    current = sum(record["role"] == "distractor" for record in records)
    if current > requested:
        raise ValueError(f"family-specific fallacy witnesses exceed distractor contract: {current}")
    for index in range(current + 1, requested + 1):
        constant = f"d{index}"
        constants.add(constant)
        if index % 2:
            records.append(
                fact(
                    f"d_{prefix}_{index}",
                    literal(f"{prefix}_noise_{index}", constant),
                    role="distractor",
                )
            )
        else:
            records.append(
                rule(
                    f"d_{prefix}_{index}",
                    [literal(f"{prefix}_missing_{index}", "?x")],
                    literal(f"{prefix}_noise_{index}", "?x"),
                    role="distractor",
                )
            )


def _base_world(records: list[dict[str, Any]], constants: set[str]) -> dict[str, Any]:
    return {
        "logic": "finite_function_free_horn_explicit_negation",
        "constants": sorted(constants),
        "records": sorted(records, key=lambda record: record["record_id"]),
    }


def _template_f01(spec: dict[str, Any]) -> dict[str, Any]:
    medium = spec["variant"] == "medium"
    records: list[dict[str, Any]] = []
    constants = {"a", "b"}
    query = literal("f01_outcome", "a")
    seed = literal("f01_seed", "a")
    critical = fact("c_f01_seed", seed, role="critical")
    records.append(critical)
    _chain(
        records,
        prefix="f01_pos",
        start=seed,
        target=query,
        depth=spec["target_proof_depth"],
        medium=medium,
    )
    _chain(
        records,
        prefix="f01_neg",
        start=opposite(seed),
        target=opposite(query),
        depth=spec["target_proof_depth"],
        medium=medium,
    )
    records.append(
        fact("d_f01_cross_entity", literal("f01_seed", "b", negated=True), role="distractor")
    )
    after = fact("c_f01_seed", opposite(seed), role="critical")
    edit = {
        "operation": "replace",
        "before": critical,
        "after": after,
        "edit_count": 1,
    }
    witness = {
        "kind": "cross_entity_transfer",
        "source": "-f01_seed(b)",
        "unsupported_claim": "-f01_seed(a)",
        "why_invalid": "no equality or transfer rule connects b to a",
    }
    _fill_distractors(records, constants, spec["distractor_count"], "f01")
    return {
        "world": _base_world(records, constants),
        "query": query,
        "edit": edit,
        "witness": witness,
    }


def _template_f02(spec: dict[str, Any]) -> dict[str, Any]:
    medium = spec["variant"] == "medium"
    records: list[dict[str, Any]] = []
    constants = {"a"}
    query = literal("f02_result", "a")
    start = literal("f02_source", "a")
    bridge = literal("f02_bridge", "a")
    records.append(fact("f_f02_source", start))
    _chain(
        records,
        prefix="f02_upstream",
        start=start,
        target=bridge,
        depth=spec["target_proof_depth"] - 1,
        medium=medium,
    )
    before = rule("c_f02_terminal", [bridge], opposite(query), role="critical")
    after = rule("c_f02_terminal", [bridge], query, role="critical")
    records.append(before)
    records.extend(
        [
            fact("d_f02_verified", literal("f02_verified", "a"), role="distractor"),
            rule(
                "d_f02_one_way",
                [literal("f02_positive_rule_active", "?x")],
                literal("f02_verified", "?x"),
                role="distractor",
            ),
        ]
    )
    edit = {"operation": "replace", "before": before, "after": after, "edit_count": 1}
    witness = {
        "kind": "converse_error",
        "source": "f02_verified(a)",
        "rule": "f02_positive_rule_active(x) -> f02_verified(x)",
        "unsupported_claim": "the positive terminal rule is active",
        "why_invalid": "the implication has no licensed reverse direction",
    }
    _fill_distractors(records, constants, spec["distractor_count"], "f02")
    return {
        "world": _base_world(records, constants),
        "query": query,
        "edit": edit,
        "witness": witness,
    }


def _template_f03(spec: dict[str, Any]) -> dict[str, Any]:
    medium = spec["variant"] == "medium"
    records: list[dict[str, Any]] = []
    constants = {"a"}
    query = literal("f03_status", "a")
    source = literal("f03_token", "a")
    critical = fact("c_f03_token", source, role="critical")
    records.append(critical)
    _chain(
        records,
        prefix="f03_proof",
        start=source,
        target=query,
        depth=spec["target_proof_depth"],
        medium=medium,
    )
    records.append(fact("d_f03_reviewed", literal("f03_reviewed", "a"), role="distractor"))
    edit = {"operation": "delete", "before": critical, "after": None, "edit_count": 1}
    witness = {
        "kind": "unsupported_bridge",
        "source": "f03_reviewed(a)",
        "unsupported_claim": "the f03_token(a) record was withdrawn",
        "why_invalid": "review status does not entail record withdrawal",
    }
    _fill_distractors(records, constants, spec["distractor_count"], "f03")
    return {
        "world": _base_world(records, constants),
        "query": query,
        "edit": edit,
        "witness": witness,
    }


def _template_f04(spec: dict[str, Any]) -> dict[str, Any]:
    medium = spec["variant"] == "medium"
    records: list[dict[str, Any]] = []
    constants = {"a"}
    query = literal("f04_eligible", "a")
    left = literal("f04_left", "a")
    right = literal("f04_right", "a")
    bridge = literal("f04_joint", "a")
    records.append(fact("f_f04_left", left))
    records.append(rule("r_f04_conjunction", [left, right], bridge))
    _chain(
        records,
        prefix="f04_proof",
        start=bridge,
        target=query,
        depth=spec["target_proof_depth"] - 1,
        medium=medium,
    )
    added = fact("c_f04_right", right, role="critical")
    edit = {"operation": "add", "before": None, "after": added, "edit_count": 1}
    witness = {
        "kind": "conjunction_weakening",
        "source": "f04_left(a)",
        "required_conjunction": ["f04_left(a)", "f04_right(a)"],
        "unsupported_claim": "f04_right(a)",
        "why_invalid": "one conjunct does not establish the missing conjunct",
    }
    _fill_distractors(records, constants, spec["distractor_count"], "f04")
    return {
        "world": _base_world(records, constants),
        "query": query,
        "edit": edit,
        "witness": witness,
    }


def _template_f05(spec: dict[str, Any]) -> dict[str, Any]:
    medium = spec["variant"] == "medium"
    records: list[dict[str, Any]] = []
    constants = {"a", "b"}
    query = literal("f05_allowed", "a")
    start = literal("f05_trigger", "a")
    bridge = literal("f05_terminal_input", "a")
    records.append(fact("f_f05_trigger", start))
    _chain(
        records,
        prefix="f05_upstream",
        start=start,
        target=bridge,
        depth=spec["target_proof_depth"] - 1,
        medium=medium,
    )
    critical = rule("c_f05_terminal", [bridge], opposite(query), role="critical")
    records.append(critical)
    records.extend(
        [
            fact("d_f05_exception_b", literal("f05_exception", "b"), role="distractor"),
            rule(
                "d_f05_exception_scope",
                [literal("f05_exception", "?x")],
                literal("f05_suspended", "?x"),
                role="distractor",
            ),
        ]
    )
    edit = {"operation": "delete", "before": critical, "after": None, "edit_count": 1}
    witness = {
        "kind": "scope_transfer",
        "source": "f05_exception(b) and f05_suspended(b)",
        "unsupported_claim": "the target-scoped terminal rule for a was withdrawn",
        "why_invalid": "a rule consequence for b does not alter a's rule record",
    }
    _fill_distractors(records, constants, spec["distractor_count"], "f05")
    return {
        "world": _base_world(records, constants),
        "query": query,
        "edit": edit,
        "witness": witness,
    }


def _template_f06(spec: dict[str, Any]) -> dict[str, Any]:
    medium = spec["variant"] == "medium"
    records: list[dict[str, Any]] = []
    edge_count = spec["target_proof_depth"] - 1
    nodes = [f"n{index}" for index in range(edge_count + 1)]
    constants = set(nodes)
    query = literal("f06_clear", nodes[0])
    missing_index = max(0, edge_count // 2)
    critical_literal = literal("f06_edge", nodes[missing_index], nodes[missing_index + 1])
    for index in range(edge_count):
        if index == missing_index:
            continue
        records.append(
            fact(
                f"f_f06_edge_{index}",
                literal("f06_edge", nodes[index], nodes[index + 1]),
            )
        )
    records.extend(
        [
            rule(
                "r_f06_reach_base",
                [literal("f06_edge", "?x", "?y")],
                literal("f06_reach", "?x", "?y"),
            ),
            rule(
                "r_f06_reach_extend",
                [literal("f06_reach", "?x", "?y"), literal("f06_edge", "?y", "?z")],
                literal("f06_reach", "?x", "?z"),
                role="state_reconstruction" if medium else "structural",
            ),
            rule(
                "r_f06_decide",
                [literal("f06_reach", nodes[0], nodes[-1])],
                opposite(query),
            ),
            fact(
                "d_f06_reverse_edge",
                literal("f06_edge", nodes[missing_index + 1], nodes[missing_index]),
                role="distractor",
            ),
        ]
    )
    added = fact("c_f06_missing_edge", critical_literal, role="critical")
    edit = {"operation": "add", "before": None, "after": added, "edit_count": 1}
    witness = {
        "kind": "converse_error",
        "source": literal_key(
            opposite_direction := literal(
                "f06_edge", nodes[missing_index + 1], nodes[missing_index]
            )
        ),
        "unsupported_claim": literal_key(critical_literal),
        "why_invalid": f"directed edge {literal_key(opposite_direction)} has no converse rule",
    }
    _fill_distractors(records, constants, spec["distractor_count"], "f06")
    return {
        "world": _base_world(records, constants),
        "query": query,
        "edit": edit,
        "witness": witness,
    }


def _template_f07(spec: dict[str, Any]) -> dict[str, Any]:
    medium = spec["variant"] == "medium"
    records: list[dict[str, Any]] = []
    constants = {"a", "b"}
    query = literal("f07_decision", "a")
    gate = literal("f07_gate", "a")
    critical = fact("c_f07_gate", gate, role="critical")
    records.extend(
        [
            critical,
            fact("f_f07_support_1", literal("f07_support_1", "a")),
            fact("f_f07_support_2", literal("f07_support_2", "a")),
        ]
    )
    for branch in (1, 2):
        branch_start = literal(f"f07_branch_{branch}", "a")
        records.append(
            rule(
                f"r_f07_gate_{branch}",
                [gate, literal(f"f07_support_{branch}", "a")],
                branch_start,
                role="state_reconstruction" if medium and branch == 1 else "structural",
            )
        )
        _chain(
            records,
            prefix=f"f07_path_{branch}",
            start=branch_start,
            target=query,
            depth=spec["target_proof_depth"] - 1,
            medium=False,
        )
    _chain(
        records,
        prefix="f07_negative",
        start=opposite(gate),
        target=opposite(query),
        depth=spec["target_proof_depth"],
        medium=medium,
    )
    records.extend(
        [
            fact("d_f07_gate_b", literal("f07_gate", "b", negated=True), role="distractor"),
            rule(
                "d_f07_other_path",
                [literal("f07_gate", "b", negated=True)],
                literal("f07_decision", "b", negated=True),
                role="distractor",
            ),
        ]
    )
    after = fact("c_f07_gate", opposite(gate), role="critical")
    edit = {"operation": "replace", "before": critical, "after": after, "edit_count": 1}
    witness = {
        "kind": "support_path_generalization",
        "source": "a valid negative support path for entity b",
        "unsupported_claim": "the gate record for entity a has the negative sign",
        "why_invalid": "a proof path grounded at b cannot be transferred to a",
    }
    _fill_distractors(records, constants, spec["distractor_count"], "f07")
    return {
        "world": _base_world(records, constants),
        "query": query,
        "edit": edit,
        "witness": witness,
    }


def _template_f08(spec: dict[str, Any]) -> dict[str, Any]:
    medium = spec["variant"] == "medium"
    records: list[dict[str, Any]] = []
    constants = {"local", "other", "bad", "good", "sys"}
    query = literal("f08_global_ok", "sys")
    before_literal = literal("f08_link", "local", "bad")
    after_literal = literal("f08_link", "local", "good")
    critical = fact("c_f08_local_link", before_literal, role="critical")
    records.append(critical)
    bad_start = literal("f08_bad_local", "local")
    good_start = literal("f08_good_local", "local")
    records.extend(
        [
            rule("r_f08_bad_decode", [before_literal], bad_start),
            rule("r_f08_good_decode", [after_literal], good_start),
        ]
    )
    _chain(
        records,
        prefix="f08_bad_global",
        start=bad_start,
        target=opposite(query),
        depth=spec["target_proof_depth"] - 1,
        medium=medium,
    )
    _chain(
        records,
        prefix="f08_good_global",
        start=good_start,
        target=query,
        depth=spec["target_proof_depth"] - 1,
        medium=medium,
    )
    other_link = literal("f08_link", "other", "good")
    records.extend(
        [
            fact("d_f08_other_good", other_link, role="distractor"),
            rule(
                "d_f08_other_decode",
                [other_link],
                literal("f08_good_local", "other"),
                role="distractor",
            ),
        ]
    )
    after = fact("c_f08_local_link", after_literal, role="critical")
    edit = {"operation": "replace", "before": critical, "after": after, "edit_count": 1}
    witness = {
        "kind": "scope_transfer",
        "source": "f08_link(other,good)",
        "unsupported_claim": "the local subsystem record is f08_link(local,good)",
        "why_invalid": "the other subsystem's record does not update the local subsystem",
    }
    _fill_distractors(records, constants, spec["distractor_count"], "f08")
    return {
        "world": _base_world(records, constants),
        "query": query,
        "edit": edit,
        "witness": witness,
    }


TEMPLATE_BUILDERS = {
    "PGF-01": _template_f01,
    "PGF-02": _template_f02,
    "PGF-03": _template_f03,
    "PGF-04": _template_f04,
    "PGF-05": _template_f05,
    "PGF-06": _template_f06,
    "PGF-07": _template_f07,
    "PGF-08": _template_f08,
}


def _expected_labels(transition: str) -> tuple[str, str]:
    before, after = transition.split("_to_", maxsplit=1)
    return before, after


def _wrong_edit_candidates(template: dict[str, Any]) -> list[dict[str, Any]]:
    world = template["world"]
    candidates: list[dict[str, Any]] = []
    distractors = [record for record in world["records"] if record["role"] == "distractor"]
    for record in distractors[:3]:
        candidates.append({"operation": "delete", "before": record, "after": None, "edit_count": 1})
    candidates.append(
        {
            "operation": "add",
            "before": None,
            "after": fact("candidate_irrelevant_add", literal("irrelevant_candidate", "a")),
            "edit_count": 1,
        }
    )
    if distractors:
        replacement = copy.deepcopy(distractors[0])
        if replacement["kind"] == "fact":
            replacement["literal"]["predicate"] += "_changed"
        else:
            replacement["head"]["predicate"] += "_changed"
        candidates.append(
            {
                "operation": "replace",
                "before": distractors[0],
                "after": replacement,
                "edit_count": 1,
            }
        )
    return candidates


def _select_edit(
    route: str, spec: dict[str, Any], template: dict[str, Any], seed: int
) -> tuple[dict[str, Any] | None, int, list[dict[str, Any]]]:
    intended = template["edit"]
    if route == "provergen_style":
        return (
            intended,
            1,
            [{"attempt": 1, "strategy": "proof_graph_realization", "accepted": True}],
        )

    candidates = _wrong_edit_candidates(template) + [intended]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    expected_initial, expected_updated = _expected_labels(spec["transition"])
    initial_evaluation = evaluate_fixed_point(template["world"])
    initial_label = label_from_evaluation(initial_evaluation, template["query"])
    trace: list[dict[str, Any]] = []
    if initial_label != expected_initial:
        return None, 0, [{"attempt": 0, "failure": "initial_world_label_mismatch"}]
    for index, candidate in enumerate(candidates[:MAX_CANDIDATES], start=1):
        try:
            updated_world = apply_edit(template["world"], candidate)
            updated_evaluation = evaluate_fixed_point(updated_world)
            updated_label = label_from_evaluation(updated_evaluation, template["query"])
            accepted = updated_label == expected_updated
            trace.append(
                {
                    "attempt": index,
                    "strategy": "atomic_neighbor_search",
                    "operation": candidate["operation"],
                    "updated_label": updated_label,
                    "accepted": accepted,
                }
            )
            if accepted:
                return candidate, index, trace
        except (KeyError, ValueError) as error:
            trace.append(
                {
                    "attempt": index,
                    "strategy": "atomic_neighbor_search",
                    "accepted": False,
                    "failure": type(error).__name__,
                }
            )
    return None, min(len(candidates), MAX_CANDIDATES), trace


def _elapsed_bucket(seconds: float) -> str:
    if seconds < 1.0:
        return "<1s"
    if seconds < 5.0:
        return "1-5s"
    return ">=5s"


def _externalize_prover_trace(
    output_root: Path,
    route: str,
    spec_id: str,
    state: str,
    polarity: str,
    result: dict[str, Any],
) -> str:
    relative = Path("solver_traces") / f"{route}__{spec_id}__{state}__{polarity}.json"
    write_json(output_root / relative, result)
    return relative.as_posix()


def _validation_summary(
    output_root: Path,
    route: str,
    spec_id: str,
    state: str,
    validation: dict[str, Any],
    query: dict[str, Any],
) -> dict[str, Any]:
    query_result = validation["prover9"]["query_result"]
    opposite_result = validation["prover9"]["opposite_result"]
    query_trace = _externalize_prover_trace(
        output_root, route, spec_id, state, "query", query_result
    )
    opposite_trace = _externalize_prover_trace(
        output_root, route, spec_id, state, "opposite", opposite_result
    )
    decisive = decisive_literal(query, validation["python_label"])
    return {
        "python_label": validation["python_label"],
        "prover9_label": validation["prover9_label"],
        "agreement": validation["agreement"],
        "closure_sha256": canonical_sha256(validation["fixed_point"]["closure"]),
        "decisive_proof": (
            shortest_proof(validation["fixed_point"], decisive) if decisive else None
        ),
        "decisive_proof_count": (
            proof_count(validation["fixed_point"], decisive) if decisive else 0
        ),
        "query_trace_file": query_trace,
        "opposite_trace_file": opposite_trace,
        "query_input_sha256": query_result["input_sha256"],
        "opposite_input_sha256": opposite_result["input_sha256"],
    }


def _counterevidence(
    output_root: Path,
    route: str,
    spec: dict[str, Any],
    world: dict[str, Any],
    query: dict[str, Any],
    initial_label: str,
    initial_evaluation: dict[str, Any],
    updated_label: str,
) -> dict[str, Any]:
    target_literal = decisive_literal(query, updated_label)
    if initial_label != "unknown":
        initial_literal = decisive_literal(query, initial_label)
        proof = shortest_proof(initial_evaluation, initial_literal)
        return {
            "type": "opposite_proof",
            "proved_literal": literal_key(initial_literal),
            "proof": proof,
            "refutes_target_label": updated_label,
        }
    if target_literal is None:
        raise ValueError("unknown-to-unknown is not a permitted transition")
    model = run_mace4_countermodel(world, target_literal)
    relative = Path("countermodels") / f"{route}__{spec['spec_id']}__f_countermodel.json"
    write_json(output_root / relative, model)
    return {
        "type": "mace4_countermodel",
        "target_literal": literal_key(target_literal),
        "model_found": model["model_found"],
        "countermodel_file": relative.as_posix(),
    }


def _difficulty_summary(
    spec: dict[str, Any],
    initial_world: dict[str, Any],
    updated_world: dict[str, Any],
    query: dict[str, Any],
    initial_validation: dict[str, Any],
    updated_validation: dict[str, Any],
) -> dict[str, Any]:
    initial_label = initial_validation["python_label"]
    decisive_state = "initial" if initial_label != "unknown" else "updated"
    validation = initial_validation if decisive_state == "initial" else updated_validation
    world = initial_world if decisive_state == "initial" else updated_world
    decisive = decisive_literal(query, validation["python_label"])
    proof = shortest_proof(validation["fixed_point"], decisive)
    if proof is None:
        raise ValueError("decisive state lacks a proof")
    by_id = {record["record_id"]: record for record in world["records"]}
    reconstruction = sum(
        by_id[record_id]["role"] == "state_reconstruction" for record_id in proof["record_ids"]
    )
    return {
        "contract": spec["variant"],
        "decisive_state": decisive_state,
        "actual_shortest_proof_depth": proof["depth"],
        "target_proof_depth": spec["target_proof_depth"],
        "distractor_count": sum(
            record["role"] == "distractor" for record in initial_world["records"]
        ),
        "state_reconstruction_steps": reconstruction,
        "proof_path_count": proof_count(validation["fixed_point"], decisive),
    }


def _arms(
    spec: dict[str, Any],
    initial_label: str,
    updated_label: str,
    edit: dict[str, Any],
    witness: dict[str, Any],
    counterevidence: dict[str, Any],
) -> dict[str, Any]:
    target = {"label": updated_label, "literal": None}
    fatal_claim = "the complete V record edit already holds"
    return {
        "N": {
            "world_delta": [],
            "instruction": "Continue with the recorded case.",
            "gold": initial_label,
        },
        "R": {
            "world_delta": [],
            "instruction": "Re-evaluate the original query from the existing records.",
            "gold": initial_label,
            "mentions_target": False,
            "contains_evidence_or_social_cue": False,
        },
        "V": {
            "world_delta": [edit],
            "gold": updated_label,
            "target": target,
            "atomic": edit_count(edit) == 1,
        },
        "F": {
            "world_delta": [],
            "gold": initial_label,
            "induced_target": target,
            "candidate_reasoning_steps": [
                {
                    "step_id": 1,
                    "kind": "observed_witness",
                    "valid": True,
                    "content": witness["source"],
                },
                {
                    "step_id": 2,
                    "kind": "unsupported_atomic_state_claim",
                    "valid": False,
                    "content": fatal_claim,
                    "claimed_edit": edit,
                },
                {
                    "step_id": 3,
                    "kind": "recompute_from_claimed_state",
                    "valid": True,
                    "content": f"the claimed state has label {updated_label}",
                },
            ],
            "fatal_error_step": 2,
            "error_class": spec["fallacy_error_class"],
            "fallacy_witness": witness,
            "countermodel_or_opposite_proof": counterevidence,
            "repair_witness": {
                "repair": "supply exactly the V edit as an authenticated record change",
                "edit": edit,
                "repaired_label": updated_label,
                "target_argument_succeeds": True,
            },
            "second_failure_point_absent": True,
            "contains_social_cue": False,
        },
    }


def construct_record(
    route: str,
    spec: dict[str, Any],
    output_root: Path,
    *,
    sentinel: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed = stable_seed(MASTER_SEED, route, spec["family_id"], spec["variant"])
    template = TEMPLATE_BUILDERS[spec["family_id"]](spec)
    edit, attempts, construction_trace = _select_edit(route, spec, template, seed)
    base = {
        "schema_version": SCHEMA_VERSION,
        "route": route,
        "spec_id": spec["spec_id"],
        "family_id": spec["family_id"],
        "variant": spec["variant"],
        "seed": seed,
        "candidate_budget": MAX_CANDIDATES,
        "construction_attempts": attempts,
        "construction_trace": construction_trace,
        "sentinel": sentinel,
    }
    if edit is None:
        return {
            **base,
            "status": "failed",
            "failure_stage": "atomic_neighbor_search",
            "failure_reason": "no candidate satisfied the common transition",
            "wall_clock_time_bucket": _elapsed_bucket(time.perf_counter() - started),
        }

    initial_world = template["world"]
    updated_world = apply_edit(initial_world, edit)
    initial_validation = validate_with_both(initial_world, template["query"])
    updated_validation = validate_with_both(updated_world, template["query"])
    expected_initial, expected_updated = _expected_labels(spec["transition"])
    checks = {
        "dual_initial_agreement": initial_validation["agreement"],
        "dual_updated_agreement": updated_validation["agreement"],
        "initial_label_matches": initial_validation["python_label"] == expected_initial,
        "updated_label_matches": updated_validation["python_label"] == expected_updated,
        "initial_consistent": initial_validation["python_label"] != "invalid",
        "updated_consistent": updated_validation["python_label"] != "invalid",
        "v_atomic": edit_count(edit) == 1,
    }
    if not all(checks.values()):
        return {
            **base,
            "status": "failed",
            "failure_stage": "dual_validation",
            "failure_reason": json.dumps(checks, sort_keys=True),
            "wall_clock_time_bucket": _elapsed_bucket(time.perf_counter() - started),
        }

    without_edit_evaluation = evaluate_fixed_point(initial_world)
    updated_target = decisive_literal(template["query"], expected_updated)
    if updated_target is not None:
        target_without_edit = literal_key(updated_target) in without_edit_evaluation["closure"]
        checks["updated_target_requires_edit"] = not target_without_edit
    else:
        initial_target = decisive_literal(template["query"], expected_initial)
        initial_proof = shortest_proof(without_edit_evaluation, initial_target)
        checks["updated_target_requires_edit"] = initial_proof is not None
    if not checks["updated_target_requires_edit"]:
        return {
            **base,
            "status": "failed",
            "failure_stage": "v_necessity",
            "failure_reason": "updated target survives removal of the V edit",
            "wall_clock_time_bucket": _elapsed_bucket(time.perf_counter() - started),
        }

    counterevidence = _counterevidence(
        output_root,
        route,
        spec,
        initial_world,
        template["query"],
        expected_initial,
        initial_validation["fixed_point"],
        expected_updated,
    )
    initial_summary = _validation_summary(
        output_root,
        route,
        spec["spec_id"],
        "initial",
        initial_validation,
        template["query"],
    )
    updated_summary = _validation_summary(
        output_root,
        route,
        spec["spec_id"],
        "updated",
        updated_validation,
        template["query"],
    )
    difficulty = _difficulty_summary(
        spec,
        initial_world,
        updated_world,
        template["query"],
        initial_validation,
        updated_validation,
    )
    arms = _arms(
        spec,
        expected_initial,
        expected_updated,
        edit,
        template["witness"],
        counterevidence,
    )
    target_literal = decisive_literal(template["query"], expected_updated)
    target = {
        "label": expected_updated,
        "literal": literal_key(target_literal) if target_literal else None,
    }
    arms["V"]["target"] = target
    arms["F"]["induced_target"] = copy.deepcopy(target)
    return {
        **base,
        "status": "valid",
        "failure_stage": None,
        "failure_reason": None,
        "structure_signature": spec["structure_signature"],
        "transition": spec["transition"],
        "symbolic_world_initial": initial_world,
        "query": template["query"],
        "explicit_opposite": opposite(template["query"]),
        "initial_label": expected_initial,
        "v_edit": edit,
        "symbolic_world_updated": updated_world,
        "updated_label": expected_updated,
        "updated_target": target,
        "arms": arms,
        "difficulty": difficulty,
        "solver_validation": {"initial": initial_summary, "updated": updated_summary},
        "hard_checks": checks,
        "wall_clock_time_bucket": _elapsed_bucket(time.perf_counter() - started),
    }


def sentinel_specs() -> list[dict[str, Any]]:
    return [
        {
            **common_specs()[0],
            "spec_id": "SENTINEL-FACT-REPLACEMENT",
            "variant": "easy",
            "target_proof_depth": 2,
            "distractor_count": 3,
        },
        {
            **next(spec for spec in common_specs() if spec["family_id"] == "PGF-05"),
            "spec_id": "SENTINEL-RULE-WITHDRAWAL",
            "variant": "easy",
            "target_proof_depth": 3,
            "distractor_count": 3,
        },
    ]


def _atom_english(item: dict[str, Any]) -> str:
    predicate = item["predicate"].replace("_", " ")
    arguments = ", ".join(item["arguments"])
    prefix = "does not have" if item["negated"] else "has"
    return f"{arguments} {prefix} property '{predicate}'"


def _record_english(record: dict[str, Any]) -> str:
    if record["kind"] == "fact":
        return _atom_english(record["literal"]) + "."
    body = " and ".join(_atom_english(item) for item in record["body"])
    head = _atom_english(record["head"])
    return f"If {body}, then {head}."


def _edit_english(edit: dict[str, Any]) -> str:
    if edit["operation"] == "add":
        return "Add the record: " + _record_english(edit["after"])
    if edit["operation"] == "delete":
        return "Withdraw the record: " + _record_english(edit["before"])
    return (
        "Replace the complete record '"
        + _record_english(edit["before"])
        + "' with '"
        + _record_english(edit["after"])
        + "'."
    )


def naturalize(record: dict[str, Any]) -> dict[str, Any]:
    context = " ".join(
        _record_english(item) for item in record["symbolic_world_initial"]["records"]
    )
    query = _atom_english(record["query"])
    witness = record["arms"]["F"]["fallacy_witness"]["source"]
    return {
        "schema_version": SCHEMA_VERSION,
        "route": record["route"],
        "spec_id": record["spec_id"],
        "variant": record["variant"],
        "initial_context": context,
        "question": f"Is the following established: {query}?",
        "answer_options": ["true", "false", "unknown"],
        "v_observation": _edit_english(record["v_edit"]),
        "f_candidate_argument": (
            f"The review points to {witness}. It therefore treats the following change as "
            f"established: {_edit_english(record['v_edit'])} Recompute the query on that basis."
        ),
        "gold_hidden_from_text": True,
        "reverse_map": {
            "initial_world_sha256": canonical_sha256(record["symbolic_world_initial"]),
            "v_edit_sha256": canonical_sha256(record["v_edit"]),
            "f_error_class": record["arms"]["F"]["error_class"],
            "fatal_error_step": record["arms"]["F"]["fatal_error_step"],
        },
    }


def _solver_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "-h"], capture_output=True, text=True, timeout=10, check=False
    )
    match = re.search(r"Prover9 \(32\) version ([^,\n]+)", completed.stdout + completed.stderr)
    return match.group(1) if match else "unknown"


def build(output_root: Path, *, sentinel_only: bool = False) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "family_blueprints8.jsonl", FAMILY_BLUEPRINTS)
    specs = common_specs()
    write_jsonl(output_root / "common_symbolic_specs16.jsonl", specs)

    sentinels: list[dict[str, Any]] = []
    for route in ROUTES:
        for spec in sentinel_specs():
            sentinels.append(construct_record(route, spec, output_root, sentinel=True))
    write_jsonl(output_root / "sentinel_results4.jsonl", sentinels)
    if any(record["status"] != "valid" for record in sentinels):
        raise RuntimeError("sentinel hard gate failed; formal route generation is prohibited")

    builder_path = Path(__file__).resolve()
    validator_path = builder_path.with_name("validate_prover_generator_feasibility.py")
    prover9 = locate_solver("prover9")
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "family_card_canonical_sha256": canonical_jsonl_sha256(
            output_root / "family_blueprints8.jsonl"
        ),
        "schema_canonical_sha256": canonical_sha256(SCHEMA_CONTRACT),
        "builder_sha256": file_sha256(builder_path),
        "validator_sha256": file_sha256(validator_path),
        "solver": {
            "name": "Prover9",
            "version": _solver_version(prover9),
            "executable_sha256": file_sha256(prover9),
            "location": "external_not_vendored",
        },
        "sentinel_canonical_sha256": canonical_jsonl_sha256(
            output_root / "sentinel_results4.jsonl"
        ),
        "sentinel_gate_passed": True,
    }
    write_json(output_root / "sentinel_freeze.json", freeze)

    if sentinel_only:
        return {
            "sentinels_valid": sum(row["status"] == "valid" for row in sentinels),
            "formal_generation_started": False,
        }

    route_rows: dict[str, list[dict[str, Any]]] = {}
    for route in ROUTES:
        rows = [construct_record(route, spec, output_root) for spec in specs]
        route_rows[route] = rows
        write_jsonl(output_root / f"{route}16.jsonl", rows)

    selected_ids = {"PGF-01-E", "PGF-01-M", "PGF-05-E", "PGF-05-M"}
    naturalization = [
        naturalize(record)
        for route in ROUTES
        for record in route_rows[route]
        if record["spec_id"] in selected_ids and record["status"] == "valid"
    ]
    write_jsonl(output_root / "naturalization_sentinel8.jsonl", naturalization)

    return {
        "sentinels_valid": sum(row["status"] == "valid" for row in sentinels),
        "route_valid": {
            route: sum(row["status"] == "valid" for row in rows)
            for route, rows in route_rows.items()
        },
        "attempts": {
            route: Counter(row["construction_attempts"] for row in rows)
            for route, rows in route_rows.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--sentinel-only", action="store_true")
    args = parser.parse_args()
    summary = build(args.output_dir.resolve(), sentinel_only=args.sentinel_only)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
