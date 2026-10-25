"""Seeded proof-graph construction without an intended-edit output channel."""

from __future__ import annotations

import copy
import random
from typing import Any

from hardening_common import fact, literal, opposite, rule

TRANSITIONS = {
    "PGF-01": "true_to_false",
    "PGF-02": "false_to_true",
    "PGF-03": "true_to_unknown",
    "PGF-04": "unknown_to_true",
    "PGF-05": "false_to_unknown",
    "PGF-06": "unknown_to_false",
    "PGF-07": "true_to_false",
    "PGF-08": "false_to_true",
}

OPERATION_CONTRACTS = {
    "PGF-01": ["fact_addition", "fact_deletion", "fact_replacement"],
    "PGF-02": ["rule_addition", "rule_deletion", "rule_replacement"],
    "PGF-03": ["fact_addition", "fact_deletion"],
    "PGF-04": ["fact_addition", "fact_deletion"],
    "PGF-05": ["rule_addition", "rule_deletion"],
    "PGF-06": ["fact_addition", "fact_deletion"],
    "PGF-07": ["fact_addition", "fact_deletion", "fact_replacement"],
    "PGF-08": ["fact_addition", "fact_deletion", "fact_replacement"],
}

FAMILY_BLUEPRINTS = [
    {
        "family_id": family_id,
        "structure_signature": signature,
        "transition": TRANSITIONS[family_id],
        "allowed_operations": OPERATION_CONTRACTS[family_id],
        "easy_depth": [2, 3],
        "medium_depth": [4, 6],
    }
    for family_id, signature in {
        "PGF-01": "signed seed selects disjoint positive and negative proof graphs",
        "PGF-02": "terminal policy polarity replacement after a shared upstream graph",
        "PGF-03": "one mutable root supports an otherwise immutable positive graph",
        "PGF-04": "a missing conjunct unlocks a positive proof graph",
        "PGF-05": "one mutable terminal policy supports a negative graph",
        "PGF-06": "one missing directed edge completes negative reachability",
        "PGF-07": "two positive paths share one mutable gate opposed by a negative graph",
        "PGF-08": "one local relational record selects opposite global propagation graphs",
    }.items()
]


def _record(record: dict[str, Any], *, mutable: bool, distractor: bool = False) -> dict[str, Any]:
    result = copy.deepcopy(record)
    result["mutable"] = mutable
    result["is_distractor"] = distractor
    return result


def _chain(
    records: list[dict[str, Any]],
    start: dict[str, Any],
    target: dict[str, Any],
    depth: int,
    stem: str,
    entity: str,
    *,
    medium: bool,
) -> None:
    current = start
    for index in range(depth):
        head = target if index == depth - 1 else literal(f"{stem}step{index}", entity)
        role = "state_reconstruction" if medium and index == depth // 2 else "schema"
        records.append(
            _record(rule(f"schema_{stem}_{index}", [current], head, role=role), mutable=False)
        )
        current = head


def _symbols(seed: int) -> tuple[random.Random, str, list[str]]:
    rng = random.Random(seed)
    syllables = ["nav", "kel", "tor", "miv", "lan", "sor", "vek", "pax"]
    rng.shuffle(syllables)
    stem = f"{syllables[0]}{seed % 997}"
    # Prover9/Mace4 reserve common x/y/z/u/v/w-style names as variables. The
    # explicit c-prefix keeps every generated entity unambiguously constant.
    entities = [f"c{syllables[index]}{(seed >> (index * 5)) % 89}" for index in range(1, 6)]
    return rng, stem, entities


def _fallacy_scaffold(
    records: list[dict[str, Any]], error_class: str, stem: str, entities: list[str]
) -> dict[str, Any]:
    a, b = entities[:2]
    prefix = f"{stem}fall"
    if error_class == "cross_entity_transfer":
        witness = literal(f"{prefix}signal", b)
        claim = literal(f"{prefix}signal", a)
        records.append(_record(fact(f"fall_{stem}_0", witness), mutable=True, distractor=True))
        premises = [witness]
    elif error_class == "converse_error":
        antecedent = literal(f"{prefix}source", a)
        witness = literal(f"{prefix}verified", a)
        records.extend(
            [
                _record(fact(f"fall_{stem}_0", witness), mutable=True, distractor=True),
                _record(
                    rule(f"fall_{stem}_1", [antecedent], witness),
                    mutable=False,
                    distractor=True,
                ),
            ]
        )
        claim = antecedent
        premises = [witness]
    elif error_class == "conjunction_weakening":
        left = literal(f"{prefix}left", a)
        right = literal(f"{prefix}right", a)
        claim = literal(f"{prefix}joined", a)
        records.extend(
            [
                _record(fact(f"fall_{stem}_0", left), mutable=True, distractor=True),
                _record(
                    rule(f"fall_{stem}_1", [left, right], claim),
                    mutable=False,
                    distractor=True,
                ),
            ]
        )
        premises = [left]
    elif error_class == "unsupported_bridge":
        witness = literal(f"{prefix}observed", a)
        claim = literal(f"{prefix}bridge", a)
        records.append(_record(fact(f"fall_{stem}_0", witness), mutable=True, distractor=True))
        premises = [witness]
    elif error_class == "scope_transfer":
        witness = literal(f"{prefix}access", a, "remote")
        claim = literal(f"{prefix}access", a, "local")
        records.append(_record(fact(f"fall_{stem}_0", witness), mutable=True, distractor=True))
        premises = [witness]
    elif error_class == "support_path_generalization":
        gate_a = literal(f"{prefix}gate", a)
        supported_a = literal(f"{prefix}supported", a)
        claim = literal(f"{prefix}supported", b)
        records.extend(
            [
                _record(fact(f"fall_{stem}_0", gate_a), mutable=True, distractor=True),
                _record(
                    rule(
                        f"fall_{stem}_1",
                        [literal(f"{prefix}gate", "?x")],
                        literal(f"{prefix}supported", "?x"),
                    ),
                    mutable=False,
                    distractor=True,
                ),
            ]
        )
        premises = [supported_a]
    else:
        raise ValueError(f"unsupported F operator: {error_class}")
    return {"premises": premises, "claimed_conclusion": claim}


def _fill_distractors(
    records: list[dict[str, Any]], requested: int, stem: str, entities: list[str]
) -> None:
    current = sum(bool(record.get("is_distractor")) for record in records)
    if current > requested:
        raise ValueError("fallacy scaffold exceeds distractor contract")
    for index in range(current, requested):
        item = literal(f"{stem}noise{index}", entities[index % len(entities)])
        records.append(
            _record(fact(f"data_{stem}_noise_{index}", item), mutable=True, distractor=True)
        )


def _candidate_literals(
    stem: str, entities: list[str], required: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    candidates = copy.deepcopy(required)
    index = 0
    while len(candidates) < count:
        candidates.append(literal(f"{stem}candidate{index}", entities[index % len(entities)]))
        index += 1
    unique = {str(item): item for item in candidates}
    return [unique[key] for key in sorted(unique)]


def _world(records: list[dict[str, Any]], constants: list[str]) -> dict[str, Any]:
    return {
        "logic": "finite_function_free_horn_explicit_negation",
        "constants": sorted(set(constants + ["local", "remote"])),
        "records": sorted(records, key=lambda item: item["record_id"]),
    }


def proof_graph_builder(
    blueprint: dict[str, Any],
    variant: str,
    seed: int,
    desired_transition: str,
    error_class: str,
) -> dict[str, Any]:
    """Build only an initial world and a route-neutral candidate vocabulary."""

    family_id = blueprint["family_id"]
    if desired_transition != TRANSITIONS[family_id]:
        raise ValueError("desired transition conflicts with the frozen family blueprint")
    if variant not in {"easy", "medium"}:
        raise ValueError("variant must be easy or medium")
    rng, stem, entities = _symbols(seed)
    medium = variant == "medium"
    depth = rng.randint(4, 6) if medium else rng.randint(2, 3)
    distractors = rng.randint(6, 10) if medium else rng.randint(2, 4)
    a = entities[0]
    query = literal(f"{stem}outcome", a)
    records: list[dict[str, Any]] = []
    required_facts: list[dict[str, Any]] = []
    required_rule_bodies: list[list[dict[str, Any]]] = []
    required_rule_heads: list[dict[str, Any]] = []

    if family_id == "PGF-01":
        root = literal(f"{stem}seed", a)
        records.append(_record(fact(f"data_{stem}_root", root), mutable=True))
        _chain(records, root, query, depth, f"{stem}pos", a, medium=medium)
        _chain(records, opposite(root), opposite(query), depth, f"{stem}neg", a, medium=medium)
        required_facts = [opposite(root)]
    elif family_id == "PGF-02":
        source = literal(f"{stem}source", a)
        bridge = literal(f"{stem}bridge", a)
        records.append(_record(fact(f"data_{stem}_source", source), mutable=True))
        _chain(records, source, bridge, depth - 1, f"{stem}up", a, medium=medium)
        records.append(
            _record(
                rule(f"policy_{stem}_terminal", [bridge], opposite(query)),
                mutable=True,
            )
        )
        required_rule_bodies = [
            [bridge],
            [literal(f"{stem}neutralbody0", a)],
            [literal(f"{stem}neutralbody1", a)],
            [literal(f"{stem}neutralbody2", a)],
        ]
        required_rule_heads = [
            query,
            opposite(query),
            literal(f"{stem}neutralhead0", a),
            literal(f"{stem}neutralhead1", a),
        ]
    elif family_id == "PGF-03":
        root = literal(f"{stem}token", a)
        records.append(_record(fact(f"data_{stem}_root", root), mutable=True))
        _chain(records, root, query, depth, f"{stem}proof", a, medium=medium)
    elif family_id == "PGF-04":
        left = literal(f"{stem}left", a)
        right = literal(f"{stem}right", a)
        joint = literal(f"{stem}joint", a)
        records.append(_record(fact(f"data_{stem}_left", left), mutable=True))
        records.append(_record(rule(f"schema_{stem}_join", [left, right], joint), mutable=False))
        _chain(records, joint, query, depth - 1, f"{stem}proof", a, medium=medium)
        required_facts = [right]
    elif family_id == "PGF-05":
        source = literal(f"{stem}trigger", a)
        bridge = literal(f"{stem}bridge", a)
        records.append(_record(fact(f"data_{stem}_source", source), mutable=True))
        _chain(records, source, bridge, depth - 1, f"{stem}up", a, medium=medium)
        records.append(
            _record(
                rule(f"policy_{stem}_terminal", [bridge], opposite(query)),
                mutable=True,
            )
        )
        required_rule_bodies = [[literal(f"{stem}rb{index}", a)] for index in range(4)]
        required_rule_heads = [literal(f"{stem}rh{index}", a) for index in range(5)]
    elif family_id == "PGF-06":
        edge_count = depth - 1
        nodes = entities[:2] + [f"c{stem}node{index}" for index in range(edge_count - 1)]
        nodes = nodes[: edge_count + 1]
        while len(nodes) < edge_count + 1:
            nodes.append(f"c{stem}node{len(nodes)}")
        missing = rng.randrange(edge_count)
        for index in range(edge_count):
            edge = literal(f"{stem}edge", nodes[index], nodes[index + 1])
            if index == missing:
                required_facts = [edge]
            else:
                records.append(_record(fact(f"data_{stem}_edge_{index}", edge), mutable=True))
        records.extend(
            [
                _record(
                    rule(
                        f"schema_{stem}_reach_base",
                        [literal(f"{stem}edge", "?x", "?y")],
                        literal(f"{stem}reach", "?x", "?y"),
                    ),
                    mutable=False,
                ),
                _record(
                    rule(
                        f"schema_{stem}_reach_extend",
                        [
                            literal(f"{stem}reach", "?x", "?y"),
                            literal(f"{stem}edge", "?y", "?z"),
                        ],
                        literal(f"{stem}reach", "?x", "?z"),
                        role="state_reconstruction" if medium else "schema",
                    ),
                    mutable=False,
                ),
                _record(
                    rule(
                        f"schema_{stem}_decide",
                        [literal(f"{stem}reach", nodes[0], nodes[-1])],
                        opposite(query),
                    ),
                    mutable=False,
                ),
            ]
        )
        entities = list(dict.fromkeys(entities + nodes))
    elif family_id == "PGF-07":
        gate = literal(f"{stem}gate", a)
        records.append(_record(fact(f"data_{stem}_gate", gate), mutable=True))
        for branch in range(2):
            support = literal(f"{stem}support{branch}", a)
            start = literal(f"{stem}branch{branch}", a)
            records.append(_record(fact(f"data_{stem}_support_{branch}", support), mutable=True))
            records.append(
                _record(
                    rule(f"schema_{stem}_gate_{branch}", [gate, support], start),
                    mutable=False,
                )
            )
            _chain(records, start, query, depth - 1, f"{stem}path{branch}", a, medium=medium)
        _chain(records, opposite(gate), opposite(query), depth, f"{stem}neg", a, medium=medium)
        required_facts = [opposite(gate)]
    elif family_id == "PGF-08":
        local, bad, good = entities[:3]
        before = literal(f"{stem}link", local, bad)
        after = literal(f"{stem}link", local, good)
        bad_start = literal(f"{stem}badstate", local)
        good_start = literal(f"{stem}goodstate", local)
        records.append(_record(fact(f"data_{stem}_link", before), mutable=True))
        records.extend(
            [
                _record(rule(f"schema_{stem}_bad", [before], bad_start), mutable=False),
                _record(rule(f"schema_{stem}_good", [after], good_start), mutable=False),
            ]
        )
        _chain(records, bad_start, opposite(query), depth - 1, f"{stem}bad", local, medium=medium)
        _chain(records, good_start, query, depth - 1, f"{stem}good", local, medium=medium)
        required_facts = [after]
    else:
        raise ValueError(f"unknown family: {family_id}")

    fallacy_context = _fallacy_scaffold(records, error_class, stem, entities)
    _fill_distractors(records, distractors, stem, entities)
    fact_count = 6 if "fact_replacement" in blueprint["allowed_operations"] else 22
    fact_literals = _candidate_literals(stem, entities, required_facts, fact_count)

    return {
        "architecture": "proof_graph_plus_atomic_edit_certification",
        "family_id": family_id,
        "variant": variant,
        "seed": seed,
        "desired_transition": desired_transition,
        "initial_world": _world(records, entities),
        "query": query,
        "explicit_opposite": opposite(query),
        "candidate_vocabulary": {
            "fact_literals": fact_literals,
            "rule_bodies": required_rule_bodies,
            "rule_heads": required_rule_heads,
        },
        "enumeration_contract": {
            "allowed_operations": copy.deepcopy(blueprint["allowed_operations"]),
            "mutable_field": "mutable",
            "max_candidates": 100,
        },
        "difficulty_contract": {
            "variant": variant,
            "proof_depth": [4, 6] if medium else [2, 3],
            "distractor_count": [6, 10] if medium else [2, 4],
            "generated_target_depth": depth,
            "generated_distractor_count": distractors,
        },
        "fallacy_context": fallacy_context,
    }


def blueprint_by_id(family_id: str) -> dict[str, Any]:
    return copy.deepcopy(next(item for item in FAMILY_BLUEPRINTS if item["family_id"] == family_id))


def assert_no_answer_channel(result: dict[str, Any]) -> None:
    prohibited = {
        "edit",
        "intended_edit",
        "updated_world",
        "successful_rank",
        "correct_record_id",
        "correct_location",
    }
    leaked = prohibited.intersection(result)
    if leaked:
        raise AssertionError(f"builder leaked answer-channel keys: {sorted(leaked)}")
