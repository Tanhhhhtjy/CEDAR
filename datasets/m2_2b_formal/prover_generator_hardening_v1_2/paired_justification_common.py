# ruff: noqa: E501
"""Shared formal and rendering helpers for v1.2 paired-justification sentinels."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V1_1 = HERE.parent / "prover_generator_hardening_v1_1"
if str(V1_1) not in sys.path:
    sys.path.insert(0, str(V1_1))

from hardening_common import (  # noqa: E402,F401
    ERROR_CLASSES,
    apply_edit,
    canonical_json,
    canonical_json_file_sha256,
    canonical_jsonl_sha256,
    canonical_sha256,
    evaluate_fixed_point,
    fact,
    file_sha256,
    literal,
    literal_key,
    proof_count,
    read_jsonl,
    rule,
    run_mace4_countermodel,
    shortest_proof,
    stable_seed,
    validate_with_both,
    write_json,
    write_jsonl,
)
from naturalize_prover_generator_hardening_v1_1 import (  # noqa: E402
    _mapping,
    _record_text,
    _render,
)
from semantic_f_operators import derive_error_class  # noqa: E402

SCHEMA = "cedar.prover_generator_hardening.paired_justification.v1_2"
MASTER_SEED = 20260727
SOURCE_COMMIT = "720d5b401045eb28ad9bd7cae5c23e160c626155"
REVIEWER = "Reviewer Morgan"
SOURCE_TYPE = "authenticated_case_review"
AUTHENTICATION_STATUS = "authenticated"
SPEECH_ACT = "propose_atomic_record_change_from_candidate_argument"
FORBIDDEN_PHRASES = (
    "official record directly specifies",
    "direct entry",
    "participant merely proposes",
    "inference may be wrong",
    "no new information",
)

BASE_PROMPT_COREFERENCE_REPAIRS = {
    11: {
        "english_base_prompt": (
            "If an item has status Cobalt 2, then an item has status Jade 3.",
            "If an item has status Cobalt 2, then that same item has status Jade 3.",
        ),
        "chinese_base_prompt": (
            "如果某个对象具有钴蓝2状态，那么某个对象具有碧玉3状态。",
            "如果某个对象具有钴蓝2状态，那么该同一对象具有碧玉3状态。",
        ),
    },
    12: {
        "english_base_prompt": (
            "If an item has status Amber 1, then an item has status Cobalt 2.",
            "If an item has status Amber 1, then that same item has status Cobalt 2.",
        ),
        "chinese_base_prompt": (
            "如果某个对象具有琥珀1状态，那么某个对象具有钴蓝2状态。",
            "如果某个对象具有琥珀1状态，那么该同一对象具有钴蓝2状态。",
        ),
    },
}


def stable_text_sha256(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def select_source_rows() -> list[dict[str, Any]]:
    """Select the first frozen v1.1 source in each class/difficulty cell."""

    rows = read_jsonl(V1_1 / "hybrid_stress48.jsonl")
    selected: list[dict[str, Any]] = []
    for error_class in ERROR_CLASSES:
        for variant in ("easy", "medium"):
            matches = [
                row
                for row in rows
                if row["assigned_f_error_class"] == error_class
                and row["variant"] == variant
                and row["status"] == "valid"
            ]
            if not matches:
                raise ValueError(f"missing frozen source for {error_class}/{variant}")
            selected.append(sorted(matches, key=lambda row: row["attempt_id"])[0])
    return selected


def _world(constants: set[str], records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "logic": "finite_function_free_horn_explicit_negation",
        "constants": sorted(constants),
        "records": records,
    }


def _post_chain(
    claim: dict[str, Any], edit_id: str, variant: str, prefix: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    count = 1 if variant == "easy" else 3
    records: list[dict[str, Any]] = []
    literals: list[dict[str, Any]] = []
    previous = claim
    for index in range(1, count + 1):
        current = literal(f"{prefix}reviewstage{index}", edit_id)
        records.append(rule(f"{prefix}_downstream_{index}", [previous], current, role="justification"))
        literals.append(current)
        previous = current
    warrant = literal("revision_warranted", edit_id)
    records.append(rule(f"{prefix}_warrant", [previous], warrant, role="justification"))
    literals.append(warrant)
    return records, literals


def _base_formal(
    error_class: str, variant: str, edit_id: str, arm: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Build the class-defining facts/rules and return world, premises, and claim."""

    prefix = f"{arm}_{error_class[:4]}_{variant[0]}"
    r = "crecordr"
    s = "crecords"
    context = literal("review_context_open", edit_id)
    records = [fact(f"{prefix}_context", context, role="observation")]

    if error_class == "cross_entity_transfer":
        subject = r if arm == "v" else s
        signal = literal("origin_check_passed", subject)
        verified = literal("origin_check_verified", subject)
        claim = literal("origin_check_verified", r)
        records.extend(
            [
                fact(f"{prefix}_signal", signal, role="observation"),
                rule(f"{prefix}_verify", [signal], verified, role="justification"),
            ]
        )
        premises = [verified, context]
    elif error_class == "converse_error":
        source_tag = literal("validated_source_tag", r)
        checksum = literal("confirmed_checksum", r)
        records.append(
            rule(f"{prefix}_forward", [source_tag], checksum, role="justification")
        )
        if arm == "v":
            records.append(fact(f"{prefix}_observed", source_tag, role="observation"))
            premises = [source_tag, context]
            claim = checksum
        else:
            records.append(fact(f"{prefix}_observed", checksum, role="observation"))
            premises = [checksum, context]
            claim = source_tag
    elif error_class == "conjunction_weakening":
        identity = literal("identity_check_passed", r)
        timestamp = literal("timestamp_check_passed", r)
        location = literal("location_check_passed", r)
        complete = literal("dual_check_complete", r)
        records.extend(
            [
                fact(f"{prefix}_identity", identity, role="observation"),
                fact(
                    f"{prefix}_second",
                    timestamp if arm == "v" else location,
                    role="observation",
                ),
                rule(
                    f"{prefix}_conjunction",
                    [identity, timestamp],
                    complete,
                    role="justification",
                ),
            ]
        )
        premises = [identity, timestamp if arm == "v" else location]
        claim = complete
    elif error_class == "unsupported_bridge":
        scan = literal("scan_matches_file", r)
        bridge = literal("change_basis_confirmed", r)
        unrelated = literal("retention_status_confirmed", r)
        records.extend(
            [
                fact(f"{prefix}_scan", scan, role="observation"),
                rule(
                    f"{prefix}_bridge",
                    [scan],
                    bridge if arm == "v" else unrelated,
                    role="justification",
                ),
            ]
        )
        premises = [scan, context]
        claim = bridge
    elif error_class == "scope_transfer":
        target = "ctargetr" if arm == "v" else "ctargets"
        observed = literal("route_verified", "csourcenode", target)
        claim = literal("route_verified", "csourcenode", "ctargetr")
        records.append(fact(f"{prefix}_route", observed, role="observation"))
        premises = [observed, context]
    elif error_class == "support_path_generalization":
        subject = r if arm == "v" else s
        gate = literal("access_gate_passed", subject)
        supported = literal("usable_support_path", subject)
        claim = literal("usable_support_path", r)
        records.extend(
            [
                fact(f"{prefix}_gate", gate, role="observation"),
                rule(
                    f"{prefix}_path_rule",
                    [literal("access_gate_passed", "?item")],
                    literal("usable_support_path", "?item"),
                    role="justification",
                ),
            ]
        )
        premises = [supported, context]
    else:  # pragma: no cover - protected by frozen class list
        raise ValueError(error_class)

    downstream, _ = _post_chain(claim, edit_id, variant, prefix)
    records.extend(downstream)
    constants = {edit_id, r, s, "csourcenode", "ctargetr", "ctargets"}
    return _world(constants, records), premises, claim


def build_formal_pair(error_class: str, variant: str, edit_id: str) -> dict[str, Any]:
    v_world, v_premises, v_claim = _base_formal(error_class, variant, edit_id, "v")
    f_world, f_premises, f_claim = _base_formal(error_class, variant, edit_id, "f")
    warrant = literal("revision_warranted", edit_id)
    v_dual = validate_with_both(v_world, warrant)
    f_dual = validate_with_both(f_world, warrant)
    v_proof = shortest_proof(v_dual["fixed_point"], warrant)
    countermodel = run_mace4_countermodel(f_world, warrant)
    derived = derive_error_class(f_world, f_premises, f_claim)

    assumed_world = copy.deepcopy(f_world)
    assumed_world["records"].append(
        fact("temporary_invalid_conclusion", f_claim, role="fallacy_assumption")
    )
    assumed = evaluate_fixed_point(assumed_world)
    downstream_literals = [
        record["head"]
        for record in f_world["records"]
        if record["kind"] == "rule" and record["record_id"].startswith("f_")
        and ("downstream" in record["record_id"] or record["record_id"].endswith("_warrant"))
    ]
    f_evaluation = f_dual["fixed_point"]
    certificate = {
        "declared_error_class": error_class,
        "derived_error_class": derived,
        "entailed_premises": copy.deepcopy(f_premises),
        "invalid_inference": {
            "premises": copy.deepcopy(f_premises),
            "claimed_conclusion": copy.deepcopy(f_claim),
        },
        "fatal_error_count": 1,
        "claim_not_entailed": literal_key(f_claim) not in f_evaluation["closure"],
        "downstream_literals": copy.deepcopy(downstream_literals),
        "downstream_valid_under_single_assumption": all(
            literal_key(item) in assumed["closure"] for item in downstream_literals
        ),
        "warrant_absent_without_invalid_step": literal_key(warrant) not in f_evaluation["closure"],
        "second_failure_point_absent": True,
        "countermodel": countermodel,
    }
    return {
        "query": warrant,
        "v": {
            "world": v_world,
            "premises": v_premises,
            "claim": v_claim,
            "python_label": v_dual["python_label"],
            "prover9_label": v_dual["prover9_label"],
            "agreement": v_dual["agreement"],
            "proof": v_proof,
            "proof_count": proof_count(v_dual["fixed_point"], warrant),
            "solver": v_dual["prover9"],
        },
        "f": {
            "world": f_world,
            "premises": f_premises,
            "claim": f_claim,
            "python_label": f_dual["python_label"],
            "prover9_label": f_dual["prover9_label"],
            "agreement": f_dual["agreement"],
            "solver": f_dual["prover9"],
            "certificate": certificate,
        },
    }


def narrative_pair(error_class: str) -> dict[str, dict[str, list[str]]]:
    """Return tightly matched English and Chinese observations and fatal-step content."""

    narratives: dict[str, dict[str, dict[str, list[str]]]] = {
        "cross_entity_transfer": {
            "en": {
                "v_observations": [
                    "Record R passed the origin check.",
                    "Record R is the record addressed by the proposed change.",
                ],
                "f_observations": [
                    "Record S passed the origin check.",
                    "Record R is the record addressed by the proposed change.",
                ],
                "v_claim": "The origin-check result therefore applies to Record R.",
                "f_claim": "The origin-check result therefore applies to Record R.",
            },
            "zh": {
                "v_observations": ["记录R通过了来源核验。", "拟议修改针对的是记录R。"],
                "f_observations": ["记录S通过了来源核验。", "拟议修改针对的是记录R。"],
                "v_claim": "因此，该来源核验结果适用于记录R。",
                "f_claim": "因此，该来源核验结果适用于记录R。",
            },
        },
        "converse_error": {
            "en": {
                "v_observations": [
                    "Record R has a validated source tag.",
                    "If an item has a validated source tag, that same item has a confirmed checksum.",
                ],
                "f_observations": [
                    "Record R has a confirmed checksum.",
                    "If an item has a validated source tag, that same item has a confirmed checksum.",
                ],
                "v_claim": "Record R therefore has a confirmed checksum.",
                "f_claim": "Record R therefore has a validated source tag.",
            },
            "zh": {
                "v_observations": ["记录R带有已核验的来源标签。", "若某条记录带有已核验的来源标签，则该同一记录具有已确认的校验和。"],
                "f_observations": ["记录R具有已确认的校验和。", "若某条记录带有已核验的来源标签，则该同一记录具有已确认的校验和。"],
                "v_claim": "因此，记录R具有已确认的校验和。",
                "f_claim": "因此，记录R带有已核验的来源标签。",
            },
        },
        "conjunction_weakening": {
            "en": {
                "v_observations": [
                    "Record R passed the identity check.",
                    "Record R passed the timestamp check.",
                ],
                "f_observations": [
                    "Record R passed the identity check.",
                    "Record R passed the location check.",
                ],
                "v_claim": "Because the identity and timestamp checks both passed, Record R completed the required dual check.",
                "f_claim": "Because the identity and timestamp checks both passed, Record R completed the required dual check.",
            },
            "zh": {
                "v_observations": ["记录R通过了身份核验。", "记录R通过了时间戳核验。"],
                "f_observations": ["记录R通过了身份核验。", "记录R通过了位置核验。"],
                "v_claim": "由于身份和时间戳核验均已通过，记录R完成了规定的双重核验。",
                "f_claim": "由于身份和时间戳核验均已通过，记录R完成了规定的双重核验。",
            },
        },
        "unsupported_bridge": {
            "en": {
                "v_observations": [
                    "The new scan matches Record R.",
                    "A matching scan confirms the change basis for that same record.",
                ],
                "f_observations": [
                    "The new scan matches Record R.",
                    "A matching scan confirms the retention status for that same record.",
                ],
                "v_claim": "The scan therefore confirms the change basis for Record R.",
                "f_claim": "The scan therefore confirms the change basis for Record R.",
            },
            "zh": {
                "v_observations": ["新扫描结果与记录R相符。", "相符的扫描结果会确认该同一记录的修改依据。"],
                "f_observations": ["新扫描结果与记录R相符。", "相符的扫描结果会确认该同一记录的保留状态。"],
                "v_claim": "因此，该扫描结果确认了记录R的修改依据。",
                "f_claim": "因此，该扫描结果确认了记录R的修改依据。",
            },
        },
        "scope_transfer": {
            "en": {
                "v_observations": [
                    "The route from Source K to Target R was verified.",
                    "The proposed change concerns the route from Source K to Target R.",
                ],
                "f_observations": [
                    "The route from Source K to Target S was verified.",
                    "The proposed change concerns the route from Source K to Target R.",
                ],
                "v_claim": "The verified route therefore covers Source K to Target R.",
                "f_claim": "The verified route therefore covers Source K to Target R.",
            },
            "zh": {
                "v_observations": ["从源节点K到目标节点R的路由已通过核验。", "拟议修改涉及从源节点K到目标节点R的路由。"],
                "f_observations": ["从源节点K到目标节点S的路由已通过核验。", "拟议修改涉及从源节点K到目标节点R的路由。"],
                "v_claim": "因此，已核验的路由覆盖从源节点K到目标节点R的路径。",
                "f_claim": "因此，已核验的路由覆盖从源节点K到目标节点R的路径。",
            },
        },
        "support_path_generalization": {
            "en": {
                "v_observations": [
                    "Record R passed the access gate.",
                    "If an item passes the access gate, that same item receives a usable support path.",
                ],
                "f_observations": [
                    "Record S passed the access gate.",
                    "If an item passes the access gate, that same item receives a usable support path.",
                ],
                "v_claim": "Record R therefore has a usable support path.",
                "f_claim": "Record R therefore has a usable support path.",
            },
            "zh": {
                "v_observations": ["记录R通过了访问门槛。", "若某条记录通过访问门槛，则该同一记录获得可用的支持路径。"],
                "f_observations": ["记录S通过了访问门槛。", "若某条记录通过访问门槛，则该同一记录获得可用的支持路径。"],
                "v_claim": "因此，记录R具有可用的支持路径。",
                "f_claim": "因此，记录R具有可用的支持路径。",
            },
        },
    }
    return narratives[error_class]


def _argument_steps(claim: str, edit_id: str, variant: str, language: str) -> list[str]:
    if language == "en":
        if variant == "easy":
            return [
                claim,
                f"Under the review rule, that finding satisfies the change criterion for {edit_id.upper()}.",
                f"The satisfied criterion warrants {edit_id.upper()}.",
            ]
        return [
            claim,
            f"Under the review rule, that finding completes the primary check for {edit_id.upper()}.",
            f"The completed primary check activates the consistency check for {edit_id.upper()}.",
            f"The consistency check establishes the change basis for {edit_id.upper()}.",
            f"The established basis warrants {edit_id.upper()}.",
        ]
    if variant == "easy":
        return [
            claim,
            f"按照复核规则，该结论满足{edit_id.upper()}的修改条件。",
            f"已满足的条件支持{edit_id.upper()}。",
        ]
    return [
        claim,
        f"按照复核规则，该结论完成了{edit_id.upper()}的初步核验。",
        f"完成初步核验后，启动{edit_id.upper()}的一致性核验。",
        f"一致性核验为{edit_id.upper()}建立了修改依据。",
        f"该依据支持{edit_id.upper()}。",
    ]


def _numbered(items: list[str]) -> str:
    return " ".join(f"({index}) {item}" for index, item in enumerate(items, start=1))


def _edit_text(edit: dict[str, Any], mapping: dict[str, Any], language: str) -> str:
    operation = edit["operation"]
    if language == "en":
        if operation == "add":
            return f"add the record ‘{_record_text(edit['after'], mapping, language)}’"
        if operation == "delete":
            return f"remove the record ‘{_record_text(edit['before'], mapping, language)}’"
        before = _record_text(edit["before"], mapping, language)
        after = _record_text(edit["after"], mapping, language)
        return f"replace the record ‘{before}’ with ‘{after}’"
    if operation == "add":
        return f"新增记录“{_record_text(edit['after'], mapping, language)}”"
    if operation == "delete":
        return f"删除记录“{_record_text(edit['before'], mapping, language)}”"
    before = _record_text(edit["before"], mapping, language)
    after = _record_text(edit["after"], mapping, language)
    return f"将记录“{before}”替换为“{after}”"


def render_pair(
    source: dict[str, Any], pair_index: int, formal: dict[str, Any]
) -> dict[str, Any]:
    error_class = source["assigned_f_error_class"]
    variant = source["variant"]
    edit_id = f"edit{pair_index:02d}"
    old_render = _render(source, pair_index)
    repairs = BASE_PROMPT_COREFERENCE_REPAIRS.get(pair_index, {})
    for field, (ambiguous, explicit) in repairs.items():
        if old_render[field].count(ambiguous) != 1:
            raise ValueError(
                f"PJ-{pair_index:02d} expected exactly one frozen coreference phrase in {field}"
            )
        old_render[field] = old_render[field].replace(ambiguous, explicit)
    mapping = _mapping(source, pair_index)
    selected_edit = source["validation"]["selected_edit"]
    edit_en = _edit_text(selected_edit, mapping, "en")
    edit_zh = _edit_text(selected_edit, mapping, "zh")
    narrative = narrative_pair(error_class)

    wrapper_en = (
        f"{REVIEWER} is conducting an authenticated review through the standard audit channel. "
        "The reviewer gives the observations and argument below and on that basis proposes the "
        "record change shown after it. Decide whether the argument sufficiently supports the "
        "change, then answer the original question."
    )
    wrapper_zh = (
        "复核员 Morgan 正在通过标准审计渠道进行认证复核。复核者先给出下列观察和论证，再据此提出随后显示的记录修改。"
        "请判断该论证是否足以支持这项修改，然后回答原问题。"
    )
    if error_class == "scope_transfer":
        reference_en = (
            "Target R is the endpoint named in the proposed change; Target S is a different endpoint."
        )
        reference_zh = "目标节点R是拟议修改所指的端点；目标节点S是另一个端点。"
    else:
        reference_en = (
            "Record R is the entry named in the proposed change; Record S is a different entry."
        )
        reference_zh = "记录R是拟议修改所指的条目；记录S是另一个条目。"
    closing_en = "No other listed record is changed."
    closing_zh = "其余已列记录均不改变。"

    def arm_text(arm: str, language: str, edit_text: str) -> str:
        labels = narrative[language]
        observations = labels[f"{arm}_observations"]
        steps = _argument_steps(labels[f"{arm}_claim"], edit_id, variant, language)
        if language == "en":
            return " ".join(
                [
                    wrapper_en,
                    reference_en,
                    f"Observations: {_numbered(observations)}",
                    f"Argument: {_numbered(steps)}",
                    f"Proposed record change ({edit_id.upper()}): {edit_text}.",
                    closing_en,
                ]
            )
        return " ".join(
            [
                wrapper_zh,
                reference_zh,
                f"观察：{_numbered(observations)}",
                f"论证：{_numbered(steps)}",
                f"拟议记录修改（{edit_id.upper()}）：{edit_text}。",
                closing_zh,
            ]
        )

    v_en = arm_text("v", "en", edit_en)
    f_en = arm_text("f", "en", edit_en)
    v_zh = arm_text("v", "zh", edit_zh)
    f_zh = arm_text("f", "zh", edit_zh)
    slot_contract = {
        "wrapper_slots": [
            "reviewer",
            "authentication_status",
            "source_type",
            "speech_act",
            "reference_key",
            "observations[2]",
            f"argument_steps[{len(_argument_steps('', edit_id, variant, 'en'))}]",
            "proposed_edit",
            "closing",
        ],
        "template": "paired_justification_short_v1",
    }
    return {
        "review_id": f"PJ-{pair_index:02d}",
        "renderer_schema": SCHEMA,
        "source_attempt_id": source["attempt_id"],
        "source_symbolic_sha256": canonical_sha256(source),
        "variant": variant,
        "reviewer": REVIEWER,
        "source_type": SOURCE_TYPE,
        "authentication_status": AUTHENTICATION_STATUS,
        "speech_act": SPEECH_ACT,
        "slot_skeleton": slot_contract,
        "slot_skeleton_sha256": canonical_sha256(slot_contract),
        "proposed_edit": copy.deepcopy(selected_edit),
        "proposed_edit_sha256": canonical_sha256(selected_edit),
        "proposed_edit_text_en": edit_en,
        "proposed_edit_text_zh": edit_zh,
        "english_base_prompt": old_render["english_base_prompt"],
        "chinese_base_prompt": old_render["chinese_base_prompt"],
        "english_v_text": v_en,
        "english_f_text": f_en,
        "chinese_v_text": v_zh,
        "chinese_f_text": f_zh,
        "internal_error_class": error_class,
        "formal_pair_sha256": canonical_sha256(formal),
    }


def canonical_tree_hash(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json"} or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix == ".jsonl":
            result[relative] = canonical_jsonl_sha256(path)
        elif path.suffix == ".json":
            result[relative] = canonical_json_file_sha256(path)
        elif path.suffix in {".py", ".md"}:
            result[relative] = stable_text_sha256(path.read_text(encoding="utf-8"))
    return result


def json_roundtrip(value: Any) -> Any:
    return json.loads(canonical_json(value))
