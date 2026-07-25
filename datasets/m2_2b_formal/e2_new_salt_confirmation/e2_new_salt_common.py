"""Frozen contracts and deterministic helpers for E2 new-salt confirmation."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V11 = HERE.parent / "prover_generator_hardening_v1_1"
V12 = HERE.parent / "prover_generator_hardening_v1_2"
DEV = HERE.parent / "prover_generator_development_v1"
DEV11 = HERE.parent / "prover_generator_development_v1_1"
E2A = HERE.parent / "e2_phase_validity_choice"
for dependency in (V11, V12, DEV, DEV11, E2A, ROOT / "src"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import naturalize_prover_generator_hardening_v1_1 as naturalizer  # noqa: E402
import paired_justification_common as pairing  # noqa: E402
from development_v1_1_common import (  # noqa: E402
    ANALYSIS_SCHEMA,
    RESPONSE_CONTRACT,
    arm_messages,
)
from hardening_common import (  # noqa: E402,F401
    apply_edit,
    canonical_json,
    canonical_json_file_sha256,
    canonical_jsonl_sha256,
    canonical_sha256,
    evaluate_fixed_point,
    fact,
    file_sha256,
    literal_key,
    proof_count,
    read_jsonl,
    run_mace4_countermodel,
    shortest_proof,
    stable_seed,
    validate_with_both,
    write_json,
    write_jsonl,
)

SCHEMA = "cedar.e2_new_salt_confirmation.v1"
SALT = "cedar-e2-new-salt-confirmation-v1-20260725"
SALT_SHA256 = "198059a92ba743b5c9576b92270d8b714540b8505fcef05eb26f37436cefe34f"
MASTER_SEED = 2026072501
BOOTSTRAP_SEED = 2026072502
BOOTSTRAP_DRAWS = 20_000
MODEL_REPO = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "sdpa"
CAP_EXPERIMENT = "layers_46:54-p0.25"
CAP_LAYERS = tuple(range(46, 54))
EXPECTED_AXIS_SHA256 = "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211"
EXPECTED_CAP_CONFIG_SHA256 = "6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24"
PARENT_REVIEW_SHA256 = "a6c8edba466ef72644547273370a64640bb42a6c841a4db2a1db7d6db4a9549e"
BASE_COMMIT = "10120d04d00e3c9122bf2a2ba590bd403373fa45"
E2A_RESULT_COMMIT = "0c8517bb5fb4640f7d5e4d51356d0cf167046643"

ARMS = ("N", "R", "V", "F")
VARIANTS = ("easy", "medium")
OPTION_LETTERS = ("A", "B", "C", "D")
SEMANTIC_LABELS = ("true", "false", "unknown", "invalid")
ERROR_CLASSES = (
    "cross_entity_transfer",
    "converse_error",
    "conjunction_weakening",
    "unsupported_bridge",
    "scope_transfer",
    "support_path_generalization",
)
ERROR_POSITIONS = ("early", "middle", "late")
STRUCTURE_CLASSES = tuple(f"PGF-{index:02d}" for index in range(1, 9))
FAMILY_IDS = tuple(f"CEDAR-E2C-F{index:02d}" for index in range(1, 13))
ORDERED_TRANSITIONS = tuple(
    (initial, target)
    for initial in OPTION_LETTERS
    for target in OPTION_LETTERS
    if initial != target
)

TOKENIZER_BINDING = {"repo": MODEL_REPO, "revision": MODEL_REVISION}
PROTOCOL_FILES = (
    "docs/E2_NEW_SALT_CONFIRMATION_PROTOCOL.md",
    "docs/E2_NEW_SALT_CONFIRMATION_ANALYSIS_PLAN.json",
    "docs/E2_PROTOCOL_FREEZE_MANIFEST.json",
)

# These lexicons are bank-local and deliberately disjoint from the earlier
# Amber/Cobalt/... and logistics/sample/... rendering pools.
NEW_DOMAINS = [
    ("orbital docket", "capsule", "轨道清单", "舱体"),
    ("habitat ledger", "module", "栖居账册", "舱段"),
    ("field archive", "specimen", "野外档案", "标本"),
    ("signal catalog", "beacon", "信号目录", "信标"),
    ("harbor manifest", "vessel", "港口清单", "船只"),
    ("workshop register", "assembly", "工坊登记簿", "组件"),
    ("survey notebook", "station", "测绘笔记", "测站"),
    ("library docket", "volume", "馆藏清单", "卷册"),
]
NEW_CODE_NAMES = [
    ("Alder", "桤木"),
    ("Birch", "白桦"),
    ("Cedar", "雪松"),
    ("Dune", "沙丘"),
    ("Ember", "余烬"),
    ("Flint", "燧石"),
    ("Grove", "林地"),
    ("Harbor", "港湾"),
    ("Iris", "鸢尾"),
    ("Juniper", "杜松"),
    ("Kestrel", "红隼"),
    ("Morrow", "晨曦"),
    ("Nimbus", "雨云"),
    ("Orbit", "轨道"),
    ("Prairie", "草原"),
    ("Quill", "羽笔"),
    ("Rowan", "花楸"),
    ("Solace", "静湾"),
    ("Timber", "木梁"),
    ("Vale", "谷地"),
    ("Willow", "柳木"),
    ("Xenon", "氙光"),
    ("Yarrow", "蓍草"),
    ("Zenith", "天顶"),
]


class GateError(ValueError):
    """Raised when a frozen construction or release invariant fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def stable_text_sha256(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonical_text_sha256(path: Path) -> str:
    return stable_text_sha256(path.read_text(encoding="utf-8"))


def object_sha256(value: Any) -> str:
    payload = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _rank(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SALT}|{namespace}|{value}".encode()).hexdigest()


def family_plan() -> list[dict[str, Any]]:
    """Create the frozen balanced plan solely from family keys and the salt."""

    family_order = sorted(FAMILY_IDS, key=lambda value: _rank("family-order", value))
    extra_structures = sorted(
        STRUCTURE_CLASSES, key=lambda value: _rank("extra-structure", value)
    )[:4]
    structure_slots = sorted(
        [*STRUCTURE_CLASSES, *extra_structures],
        key=lambda value: _rank("structure-slot", value),
    )
    class_slots = sorted(
        [name for name in ERROR_CLASSES for _ in range(2)],
        key=lambda value: _rank("f-class-slot", value),
    )
    assigned: dict[str, dict[str, Any]] = {}
    for family_id, structure, error_class in zip(
        family_order, structure_slots, class_slots, strict=True
    ):
        assigned[family_id] = {
            "family_id": family_id,
            "structure_class": structure,
            "error_class": error_class,
        }

    transition_maps: dict[str, dict[str, tuple[str, str]]] = {}
    position_maps: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        ordered_families = sorted(
            FAMILY_IDS, key=lambda value: _rank(f"transition-{variant}", value)
        )
        transitions = sorted(
            ORDERED_TRANSITIONS,
            key=lambda value: _rank(f"transition-slot-{variant}", "->".join(value)),
        )
        transition_maps[variant] = dict(zip(ordered_families, transitions, strict=True))
        position_order = sorted(
            FAMILY_IDS, key=lambda value: _rank(f"error-position-{variant}", value)
        )
        position_maps[variant] = {
            family_id: ERROR_POSITIONS[index // 4]
            for index, family_id in enumerate(position_order)
        }

    result = []
    for family_id in FAMILY_IDS:
        item = assigned[family_id]
        result.append(
            {
                **item,
                "variants": {
                    variant: {
                        "initial_answer": transition_maps[variant][family_id][0],
                        "target_answer": transition_maps[variant][family_id][1],
                        "error_position": position_maps[variant][family_id],
                    }
                    for variant in VARIANTS
                },
            }
        )
    return result


def semantic_mapping(
    initial_label: str, target_label: str, initial: str, target: str
) -> dict[str, str]:
    require(initial_label != target_label, "Semantic transition must change")
    require(initial != target, "Letter transition must change")
    mapping = {initial_label: initial, target_label: target}
    remaining_labels = [label for label in SEMANTIC_LABELS if label not in mapping]
    remaining_letters = [letter for letter in OPTION_LETTERS if letter not in mapping.values()]
    mapping.update(dict(zip(remaining_labels, remaining_letters, strict=True)))
    require(set(mapping) == set(SEMANTIC_LABELS), "Semantic map incomplete")
    require(set(mapping.values()) == set(OPTION_LETTERS), "Letter map is not bijective")
    return mapping


def _new_mapping(row: dict[str, Any], review_index: int) -> dict[str, Any]:
    world = row["builder_output"]["initial_world"]
    predicates: dict[str, int] = {}
    for record in world["records"]:
        literals = (
            [record["literal"]]
            if record["kind"] == "fact"
            else [*record["body"], record["head"]]
        )
        for item in literals:
            predicates[item["predicate"]] = len(item["arguments"])
    query = row["builder_output"]["query"]
    predicates[query["predicate"]] = len(query["arguments"])
    certificate = row["validation"]["f_certificate"]
    for item in [
        *certificate["entailed_premises"],
        certificate["invalid_inference"]["claimed_conclusion"],
    ]:
        predicates[item["predicate"]] = len(item["arguments"])

    offset = int(_rank("predicate-lexicon", row["attempt_id"]), 16) % len(NEW_CODE_NAMES)
    predicate_map = {}
    for index, predicate in enumerate(sorted(predicates)):
        en, zh = NEW_CODE_NAMES[(offset + index) % len(NEW_CODE_NAMES)]
        predicate_map[predicate] = {
            "en": f"{en} {index + 1}",
            "zh": f"{zh}{index + 1}",
            "arity": predicates[predicate],
        }
    _, entity_noun, _, entity_noun_zh = NEW_DOMAINS[(review_index - 1) % len(NEW_DOMAINS)]
    constants = sorted(world["constants"])
    chinese_labels = list("甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地玄黄")
    entity_map = {
        constant: {
            "en": f"{entity_noun.title()} {chr(65 + index)}"
            if index < 26
            else f"{entity_noun.title()} {index + 1}",
            "zh": f"{entity_noun_zh}{chinese_labels[index]}"
            if index < len(chinese_labels)
            else f"{entity_noun_zh}{index + 1}",
        }
        for index, constant in enumerate(constants)
    }
    return {"predicates": predicate_map, "entities": entity_map}


def install_new_lexicon() -> None:
    naturalizer.DOMAINS = NEW_DOMAINS
    naturalizer.CODE_NAMES = NEW_CODE_NAMES
    naturalizer._mapping = _new_mapping
    pairing._mapping = _new_mapping


install_new_lexicon()


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\bedit\s*\d+\b", "edit#", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def surface_payload(base: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "base_id": base["base_id"],
        "english_base_prompt": base["english_base_prompt"],
        "chinese_base_prompt": base["chinese_base_prompt"],
        "english_v_text": base["english_v_text"],
        "chinese_v_text": base["chinese_v_text"],
        "english_f_text": base["english_f_text"],
        "chinese_f_text": base["chinese_f_text"],
    }


def arm_row(base: Mapping[str, Any], arm: str, followup: str) -> dict[str, Any]:
    require(arm in ARMS, f"Unknown arm: {arm}")
    gold = base["target_answer"] if arm == "V" else base["initial_answer"]
    messages = arm_messages(
        str(base["english_base_prompt"]), str(base["initial_answer"]), followup
    )
    return {
        "schema": SCHEMA,
        "id": f"{base['base_id']}-{arm}",
        "base_id": base["base_id"],
        "family_id": base["family_id"],
        "structure_class": base["structure_class"],
        "variant": base["variant"],
        "arm": arm,
        "initial_answer": base["initial_answer"],
        "target_answer": base["target_answer"],
        "gold_answer": gold,
        "expected_outcome": "induced_target" if arm == "V" else "initial",
        "error_class": base["error_class"] if arm == "F" else None,
        "error_position": base["error_position"] if arm == "F" else None,
        "followup": followup,
        "messages": messages,
        "prompt_canonical_sha256": object_sha256(messages),
        "response_contract": RESPONSE_CONTRACT,
        "analysis_schema": json.loads(json.dumps(ANALYSIS_SCHEMA)),
        "position_transition": f"{base['initial_answer']}->{base['target_answer']}",
    }


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)
