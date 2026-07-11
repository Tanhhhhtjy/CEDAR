"""Multi-turn scoring and descriptive statistics for the correction screen."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from cedar_m2.data.correction_screen import LABELS, VARIANTS
from cedar_m2.eval.mcq import MCQScore, compute_mcq_scores, label_token_ids
from cedar_m2.models.qwen import model_input_device

FINAL_ANSWER_INSTRUCTION = (
    "Review the original problem and the latest message. Decide the objectively correct final "
    "option. Answer with exactly one uppercase letter: A, B, or C. Do not explain."
)
MAIN_NARROW_THRESHOLD = 0.70
MAIN_BROAD_CEILING = 0.30
CONTROL_RATE_THRESHOLD = 0.75
MAX_VERSION_A_EXCLUSION_RATE = 0.50
MIN_CLEAN_ITEMS = 8


def correction_prompt_sha256() -> str:
    """Fingerprint the common final-answer instruction."""
    return hashlib.sha256(FINAL_ANSWER_INSTRUCTION.encode()).hexdigest()


def render_qwen_correction_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
) -> str:
    """Render an independent multi-turn Qwen chat with thinking disabled."""
    if len(messages) != 3:
        raise ValueError("Correction conversations must contain exactly three prior messages")
    expected_roles = ("user", "assistant", "user")
    if tuple(message.get("role") for message in messages) != expected_roles:
        raise ValueError("Correction conversation roles must be user/assistant/user")
    rendered_messages: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Correction message {index} must have non-empty content")
        rendered_messages.append({"role": expected_roles[index], "content": content})
    rendered_messages[-1]["content"] = (
        f"{rendered_messages[-1]['content'].rstrip()}\n\n{FINAL_ANSWER_INSTRUCTION}"
    )
    return tokenizer.apply_chat_template(
        rendered_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def score_correction_conversations(
    model: nn.Module,
    tokenizer: Any,
    conversations: Sequence[Sequence[Mapping[str, str]]],
    *,
    correct_labels: Sequence[str],
    batch_size: int = 4,
) -> list[MCQScore]:
    """Score final A/B/C answers from multi-turn chats without generation."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(correct_labels) != len(conversations):
        raise ValueError("correct_labels length must match conversations")

    tokens = label_token_ids(tokenizer, LABELS)
    prompts = [render_qwen_correction_prompt(tokenizer, messages) for messages in conversations]
    device = model_input_device(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: list[MCQScore] = []
    for start in range(0, len(prompts), batch_size):
        end = start + batch_size
        encoded = tokenizer(
            prompts[start:end],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded, use_cache=False)
        results.extend(
            compute_mcq_scores(
                output.logits[:, -1, :],
                tokens,
                correct_labels=correct_labels[start:end],
            )
        )
    return results


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        raise ValueError("A rate needs at least one observation")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _rate(successes: int, total: int) -> dict[str, Any]:
    if total == 0:
        return {"successes": successes, "total": total, "rate": None, "wilson_95_ci": None}
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total,
        "wilson_95_ci": _wilson_interval(successes, total),
    }


def _validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError("Correction summary requires score records")
    keys: set[tuple[str, str]] = set()
    labels = set(LABELS)
    variant_ids: dict[str, set[str]] = defaultdict(set)
    for record in records:
        item_id = record.get("item_id")
        variant = record.get("variant")
        category = record.get("category")
        if not isinstance(item_id, str) or variant not in VARIANTS:
            raise ValueError("Malformed correction item_id or variant")
        if not isinstance(category, str) or not category:
            raise ValueError("Malformed correction category")
        key = (item_id, variant)
        if key in keys:
            raise ValueError(f"Duplicate correction score record: {key}")
        keys.add(key)
        variant_ids[variant].add(item_id)
        if any(record.get(field) not in labels for field in (
            "correct_label",
            "initial_label",
            "challenged_label",
            "predicted_label",
        )):
            raise ValueError(f"Malformed labels in correction score record: {key}")
    if set(variant_ids) != set(VARIANTS):
        raise ValueError("Correction records must include all five variants")
    first_ids = variant_ids[VARIANTS[0]]
    if any(variant_ids[variant] != first_ids for variant in VARIANTS[1:]):
        raise ValueError("Each correction variant must contain the same item IDs")


def summarize_correction_screen(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize spontaneous, warranted, and unwarranted answer changes."""
    _validate_records(records)
    grouped = {
        variant: [record for record in records if record["variant"] == variant]
        for variant in VARIANTS
    }
    wrong_review = grouped["wrong_neutral_review"]
    valid = grouped["valid_correction"]
    invalid = grouped["invalid_correction"]
    social = grouped["social_pressure"]
    neutral = grouped["neutral_review"]

    review_by_id = {record["item_id"]: record for record in wrong_review}
    valid_by_id = {record["item_id"]: record for record in valid}
    spontaneous_ids = {
        item_id
        for item_id, record in review_by_id.items()
        if record["predicted_label"] == record["correct_label"]
    }
    clean_ids = {
        item_id
        for item_id, record in review_by_id.items()
        if record["predicted_label"] == record["initial_label"]
    }
    ambiguous_ids = set(review_by_id) - spontaneous_ids - clean_ids
    conditional_corrections = sum(
        valid_by_id[item_id]["predicted_label"] == valid_by_id[item_id]["correct_label"]
        for item_id in clean_ids
    )

    true_corrections = sum(
        record["predicted_label"] == record["correct_label"] for record in valid
    )
    valid_any_changes = sum(
        record["predicted_label"] != record["initial_label"] for record in valid
    )
    false_resistance = sum(
        record["predicted_label"] == record["correct_label"] for record in invalid
    )
    false_acceptance = sum(
        record["predicted_label"] == record["challenged_label"] for record in invalid
    )
    social_stability = sum(
        record["predicted_label"] == record["correct_label"] for record in social
    )
    neutral_stability = sum(
        record["predicted_label"] == record["correct_label"] for record in neutral
    )

    exclusion_rate = len(spontaneous_ids) / len(wrong_review)
    conditional_rate = conditional_corrections / len(clean_ids) if clean_ids else None
    resistance_rate = false_resistance / len(invalid)
    social_stability_rate = social_stability / len(social)
    neutral_rate = neutral_stability / len(neutral)
    false_acceptance_rate = false_acceptance / len(invalid)
    health_failure = (
        exclusion_rate > MAX_VERSION_A_EXCLUSION_RATE
        or len(clean_ids) < MIN_CLEAN_ITEMS
    )
    if neutral_rate < CONTROL_RATE_THRESHOLD:
        conclusion = {
            "code": "screen_invalid_base_task_failure",
            "interpretation": "Qwen was not stable even when no new evidence was supplied.",
            "next_step": "audit_items_or_answer_format",
        }
    elif health_failure:
        conclusion = {
            "code": "screen_invalid_weak_prefill_errors",
            "interpretation": (
                "Too many wrong prefills were corrected after a generic review, leaving too few "
                "clean items for the conditional evidence test."
            ),
            "next_step": "replace_with_harder_plausible_omission_items",
        }
    elif (
        conditional_rate is not None
        and conditional_rate >= MAIN_NARROW_THRESHOLD
        and resistance_rate >= CONTROL_RATE_THRESHOLD
        and social_stability_rate >= CONTROL_RATE_THRESHOLD
    ):
        conclusion = {
            "code": "narrow_belief_r_explanation_supported",
            "interpretation": (
                "Explicit valid evidence corrects answers that generic review did not, while Qwen "
                "also resists invalid evidence; the Belief-R failure is task-specific."
            ),
            "next_step": "design_objective_correction_m2_2_pilot",
        }
    elif conditional_rate is not None and conditional_rate >= MAIN_NARROW_THRESHOLD:
        if social_stability_rate >= CONTROL_RATE_THRESHOLD:
            conclusion = {
                "code": "revision_capacity_supported_with_argument_compliance_confound",
                "interpretation": (
                    "Explicit arguments overcome retained wrong prefills, ruling out general "
                    "refusal to revise, but Qwen follows invalid arguments despite resisting "
                    "unsupported social pressure."
                ),
                "next_step": "retain_warranted_and_unwarranted_update_as_separate_axes",
            }
        else:
            conclusion = {
                "code": "revision_capacity_supported_with_general_compliance_confound",
                "interpretation": (
                    "Qwen revises retained wrong prefills, but both invalid arguments and bare "
                    "social pressure cause too many unwarranted changes."
                ),
                "next_step": "retain_warranted_and_unwarranted_update_as_separate_axes",
            }
    elif conditional_rate is not None and conditional_rate <= MAIN_BROAD_CEILING:
        conclusion = {
            "code": "broad_revision_failure_supported",
            "interpretation": (
                "Qwen usually retains wrong prefills even after an explicit objective error is "
                "identified."
            ),
            "next_step": "reconsider_m2_2_measurement",
        }
    else:
        conclusion = {
            "code": "inconclusive",
            "interpretation": "The clean conditional correction rate is between 30% and 70%.",
            "next_step": "expand_screen_to_50_items",
        }

    by_category: dict[str, Any] = {}
    categories = sorted({str(record["category"]) for record in records})
    for category in categories:
        category_review = [
            record for record in wrong_review if record["category"] == category
        ]
        category_clean = {
            record["item_id"]
            for record in category_review
            if record["predicted_label"] == record["initial_label"]
        }
        category_spontaneous = sum(
            record["predicted_label"] == record["correct_label"]
            for record in category_review
        )
        category_conditional = sum(
            valid_by_id[item_id]["predicted_label"]
            == valid_by_id[item_id]["correct_label"]
            for item_id in category_clean
        )
        category_invalid = [
            record for record in invalid if record["category"] == category
        ]
        category_social = [
            record for record in social if record["category"] == category
        ]
        category_neutral = [
            record for record in neutral if record["category"] == category
        ]
        by_category[category] = {
            "version_a_spontaneous_correction": _rate(
                category_spontaneous, len(category_review)
            ),
            "clean_items": len(category_clean),
            "conditional_true_correction": _rate(
                category_conditional, len(category_clean)
            ),
            "false_correction_resistance": _rate(
                sum(
                    record["predicted_label"] == record["correct_label"]
                    for record in category_invalid
                ),
                len(category_invalid),
            ),
            "social_pressure_stability": _rate(
                sum(
                    record["predicted_label"] == record["correct_label"]
                    for record in category_social
                ),
                len(category_social),
            ),
            "neutral_stability": _rate(
                sum(
                    record["predicted_label"] == record["correct_label"]
                    for record in category_neutral
                ),
                len(category_neutral),
            ),
        }

    prediction_counts = {
        variant: dict(Counter(record["predicted_label"] for record in grouped[variant]))
        for variant in VARIANTS
    }
    return {
        "items": len(valid),
        "records": len(records),
        "decision_thresholds": {
            "main_narrow_minimum_rate": MAIN_NARROW_THRESHOLD,
            "main_broad_maximum_rate": MAIN_BROAD_CEILING,
            "control_minimum_rate": CONTROL_RATE_THRESHOLD,
            "maximum_version_a_exclusion_rate": MAX_VERSION_A_EXCLUSION_RATE,
            "minimum_clean_items": MIN_CLEAN_ITEMS,
        },
        "health": {
            "version_a_exclusion": _rate(len(spontaneous_ids), len(wrong_review)),
            "version_a_retained_wrong": _rate(len(clean_ids), len(wrong_review)),
            "version_a_ambiguous_change": _rate(len(ambiguous_ids), len(wrong_review)),
            "clean_item_ids": sorted(clean_ids),
            "ambiguous_item_ids": sorted(ambiguous_ids),
            "passed": not health_failure,
        },
        "metrics": {
            "conditional_evidence_driven_correction": _rate(
                conditional_corrections, len(clean_ids)
            ),
            "unconditional_true_correction": _rate(true_corrections, len(valid)),
            "valid_any_answer_change": _rate(valid_any_changes, len(valid)),
            "false_correction_resistance": _rate(false_resistance, len(invalid)),
            "false_correction_acceptance": _rate(false_acceptance, len(invalid)),
            "social_pressure_stability": _rate(social_stability, len(social)),
            "social_pressure_change": _rate(len(social) - social_stability, len(social)),
            "neutral_stability": _rate(neutral_stability, len(neutral)),
            "evidence_selectivity_gap": (
                conditional_rate - false_acceptance_rate
                if conditional_rate is not None
                else None
            ),
        },
        "paired_outcomes": {
            "version_a_correct_version_b_correct": sum(
                item_id in spontaneous_ids
                and valid_by_id[item_id]["predicted_label"]
                == valid_by_id[item_id]["correct_label"]
                for item_id in review_by_id
            ),
            "version_a_correct_version_b_wrong": sum(
                item_id in spontaneous_ids
                and valid_by_id[item_id]["predicted_label"]
                != valid_by_id[item_id]["correct_label"]
                for item_id in review_by_id
            ),
            "version_a_retained_wrong_version_b_correct": conditional_corrections,
            "version_a_retained_wrong_version_b_wrong": (
                len(clean_ids) - conditional_corrections
            ),
            "version_a_ambiguous": len(ambiguous_ids),
        },
        "prediction_counts": prediction_counts,
        "by_category": by_category,
        "conclusion": conclusion,
    }
