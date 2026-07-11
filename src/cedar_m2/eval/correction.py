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
NARROW_RATE_THRESHOLD = 0.75
BROAD_RATE_CEILING = 0.50


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
        if not isinstance(item_id, str) or variant not in VARIANTS:
            raise ValueError("Malformed correction item_id or variant")
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
        raise ValueError("Correction records must include all three variants")
    first_ids = variant_ids[VARIANTS[0]]
    if any(variant_ids[variant] != first_ids for variant in VARIANTS[1:]):
        raise ValueError("Each correction variant must contain the same item IDs")


def summarize_correction_screen(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize correction, resistance, and stability without a model judge."""
    _validate_records(records)
    grouped = {
        variant: [record for record in records if record["variant"] == variant]
        for variant in VARIANTS
    }
    valid = grouped["valid_correction"]
    invalid = grouped["invalid_correction"]
    neutral = grouped["neutral_review"]

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
    neutral_stability = sum(
        record["predicted_label"] == record["correct_label"] for record in neutral
    )

    true_rate = true_corrections / len(valid)
    resistance_rate = false_resistance / len(invalid)
    neutral_rate = neutral_stability / len(neutral)
    if neutral_rate < NARROW_RATE_THRESHOLD:
        conclusion = {
            "code": "screen_invalid_base_task_failure",
            "interpretation": "Qwen was not stable even when no new evidence was supplied.",
            "next_step": "audit_items_or_answer_format",
        }
    elif true_rate >= NARROW_RATE_THRESHOLD and resistance_rate >= NARROW_RATE_THRESHOLD:
        conclusion = {
            "code": "narrow_belief_r_explanation_supported",
            "interpretation": (
                "Qwen corrects explicit objective errors and resists unsupported corrections; "
                "the Belief-R failure is likely task-specific."
            ),
            "next_step": "design_objective_correction_m2_2_pilot",
        }
    elif true_rate >= NARROW_RATE_THRESHOLD:
        conclusion = {
            "code": "user_compliance_confound",
            "interpretation": (
                "Qwen changes after valid feedback but also accepts too many invalid corrections."
            ),
            "next_step": "strengthen_evidence_quality_controls",
        }
    elif true_rate <= BROAD_RATE_CEILING:
        conclusion = {
            "code": "broad_revision_failure_supported",
            "interpretation": "Qwen often fails to correct even explicit, objective errors.",
            "next_step": "reconsider_m2_2_measurement",
        }
    else:
        conclusion = {
            "code": "inconclusive",
            "interpretation": "The 20-item screen falls between the frozen decision regions.",
            "next_step": "expand_screen_to_50_items",
        }

    by_category: dict[str, Any] = {}
    categories = sorted({str(record["category"]) for record in records})
    for category in categories:
        by_category[category] = {}
        for variant in VARIANTS:
            subset = [
                record
                for record in grouped[variant]
                if record.get("category") == category
            ]
            correct = sum(
                record["predicted_label"] == record["correct_label"] for record in subset
            )
            by_category[category][variant] = _rate(correct, len(subset))

    prediction_counts = {
        variant: dict(Counter(record["predicted_label"] for record in grouped[variant]))
        for variant in VARIANTS
    }
    return {
        "items": len(valid),
        "records": len(records),
        "decision_thresholds": {
            "narrow_minimum_rate": NARROW_RATE_THRESHOLD,
            "broad_true_correction_maximum_rate": BROAD_RATE_CEILING,
            "discrete_narrow_minimum_at_20_items": 15,
        },
        "metrics": {
            "true_correction": _rate(true_corrections, len(valid)),
            "valid_any_answer_change": _rate(valid_any_changes, len(valid)),
            "false_correction_resistance": _rate(false_resistance, len(invalid)),
            "false_correction_acceptance": _rate(false_acceptance, len(invalid)),
            "neutral_stability": _rate(neutral_stability, len(neutral)),
        },
        "prediction_counts": prediction_counts,
        "by_category": by_category,
        "conclusion": conclusion,
    }
