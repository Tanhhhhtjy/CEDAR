"""Deterministic single-token multiple-choice scoring."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from cedar_m2.models.qwen import model_input_device

LABELS = ("A", "B", "C")
MCQ_TASK_INSTRUCTION = (
    "This is a defeasible-reasoning task. Interpret the conditionals using commonsense causal "
    "dependencies, not as strict material implications. With two premises, infer the initial "
    "conclusion. If a third conditional premise is present, decide whether its antecedent is an "
    "additional requirement that must combine with the first antecedent (conjunctive: update the "
    'prior conclusion to "may or may not") or a separate alternative route to the same consequent '
    "(disjunctive: maintain the prior conclusion)."
)
MCQ_ANSWER_INSTRUCTION = "Answer with exactly one uppercase letter: A, B, or C. Do not explain."
MCQ_PROMPT_SUFFIX = f"{MCQ_TASK_INSTRUCTION}\n\n{MCQ_ANSWER_INSTRUCTION}"


def mcq_prompt_sha256() -> str:
    """Fingerprint the task and answer instructions frozen after development."""
    return hashlib.sha256(MCQ_PROMPT_SUFFIX.encode()).hexdigest()


@dataclass(frozen=True)
class MCQScore:
    predicted_label: str
    label_logits: dict[str, float]
    correct_label: str | None = None
    correct: bool | None = None
    margin: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def label_token_ids(tokenizer: Any, labels: Sequence[str] = LABELS) -> dict[str, int]:
    """Return label token IDs, rejecting tokenizers that split a label."""
    result: dict[str, int] = {}
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Label {label!r} is not a single token: {encoded}")
        result[label] = int(encoded[0])
    if len(set(result.values())) != len(result):
        raise ValueError("Multiple labels map to the same token ID")
    return result


def render_qwen_mcq_prompt(tokenizer: Any, question: str) -> str:
    """Render a fixed no-thinking Qwen chat prompt ending before its answer."""
    user_content = f"{question.rstrip()}\n\n{MCQ_PROMPT_SUFFIX}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def compute_mcq_scores(
    logits: Tensor,
    token_ids: dict[str, int],
    *,
    correct_labels: Sequence[str] | None = None,
) -> list[MCQScore]:
    """Convert full-vocabulary last-token logits into label predictions/margins."""
    if logits.ndim != 2:
        raise ValueError("Logits must have shape [batch, vocabulary]")
    labels = tuple(token_ids)
    if not labels:
        raise ValueError("At least one label is required")
    if correct_labels is not None and len(correct_labels) != logits.shape[0]:
        raise ValueError("correct_labels length must match batch size")

    ids = torch.tensor([token_ids[label] for label in labels], device=logits.device)
    selected = logits.index_select(dim=-1, index=ids).detach().float().cpu()
    scores: list[MCQScore] = []
    for row_index, row in enumerate(selected):
        predicted_index = int(row.argmax().item())
        predicted = labels[predicted_index]
        values = {label: float(row[index].item()) for index, label in enumerate(labels)}

        if correct_labels is None:
            scores.append(MCQScore(predicted_label=predicted, label_logits=values))
            continue

        correct_label = correct_labels[row_index].upper()
        if correct_label not in token_ids:
            raise ValueError(f"Unknown correct label {correct_label!r}")
        correct_index = labels.index(correct_label)
        incorrect = torch.cat((row[:correct_index], row[correct_index + 1 :]))
        margin = float((row[correct_index] - incorrect.max()).item())
        scores.append(
            MCQScore(
                predicted_label=predicted,
                label_logits=values,
                correct_label=correct_label,
                correct=predicted == correct_label,
                margin=margin,
            )
        )
    return scores


def score_mcq_prompts(
    model: nn.Module,
    tokenizer: Any,
    questions: Sequence[str],
    *,
    correct_labels: Sequence[str] | None = None,
    batch_size: int = 4,
) -> list[MCQScore]:
    """Score Qwen prompts without sampling or generating explanation tokens."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if correct_labels is not None and len(correct_labels) != len(questions):
        raise ValueError("correct_labels length must match questions")

    tokens = label_token_ids(tokenizer)
    prompts = [render_qwen_mcq_prompt(tokenizer, question) for question in questions]
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
        batch_correct = correct_labels[start:end] if correct_labels is not None else None
        results.extend(
            compute_mcq_scores(output.logits[:, -1, :], tokens, correct_labels=batch_correct)
        )
    return results
