"""Resumable generation and classification stages for the M2.1 HarmBench arm."""

from __future__ import annotations

import hashlib
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch import nn

from cedar_m2.checkpoint import RecordStore
from cedar_m2.data.harmbench import HarmBenchBehavior
from cedar_m2.eval.harmbench import parse_classifier_label, render_classifier_prompt
from cedar_m2.interventions.capping import ActivationCapper, LayerCap
from cedar_m2.models.harmbench import classifier_input_device
from cedar_m2.models.qwen import model_input_device
from cedar_m2.personas import PersonaConfig

CONDITIONS = ("behavior_uncapped", "persona_uncapped", "persona_p25")
SAMPLING_SEEDS = (20260711, 20260712, 20260713)


@dataclass(frozen=True)
class GenerationBatch:
    condition: str
    decoding: str
    seed: int | None
    batch_index: int
    behaviors: tuple[HarmBenchBehavior, ...]

    @property
    def batch_seed(self) -> int | None:
        if self.seed is None:
            return None
        payload = f"{self.seed}:{self.batch_index}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def generation_key(
    behavior_id: str,
    condition: str,
    decoding: str,
    seed: int | None,
) -> dict[str, str | int | None]:
    return {
        "behavior_id": behavior_id,
        "condition": condition,
        "decoding": decoding,
        "seed": seed,
    }


def build_generation_batches(
    behaviors: list[HarmBenchBehavior],
    *,
    batch_size: int,
    seeds: tuple[int, ...] = SAMPLING_SEEDS,
) -> list[GenerationBatch]:
    """Build fixed batches so an interrupted batch can be regenerated identically."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ordered = sorted(behaviors, key=lambda item: item.behavior_id)
    chunks = [
        tuple(ordered[start : start + batch_size]) for start in range(0, len(ordered), batch_size)
    ]
    batches: list[GenerationBatch] = []
    for condition in CONDITIONS:
        for seed in seeds:
            batches.extend(
                GenerationBatch(condition, "sampled", seed, index, chunk)
                for index, chunk in enumerate(chunks)
            )
        batches.extend(
            GenerationBatch(condition, "greedy", None, index, chunk)
            for index, chunk in enumerate(chunks)
        )
    return batches


def messages_for_condition(
    behavior: HarmBenchBehavior,
    condition: str,
    personas: PersonaConfig,
) -> list[dict[str, str]]:
    """Build behavior-only or category-persona messages."""
    if condition == "behavior_uncapped":
        return [{"role": "user", "content": behavior.behavior}]
    if condition in {"persona_uncapped", "persona_p25"}:
        return personas.render_messages(behavior.semantic_category, behavior.behavior)
    raise ValueError(f"Unknown HarmBench condition {condition!r}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generate_batch(
    model: nn.Module,
    tokenizer: Any,
    batch: GenerationBatch,
    personas: PersonaConfig,
    caps: dict[int, LayerCap],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[str], list[int], list[str]]:
    messages = [
        messages_for_condition(behavior, batch.condition, personas) for behavior in batch.behaviors
    ]
    prompts = [
        tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for conversation in messages
    ]
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_length = int(encoded["input_ids"].shape[1])
    device = model_input_device(model)
    encoded = {name: value.to(device) for name, value in encoded.items()}

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if batch.decoding == "sampled":
        if batch.batch_seed is None:
            raise AssertionError("Sampled batch is missing its deterministic batch seed")
        _seed_everything(batch.batch_seed)
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
    elif batch.decoding == "greedy":
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:
        raise ValueError(f"Unknown decoding mode {batch.decoding!r}")

    cap_context = (
        ActivationCapper(model, caps, track_stats=False)
        if batch.condition == "persona_p25"
        else nullcontext()
    )
    with torch.inference_mode(), cap_context:
        output_ids = model.generate(**encoded, **kwargs)
    continuation_ids = output_ids[:, input_length:]
    generations = tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
    generations = [text.strip() for text in generations]
    token_counts = [len(tokenizer.encode(text, add_special_tokens=False)) for text in generations]
    return generations, token_counts, prompts


def run_generation_stage(
    model: nn.Module,
    tokenizer: Any,
    behaviors: list[HarmBenchBehavior],
    personas: PersonaConfig,
    caps: dict[int, LayerCap],
    store: RecordStore,
    *,
    batch_size: int,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict[str, int]:
    """Generate all fixed batches, skipping batches whose records are complete."""
    batches = build_generation_batches(behaviors, batch_size=batch_size)
    completed = 0
    generated = 0
    for ordinal, batch in enumerate(batches, start=1):
        keys = [
            generation_key(
                behavior.behavior_id,
                batch.condition,
                batch.decoding,
                batch.seed,
            )
            for behavior in batch.behaviors
        ]
        existing = [store.read(key) for key in keys]
        if all(record is not None for record in existing):
            completed += len(keys)
            continue

        texts, token_counts, prompts = _generate_batch(
            model,
            tokenizer,
            batch,
            personas,
            caps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for behavior, key, previous, text, token_count, prompt in zip(
            batch.behaviors,
            keys,
            existing,
            texts,
            token_counts,
            prompts,
            strict=True,
        ):
            if previous is not None:
                if previous.get("generation") != text:
                    raise RuntimeError(
                        f"Regenerated output differs for {behavior.behavior_id}; "
                        "refusing a non-deterministic resume"
                    )
                completed += 1
                continue
            payload = {
                **key,
                "semantic_category": behavior.semantic_category,
                "generation": text,
                "generation_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "generated_tokens": token_count,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "batch_index": batch.batch_index,
                "batch_seed": batch.batch_seed,
                "created_at": datetime.now().astimezone().isoformat(),
            }
            store.write(key, payload)
            generated += 1
        print(
            json.dumps(
                {
                    "stage": "generate",
                    "batch": ordinal,
                    "batches": len(batches),
                    "condition": batch.condition,
                    "decoding": batch.decoding,
                }
            ),
            flush=True,
        )
    return {"already_complete": completed, "newly_generated": generated}


def expected_generation_count(behavior_count: int) -> int:
    return behavior_count * len(CONDITIONS) * (len(SAMPLING_SEEDS) + 1)


def _clip_generation(tokenizer: Any, generation: str, max_tokens: int = 512) -> str:
    token_ids = tokenizer.encode(generation, max_length=max_tokens, truncation=True)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _classify_batch(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
) -> tuple[list[int], list[str], list[int]]:
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    input_length = int(encoded["input_ids"].shape[1])
    input_tokens = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    device = classifier_input_device(model)
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    raw = tokenizer.batch_decode(output_ids[:, input_length:], skip_special_tokens=True)
    labels = [parse_classifier_label(text) for text in raw]
    return labels, [text.strip() for text in raw], input_tokens


def run_classification_stage(
    model: nn.Module,
    tokenizer: Any,
    behaviors: list[HarmBenchBehavior],
    generations: RecordStore,
    classifications: RecordStore,
    *,
    batch_size: int,
) -> dict[str, int]:
    """Classify all generated records with the official one-token prompt."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    behavior_by_id = {item.behavior_id: item for item in behaviors}
    records = list(generations.payloads())
    expected = expected_generation_count(len(behaviors))
    if len(records) != expected:
        raise ValueError(
            f"Expected {expected} generation records before classification, found {len(records)}"
        )
    records.sort(
        key=lambda item: (
            str(item["condition"]),
            str(item["decoding"]),
            -1 if item["seed"] is None else int(item["seed"]),
            str(item["behavior_id"]),
        )
    )

    completed = 0
    classified = 0
    pending: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for record in records:
        key = generation_key(
            str(record["behavior_id"]),
            str(record["condition"]),
            str(record["decoding"]),
            None if record["seed"] is None else int(record["seed"]),
        )
        if classifications.contains(key):
            completed += 1
            continue
        behavior = behavior_by_id[str(record["behavior_id"])]
        clipped = _clip_generation(tokenizer, str(record["generation"]))
        prompt = render_classifier_prompt(behavior.behavior, clipped)
        pending.append((key, record, prompt))

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        labels, raw_outputs, input_tokens = _classify_batch(
            model, tokenizer, [item[2] for item in batch]
        )
        for (key, generation, _prompt), label, raw, token_count in zip(
            batch, labels, raw_outputs, input_tokens, strict=True
        ):
            payload = {
                **key,
                "semantic_category": generation["semantic_category"],
                "generation_sha256": generation["generation_sha256"],
                "label": label,
                "classifier_output": raw,
                "classifier_input_tokens": token_count,
                "created_at": datetime.now().astimezone().isoformat(),
            }
            classifications.write(key, payload)
            classified += 1
        print(
            json.dumps(
                {
                    "stage": "classify",
                    "classified": min(start + len(batch), len(pending)),
                    "pending": len(pending),
                }
            ),
            flush=True,
        )
    return {"already_complete": completed, "newly_classified": classified}
