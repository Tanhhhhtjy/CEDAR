"""Pinned local loading for the official HarmBench text classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class HarmBenchClassifierSpec:
    repo_id: str = "cais/HarmBench-Llama-2-13b-cls"
    revision: str = "bda705349d1144fa618770bea64d99ce54e3835b"
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"


def load_harmbench_classifier(
    spec: HarmBenchClassifierSpec | None = None,
    *,
    device_map: str | dict[str, int | str] = "auto",
    max_memory: dict[int | str, str] | None = None,
    local_files_only: bool = False,
) -> tuple[nn.Module, Any]:
    """Load the pinned classifier with Transformers rather than vLLM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cedar_m2.models.qwen import resolve_dtype

    spec = spec or HarmBenchClassifierSpec()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repo_id,
        revision=spec.revision,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.repo_id,
        revision=spec.revision,
        dtype=resolve_dtype(spec.dtype),
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        attn_implementation=spec.attention_implementation,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    model.eval()
    return model, tokenizer


def classifier_input_device(model: nn.Module) -> torch.device:
    """Return the classifier token-embedding device."""
    from cedar_m2.models.qwen import model_input_device

    return model_input_device(model)
