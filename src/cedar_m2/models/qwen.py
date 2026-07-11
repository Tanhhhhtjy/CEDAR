"""Pinned Qwen model loading for the M2 pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class QwenModelSpec:
    repo_id: str = "Qwen/Qwen3-32B"
    revision: str = "9216db5781bf21249d130ec9da846c4624c16137"
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve the small set of dtypes allowed by the preregistered pipeline."""
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return dtypes[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(dtypes)}") from error


def model_input_device(model: nn.Module) -> torch.device:
    """Return the device hosting token embeddings in a sharded causal LM."""
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("Cannot infer input device from a parameterless model") from error


def load_qwen_model(
    spec: QwenModelSpec | None = None,
    *,
    device_map: str | dict[str, int | str] = "auto",
    max_memory: dict[int | str, str] | None = None,
) -> tuple[nn.Module, Any]:
    """Load the pinned, unquantized Qwen model and tokenizer.

    Imports Transformers lazily so data preparation and CPU unit tests do not
    initialize CUDA or require the full inference environment.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = spec or QwenModelSpec()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repo_id,
        revision=spec.revision,
        trust_remote_code=False,
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
    )
    model.eval()
    return model, tokenizer
