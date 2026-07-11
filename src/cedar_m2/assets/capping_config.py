"""Load and validate the official Assistant Axis capping assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from cedar_m2.interventions.capping import LayerCap

ASSISTANT_AXIS_REPO = "lu-christina/assistant-axis-vectors"
ASSISTANT_AXIS_REVISION = "3b3b788432ad33e3a28d9ff08e88a530c0740814"
QWEN_AXIS_FILE = "qwen-3-32b/assistant_axis.pt"
QWEN_CAPPING_CONFIG_FILE = "qwen-3-32b/capping_config.pt"


def download_qwen_capping_assets(*, cache_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Download the pinned Qwen Assistant Axis and capping config."""
    from huggingface_hub import hf_hub_download

    kwargs = {
        "repo_id": ASSISTANT_AXIS_REPO,
        "repo_type": "dataset",
        "revision": ASSISTANT_AXIS_REVISION,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
    }
    axis_path = hf_hub_download(filename=QWEN_AXIS_FILE, **kwargs)
    config_path = hf_hub_download(filename=QWEN_CAPPING_CONFIG_FILE, **kwargs)
    return Path(axis_path), Path(config_path)


def _load_weights(path: str | Path) -> Any:
    """Load tensor-only upstream artifacts without enabling pickle execution."""
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def load_axis(path: str | Path) -> Tensor:
    """Load an Assistant Axis tensor with shape ``[layers, hidden_size]``."""
    axis = _load_weights(path)
    if not isinstance(axis, Tensor) or axis.ndim != 2:
        raise ValueError("Assistant Axis must be a rank-2 tensor [layers, hidden_size]")
    if not torch.isfinite(axis).all():
        raise ValueError("Assistant Axis contains non-finite values")
    return axis


def load_capping_experiment(path: str | Path, experiment_id: str) -> dict[int, LayerCap]:
    """Parse one experiment from the official ``capping_config.pt`` format."""
    config = _load_weights(path)
    if not isinstance(config, dict) or not {"vectors", "experiments"} <= config.keys():
        raise ValueError("Capping config must contain 'vectors' and 'experiments'")

    experiments = [item for item in config["experiments"] if item.get("id") == experiment_id]
    if len(experiments) != 1:
        raise ValueError(
            f"Expected exactly one experiment named {experiment_id!r}, found {len(experiments)}"
        )

    caps: dict[int, LayerCap] = {}
    for intervention in experiments[0].get("interventions", []):
        vector_name = intervention.get("vector")
        if vector_name not in config["vectors"]:
            raise ValueError(f"Unknown capping vector {vector_name!r}")

        vector_record = config["vectors"][vector_name]
        if not isinstance(vector_record, dict) or not {"vector", "layer"} <= vector_record.keys():
            raise ValueError(f"Malformed vector record {vector_name!r}")

        vector = vector_record["vector"]
        layer = int(vector_record["layer"])
        threshold = float(intervention["cap"])
        if not isinstance(vector, Tensor) or vector.ndim != 1:
            raise ValueError(f"Capping vector {vector_name!r} must be rank 1")
        if layer in caps:
            raise ValueError(f"Experiment {experiment_id!r} contains duplicate layer {layer}")
        if not torch.isfinite(vector).all() or not torch.isfinite(torch.tensor(threshold)):
            raise ValueError(f"Non-finite vector or threshold at layer {layer}")

        caps[layer] = LayerCap(layer=layer, direction=vector.detach(), threshold=threshold)

    if not caps:
        raise ValueError(f"Experiment {experiment_id!r} contains no interventions")
    return caps


def verify_axis_orientation(
    caps: dict[int, LayerCap],
    assistant_axis: Tensor,
    *,
    maximum_cosine: float = -0.999,
) -> dict[int, float]:
    """Verify that official cap vectors point opposite to the Assistant Axis.

    The published Qwen capping configuration uses an upper cap along vectors
    that are the negative of the human-readable Assistant Axis. A sign error
    here reverses the intervention, so validation is intentionally strict.
    """
    if assistant_axis.ndim != 2:
        raise ValueError("Assistant Axis must be rank 2")

    similarities: dict[int, float] = {}
    for layer, cap in caps.items():
        if not 0 <= layer < assistant_axis.shape[0]:
            raise ValueError(f"Layer {layer} is outside Assistant Axis bounds")
        if cap.direction.numel() != assistant_axis.shape[1]:
            raise ValueError(f"Hidden-size mismatch at layer {layer}")

        cosine = float(
            F.cosine_similarity(cap.direction.float(), assistant_axis[layer].float(), dim=0).item()
        )
        similarities[layer] = cosine
        if cosine > maximum_cosine:
            raise ValueError(
                f"Capping vector at layer {layer} has cosine {cosine:.6f}; "
                f"expected <= {maximum_cosine:.6f}"
            )
    return similarities
