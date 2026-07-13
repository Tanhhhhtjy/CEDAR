"""Minimal multi-layer activation capping for Transformers decoder models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LayerCap:
    """One upper cap applied along a direction at one decoder layer."""

    layer: int
    direction: Tensor
    threshold: float


@dataclass
class CapStats:
    """Aggregated intervention dose for one layer."""

    total_positions: int = 0
    triggered_positions: int = 0
    projection_correction_sum: float = 0.0

    @property
    def trigger_rate(self) -> float:
        return self.triggered_positions / self.total_positions if self.total_positions else 0.0

    @property
    def mean_projection_correction(self) -> float:
        if not self.total_positions:
            return 0.0
        return self.projection_correction_sum / self.total_positions

    def update(self, other: CapStats) -> None:
        self.total_positions += other.total_positions
        self.triggered_positions += other.triggered_positions
        self.projection_correction_sum += other.projection_correction_sum


def _normalized_direction(activations: Tensor, direction: Tensor) -> Tensor:
    if activations.ndim < 1:
        raise ValueError("Activations must have a hidden dimension")
    if direction.ndim != 1 or direction.numel() != activations.shape[-1]:
        raise ValueError("Direction must be rank 1 and match activation hidden size")
    vector = direction.to(device=activations.device, dtype=activations.dtype)
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm.detach()) == 0.0:
        raise ValueError("Direction must have a finite, non-zero norm")
    return vector / norm


def _apply_unit_upper_cap(
    activations: Tensor,
    vector: Tensor,
    threshold: float,
    positions: Literal["all", "last"],
) -> tuple[Tensor, Tensor]:
    """Apply an upper cap with a validated unit vector and return its excess."""
    if positions not in {"all", "last"}:
        raise ValueError("positions must be 'all' or 'last'")
    if positions == "last" and activations.ndim < 2:
        raise ValueError("positions='last' requires a sequence dimension")
    if not torch.isfinite(torch.tensor(threshold)):
        raise ValueError("Threshold must be finite")

    selected = activations if positions == "all" else activations[..., -1, :]
    projections = torch.einsum("...d,d->...", selected, vector)
    excess = torch.clamp(projections - threshold, min=0.0)
    corrected = selected - excess.unsqueeze(-1) * vector

    if positions == "all":
        output = corrected
    else:
        output = activations.clone()
        output[..., -1, :] = corrected
    return output, excess


def _stats_from_excess(excess: Tensor) -> CapStats:
    detached_excess = excess.detach().float()
    return CapStats(
        total_positions=detached_excess.numel(),
        triggered_positions=int(torch.count_nonzero(detached_excess > 0).item()),
        projection_correction_sum=float(detached_excess.sum().item()),
    )


def apply_upper_cap(
    activations: Tensor,
    direction: Tensor,
    threshold: float,
    *,
    positions: Literal["all", "last"] = "all",
) -> tuple[Tensor, CapStats]:
    """Apply ``h <- h - v * max(<h,v> - threshold, 0)``.

    The final dimension is interpreted as hidden size. ``positions='last'``
    only changes the final sequence position; ``positions='all'`` changes all
    leading positions. The implementation is out-of-place to avoid corrupting
    cached tensors retained by Transformers.
    """
    vector = _normalized_direction(activations, direction)
    output, excess = _apply_unit_upper_cap(activations, vector, threshold, positions)
    return output, _stats_from_excess(excess)


class ActivationCapper:
    """Context manager that installs upper-cap hooks on decoder blocks."""

    _LAYER_PATHS: Sequence[str] = (
        "model.layers",  # Qwen, Llama, Gemma
        "transformer.h",
        "language_model.layers",
        "gpt_neox.layers",
    )

    def __init__(
        self,
        model: nn.Module,
        caps: Mapping[int, LayerCap],
        *,
        positions: Literal["all", "last"] = "all",
        track_stats: bool = True,
        defer_stats_sync: bool = False,
    ) -> None:
        if not caps:
            raise ValueError("At least one layer cap is required")
        self.model = model
        self.caps = dict(caps)
        self.positions = positions
        self.track_stats = track_stats
        self.defer_stats_sync = defer_stats_sync
        self.stats = {layer: CapStats() for layer in self.caps}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._direction_cache: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}
        self._deferred_total_positions = {layer: 0 for layer in self.caps}
        self._deferred_triggered: dict[int, Tensor] = {}
        self._deferred_correction: dict[int, Tensor] = {}

        for layer, cap in self.caps.items():
            if layer != cap.layer:
                raise ValueError(f"Cap key {layer} does not match LayerCap.layer {cap.layer}")

    def _locate_layers(self) -> Sequence[nn.Module]:
        for path in self._LAYER_PATHS:
            current: object = self.model
            for component in path.split("."):
                if not hasattr(current, component):
                    break
                current = getattr(current, component)
            else:
                if isinstance(current, Sequence | nn.ModuleList):
                    return current
        raise ValueError("Could not locate decoder layers on model")

    def _hook(self, layer: int):
        def apply_cap(_module: nn.Module, _inputs: tuple[object, ...], output: object):
            if isinstance(output, Tensor):
                tensor = output
                container = "tensor"
            elif isinstance(output, tuple) and output and isinstance(output[0], Tensor):
                tensor = output[0]
                container = "tuple"
            elif isinstance(output, list) and output and isinstance(output[0], Tensor):
                tensor = output[0]
                container = "list"
            else:
                raise TypeError(f"Unsupported decoder output type at layer {layer}: {type(output)}")

            cap = self.caps[layer]
            cache_key = (layer, tensor.device, tensor.dtype)
            vector = self._direction_cache.get(cache_key)
            if vector is None:
                vector = _normalized_direction(tensor, cap.direction)
                self._direction_cache[cache_key] = vector
            modified, excess = _apply_unit_upper_cap(tensor, vector, cap.threshold, self.positions)
            if self.track_stats:
                if self.defer_stats_sync:
                    detached = excess.detach().float()
                    self._deferred_total_positions[layer] += detached.numel()
                    triggered = torch.count_nonzero(detached > 0)
                    correction = detached.sum()
                    if layer not in self._deferred_triggered:
                        self._deferred_triggered[layer] = triggered
                        self._deferred_correction[layer] = correction
                    else:
                        self._deferred_triggered[layer].add_(triggered)
                        self._deferred_correction[layer].add_(correction)
                else:
                    self.stats[layer].update(_stats_from_excess(excess))

            if container == "tensor":
                return modified
            if container == "tuple":
                return (modified, *output[1:])
            return [modified, *output[1:]]

        return apply_cap

    def __enter__(self) -> ActivationCapper:
        if self._handles:
            raise RuntimeError("ActivationCapper is already active")
        layers = self._locate_layers()
        try:
            for layer in sorted(self.caps):
                if not 0 <= layer < len(layers):
                    raise IndexError(f"Layer {layer} is outside model range [0, {len(layers)})")
                self._handles.append(layers[layer].register_forward_hook(self._hook(layer)))
        except Exception:
            self.remove()
            raise
        return self

    def __exit__(self, *_exc: object) -> None:
        self.remove()

    def reset_stats(self) -> None:
        self.stats = {layer: CapStats() for layer in self.caps}
        self._deferred_total_positions = {layer: 0 for layer in self.caps}
        self._deferred_triggered.clear()
        self._deferred_correction.clear()

    def _materialize_deferred_stats(self) -> None:
        if not self.defer_stats_sync:
            return
        for layer in self.caps:
            total = self._deferred_total_positions[layer]
            triggered = self._deferred_triggered.get(layer)
            correction = self._deferred_correction.get(layer)
            if not total:
                continue
            if triggered is None or correction is None:
                raise RuntimeError(f"Deferred capping stats are incomplete at layer {layer}")
            self.stats[layer].update(
                CapStats(
                    total_positions=total,
                    triggered_positions=int(triggered.item()),
                    projection_correction_sum=float(correction.item()),
                )
            )
        self._deferred_total_positions = {layer: 0 for layer in self.caps}
        self._deferred_triggered.clear()
        self._deferred_correction.clear()

    def remove(self) -> None:
        self._materialize_deferred_stats()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._direction_cache.clear()
