"""Independent residual-stream addition steering with projection telemetry."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn


def _locate_decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    paths = (
        "model.layers",  # Qwen, Llama, Gemma
        "transformer.h",
        "language_model.layers",
        "gpt_neox.layers",
    )
    for path in paths:
        current: object = model
        for component in path.split("."):
            if not hasattr(current, component):
                break
            current = getattr(current, component)
        else:
            if isinstance(current, Sequence | nn.ModuleList):
                return current
    raise ValueError("Could not locate decoder layers")


def _tensor_from_output(output: object) -> tuple[Tensor, Literal["tensor", "tuple", "list"]]:
    if isinstance(output, Tensor):
        return output, "tensor"
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0], "tuple"
    if isinstance(output, list) and output and isinstance(output[0], Tensor):
        return output[0], "list"
    raise TypeError(f"Unsupported decoder output type: {type(output)}")


def _replace_tensor(
    output: object,
    modified: Tensor,
    container: Literal["tensor", "tuple", "list"],
) -> object:
    if container == "tensor":
        return modified
    if container == "tuple":
        assert isinstance(output, tuple)
        return (modified, *output[1:])
    assert isinstance(output, list)
    return [modified, *output[1:]]


def normalize_direction_float32(direction: Tensor) -> tuple[Tensor, float]:
    """Validate and normalize a one-dimensional direction in float32."""
    if direction.ndim != 1:
        raise ValueError("Addition direction must be rank 1")
    value = direction.detach().to(device="cpu", dtype=torch.float32).clone()
    if not torch.isfinite(value).all():
        raise ValueError("Addition direction contains NaN or Inf")
    raw_norm = float(torch.linalg.vector_norm(value).item())
    if not math.isfinite(raw_norm) or raw_norm == 0.0:
        raise ValueError("Addition direction must have a finite, non-zero norm")
    return value / raw_norm, raw_norm


@dataclass
class ProjectionStats:
    """Aggregated pre/post projection telemetry for one hook context."""

    expected_shift: float
    tolerance: float
    calls: int = 0
    total_positions: int = 0
    valid_positions: int = 0
    modified_positions: int = 0
    finite_positions: int = 0
    within_tolerance_positions: int = 0
    projection_before_sum: float = 0.0
    projection_after_sum: float = 0.0
    projection_shift_sum: float = 0.0
    projection_before_min: float = math.inf
    projection_before_max: float = -math.inf
    projection_after_min: float = math.inf
    projection_after_max: float = -math.inf
    projection_shift_min: float = math.inf
    projection_shift_max: float = -math.inf
    max_abs_shift_error: float = 0.0

    def update(
        self,
        before: Tensor,
        after: Tensor,
        *,
        valid_mask: Tensor | None,
        modified_positions: int,
    ) -> None:
        if before.shape != after.shape:
            raise ValueError("Before/after projection shapes differ")
        before_values = before.reshape(-1)
        after_values = after.reshape(-1)
        self.total_positions += before_values.numel()
        self.modified_positions += modified_positions
        if valid_mask is not None:
            mask = valid_mask.to(device=before.device, dtype=torch.bool).reshape(-1)
            if mask.numel() != before_values.numel():
                raise ValueError("Projection mask does not match token positions")
            before_values = before_values[mask]
            after_values = after_values[mask]
        if before_values.numel() == 0:
            raise ValueError("Projection telemetry has no valid positions")
        shift = after_values - before_values
        finite = (
            torch.isfinite(before_values)
            & torch.isfinite(after_values)
            & torch.isfinite(shift)
        )
        error = torch.abs(shift - self.expected_shift)
        self.calls += 1
        self.valid_positions += before_values.numel()
        self.finite_positions += int(torch.count_nonzero(finite).item())
        self.within_tolerance_positions += int(
            torch.count_nonzero(finite & (error <= self.tolerance)).item()
        )
        self.projection_before_sum += float(before_values.sum().item())
        self.projection_after_sum += float(after_values.sum().item())
        self.projection_shift_sum += float(shift.sum().item())
        self.projection_before_min = min(
            self.projection_before_min, float(before_values.min().item())
        )
        self.projection_before_max = max(
            self.projection_before_max, float(before_values.max().item())
        )
        self.projection_after_min = min(
            self.projection_after_min, float(after_values.min().item())
        )
        self.projection_after_max = max(
            self.projection_after_max, float(after_values.max().item())
        )
        self.projection_shift_min = min(self.projection_shift_min, float(shift.min().item()))
        self.projection_shift_max = max(self.projection_shift_max, float(shift.max().item()))
        self.max_abs_shift_error = max(self.max_abs_shift_error, float(error.max().item()))

    def materialize(self) -> dict[str, Any]:
        valid = self.valid_positions
        return {
            "calls": self.calls,
            "total_positions": self.total_positions,
            "valid_positions": valid,
            "modified_positions": self.modified_positions,
            "finite_positions": self.finite_positions,
            "finite": self.finite_positions == valid and valid > 0,
            "expected_shift": self.expected_shift,
            "tolerance": self.tolerance,
            "within_tolerance_positions": self.within_tolerance_positions,
            "within_tolerance_fraction": (
                self.within_tolerance_positions / valid if valid else None
            ),
            "projection_before": self.projection_before_sum / valid if valid else None,
            "projection_after": self.projection_after_sum / valid if valid else None,
            "projection_shift": self.projection_shift_sum / valid if valid else None,
            "projection_before_min": self.projection_before_min if valid else None,
            "projection_before_max": self.projection_before_max if valid else None,
            "projection_after_min": self.projection_after_min if valid else None,
            "projection_after_max": self.projection_after_max if valid else None,
            "projection_shift_min": self.projection_shift_min if valid else None,
            "projection_shift_max": self.projection_shift_max if valid else None,
            "max_abs_shift_error": self.max_abs_shift_error if valid else None,
        }


class _ProjectionHookBase:
    def __init__(
        self,
        model: nn.Module,
        direction: Tensor,
        *,
        layer: int,
        positions: Literal["all", "last"],
        position_mask: Tensor | None,
        expected_shift: float,
    ) -> None:
        if positions not in {"all", "last"}:
            raise ValueError("positions must be 'all' or 'last'")
        if not math.isfinite(expected_shift):
            raise ValueError("Expected projection shift must be finite")
        self.model = model
        self.layer = int(layer)
        self.positions = positions
        self.position_mask = (
            position_mask.detach().bool().cpu() if position_mask is not None else None
        )
        self.unit_direction, self.raw_direction_norm = normalize_direction_float32(direction)
        self.unit_direction_norm = float(torch.linalg.vector_norm(self.unit_direction).item())
        self.stats = ProjectionStats(
            expected_shift=expected_shift,
            tolerance=max(2.0, 0.02 * abs(expected_shift)),
        )
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._float_direction_cache: dict[torch.device, Tensor] = {}

    @property
    def active(self) -> bool:
        return bool(self._handles)

    def _float_direction(self, tensor: Tensor) -> Tensor:
        value = self._float_direction_cache.get(tensor.device)
        if value is None:
            value = self.unit_direction.to(device=tensor.device, dtype=torch.float32)
            self._float_direction_cache[tensor.device] = value
        if value.numel() != tensor.shape[-1]:
            raise ValueError(
                f"Addition direction hidden size {value.numel()} != activation {tensor.shape[-1]}"
            )
        return value

    def _selected(self, tensor: Tensor) -> Tensor:
        if self.positions == "all":
            return tensor
        if tensor.ndim < 2:
            raise ValueError("positions='last' requires a sequence dimension")
        return tensor[..., -1, :]

    def _selected_mask(self, tensor: Tensor) -> Tensor | None:
        if self.position_mask is None:
            return None
        mask = self.position_mask
        expected = tensor.shape[:-1]
        if tuple(mask.shape) != tuple(expected):
            raise ValueError(
                f"Position mask {tuple(mask.shape)} does not match "
                f"activation positions {tuple(expected)}"
            )
        return mask if self.positions == "all" else mask[..., -1]

    def __enter__(self):
        if self.active:
            raise RuntimeError("Projection hook is already active")
        layers = _locate_decoder_layers(self.model)
        if not 0 <= self.layer < len(layers):
            raise IndexError(f"Layer {self.layer} outside [0, {len(layers)})")
        self._handles.append(layers[self.layer].register_forward_hook(self._hook()))
        return self

    def __exit__(self, *_exc: object) -> None:
        self.remove()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._float_direction_cache.clear()

    def _hook(self):
        raise NotImplementedError


class ActivationAddition(_ProjectionHookBase):
    """Add ``alpha * unit_direction`` to one decoder block output."""

    def __init__(
        self,
        model: nn.Module,
        direction: Tensor,
        alpha: float,
        *,
        layer: int,
        positions: Literal["all", "last"] = "all",
        position_mask: Tensor | None = None,
    ) -> None:
        if not math.isfinite(alpha):
            raise ValueError("Addition alpha must be finite")
        self.alpha = float(alpha)
        super().__init__(
            model,
            direction,
            layer=layer,
            positions=positions,
            position_mask=position_mask,
            expected_shift=self.alpha,
        )
        self._activation_direction_cache: dict[tuple[torch.device, torch.dtype], Tensor] = {}

    def _activation_direction(self, tensor: Tensor) -> Tensor:
        key = (tensor.device, tensor.dtype)
        value = self._activation_direction_cache.get(key)
        if value is None:
            value = self.unit_direction.to(device=tensor.device, dtype=tensor.dtype)
            self._activation_direction_cache[key] = value
        if value.numel() != tensor.shape[-1]:
            raise ValueError(
                f"Addition direction hidden size {value.numel()} != activation {tensor.shape[-1]}"
            )
        return value

    def _hook(self):
        def add(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
            tensor, container = _tensor_from_output(output)
            selected = self._selected(tensor)
            direction = self._activation_direction(tensor)
            if self.positions == "all":
                modified = tensor + self.alpha * direction
                modified_selected = modified
                modified_positions = tensor[..., 0].numel()
            else:
                modified = tensor.clone()
                modified[..., -1, :] = selected + self.alpha * direction
                modified_selected = modified[..., -1, :]
                modified_positions = selected[..., 0].numel()
            float_direction = self._float_direction(tensor)
            before_projection = torch.einsum(
                "...d,d->...", selected.detach().float(), float_direction
            )
            after_projection = torch.einsum(
                "...d,d->...", modified_selected.detach().float(), float_direction
            )
            self.stats.update(
                before_projection,
                after_projection,
                valid_mask=self._selected_mask(tensor),
                modified_positions=modified_positions,
            )
            if not torch.isfinite(modified).all():
                raise FloatingPointError("Addition produced NaN or Inf activations")
            return _replace_tensor(output, modified, container)

        return add

    def remove(self) -> None:
        super().remove()
        self._activation_direction_cache.clear()


class ActivationProjectionMonitor(_ProjectionHookBase):
    """Read-only layer projection monitor for the true no-hook control."""

    def __init__(
        self,
        model: nn.Module,
        direction: Tensor,
        *,
        layer: int,
        positions: Literal["all", "last"] = "all",
        position_mask: Tensor | None = None,
    ) -> None:
        super().__init__(
            model,
            direction,
            layer=layer,
            positions=positions,
            position_mask=position_mask,
            expected_shift=0.0,
        )

    def _hook(self):
        def monitor(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            tensor, _container = _tensor_from_output(output)
            selected = self._selected(tensor)
            float_direction = self._float_direction(tensor)
            projection = torch.einsum("...d,d->...", selected.detach().float(), float_direction)
            self.stats.update(
                projection,
                projection,
                valid_mask=self._selected_mask(tensor),
                modified_positions=0,
            )
            if not torch.isfinite(tensor).all():
                raise FloatingPointError("Projection monitor observed NaN or Inf activations")
            return None

        return monitor
