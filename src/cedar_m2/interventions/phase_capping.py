"""Phase-gated activation capping for decoder-only generation.

The formal E2 runner keeps the historical :class:`ActivationCapper` path for
``full_p25``.  This module is used only for ``prefill_only``, ``decode_only``,
and read-only monitoring.  A forward with sequence length greater than one is
classified as prefill; a single-position forward is classified as decode.
The E2 prompts are all longer than one token, and batch size is frozen to one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from cedar_m2.interventions.capping import (
    LayerCap,
    _apply_unit_upper_cap,
    _normalized_direction,
)

Phase = Literal["prefill", "decode"]


@dataclass
class PhaseStats:
    """Telemetry for one layer and one generation phase."""

    forward_calls: int = 0
    observed_positions: int = 0
    applied_positions: int = 0
    triggered_positions: int = 0
    projection_correction_sum: float = 0.0

    @property
    def trigger_rate(self) -> float:
        if not self.applied_positions:
            return 0.0
        return self.triggered_positions / self.applied_positions

    @property
    def mean_projection_correction(self) -> float:
        if not self.applied_positions:
            return 0.0
        return self.projection_correction_sum / self.applied_positions


@dataclass
class _DeferredStats:
    forward_calls: int = 0
    observed_positions: int = 0
    applied_positions: int = 0
    triggered_positions: Tensor | None = None
    projection_correction_sum: Tensor | None = None


class PhaseActivationCapper:
    """Install layer hooks that cap only frozen prefill/decode phases.

    ``enabled_phases`` may be empty to count phase calls only.  Setting
    ``apply_changes=False`` with both phases enabled computes would-be cap dose
    without modifying activations.  Telemetry always counts observed phases,
    which lets the audit prove that an inactive phase was seen but not modified.
    """

    _LAYER_PATHS: Sequence[str] = (
        "model.layers",
        "transformer.h",
        "language_model.layers",
        "gpt_neox.layers",
    )

    def __init__(
        self,
        model: nn.Module,
        caps: Mapping[int, LayerCap],
        *,
        enabled_phases: Sequence[Phase],
        apply_changes: bool = True,
        defer_stats_sync: bool = False,
    ) -> None:
        if not caps:
            raise ValueError("At least one layer cap is required")
        phases = frozenset(enabled_phases)
        if not phases.issubset({"prefill", "decode"}):
            raise ValueError("enabled_phases must contain only prefill/decode")
        self.model = model
        self.caps = dict(caps)
        self.enabled_phases = phases
        self.apply_changes = apply_changes
        self.defer_stats_sync = defer_stats_sync
        self.stats = {
            layer: {phase: PhaseStats() for phase in ("prefill", "decode")} for layer in self.caps
        }
        self._deferred = {
            layer: {phase: _DeferredStats() for phase in ("prefill", "decode")}
            for layer in self.caps
        }
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._direction_cache: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}
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

    @staticmethod
    def _tensor(output: object, layer: int) -> tuple[Tensor, str]:
        if isinstance(output, Tensor):
            return output, "tensor"
        if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
            return output[0], "tuple"
        if isinstance(output, list) and output and isinstance(output[0], Tensor):
            return output[0], "list"
        raise TypeError(f"Unsupported decoder output type at layer {layer}: {type(output)}")

    @staticmethod
    def _replace(output: object, modified: Tensor, container: str) -> object:
        if container == "tensor":
            return modified
        if container == "tuple":
            assert isinstance(output, tuple)
            return (modified, *output[1:])
        assert isinstance(output, list)
        return [modified, *output[1:]]

    @staticmethod
    def _phase(tensor: Tensor) -> Phase:
        if tensor.ndim < 3:
            raise ValueError("E2 phase detection requires [batch, sequence, hidden] activations")
        return "prefill" if int(tensor.shape[-2]) > 1 else "decode"

    def _record(
        self,
        layer: int,
        phase: Phase,
        tensor: Tensor,
        excess: Tensor | None,
    ) -> None:
        positions = tensor.numel() // tensor.shape[-1]
        if self.defer_stats_sync:
            item = self._deferred[layer][phase]
            item.forward_calls += 1
            item.observed_positions += positions
            if excess is not None:
                item.applied_positions += excess.numel()
                triggered = torch.count_nonzero(excess.detach() > 0)
                correction = excess.detach().float().sum()
                if item.triggered_positions is None:
                    item.triggered_positions = triggered
                    item.projection_correction_sum = correction
                else:
                    item.triggered_positions.add_(triggered)
                    assert item.projection_correction_sum is not None
                    item.projection_correction_sum.add_(correction)
            return
        item = self.stats[layer][phase]
        item.forward_calls += 1
        item.observed_positions += positions
        if excess is not None:
            detached = excess.detach().float()
            item.applied_positions += detached.numel()
            item.triggered_positions += int(torch.count_nonzero(detached > 0).item())
            item.projection_correction_sum += float(detached.sum().item())

    def _hook(self, layer: int):
        def apply_cap(_module: nn.Module, _inputs: tuple[object, ...], output: object):
            tensor, container = self._tensor(output, layer)
            phase = self._phase(tensor)
            if phase not in self.enabled_phases:
                self._record(layer, phase, tensor, None)
                return output
            cap = self.caps[layer]
            cache_key = (layer, tensor.device, tensor.dtype)
            vector = self._direction_cache.get(cache_key)
            if vector is None:
                vector = _normalized_direction(tensor, cap.direction)
                self._direction_cache[cache_key] = vector
            modified, excess = _apply_unit_upper_cap(tensor, vector, cap.threshold, "all")
            self._record(layer, phase, tensor, excess)
            if not self.apply_changes:
                return output
            return self._replace(output, modified, container)

        return apply_cap

    def __enter__(self) -> PhaseActivationCapper:
        if self._handles:
            raise RuntimeError("PhaseActivationCapper is already active")
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

    def _materialize_deferred(self) -> None:
        if not self.defer_stats_sync:
            return
        for layer, phases in self._deferred.items():
            for phase, deferred in phases.items():
                item = self.stats[layer][phase]
                item.forward_calls += deferred.forward_calls
                item.observed_positions += deferred.observed_positions
                item.applied_positions += deferred.applied_positions
                if deferred.triggered_positions is not None:
                    item.triggered_positions += int(deferred.triggered_positions.item())
                if deferred.projection_correction_sum is not None:
                    item.projection_correction_sum += float(
                        deferred.projection_correction_sum.item()
                    )
        self._deferred = {
            layer: {phase: _DeferredStats() for phase in ("prefill", "decode")}
            for layer in self.caps
        }

    def remove(self) -> None:
        self._materialize_deferred()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._direction_cache.clear()

    def telemetry(self) -> dict[str, object]:
        if self._handles and self.defer_stats_sync:
            raise RuntimeError("Deferred telemetry is available only after hooks are removed")
        return {
            "enabled_phases": sorted(self.enabled_phases),
            "apply_changes": self.apply_changes,
            "by_layer": {
                str(layer): {
                    phase: {
                        **asdict(item),
                        "trigger_rate": item.trigger_rate,
                        "mean_projection_correction": item.mean_projection_correction,
                    }
                    for phase, item in phases.items()
                }
                for layer, phases in sorted(self.stats.items())
            },
        }
