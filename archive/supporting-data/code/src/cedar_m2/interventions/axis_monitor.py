"""Read-only Assistant Axis telemetry for causal diagnostic runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from cedar_m2.interventions.capping import LayerCap


@dataclass
class _Moments:
    count: int = 0
    total: Tensor | None = None
    total_sq: Tensor | None = None
    minimum: Tensor | None = None
    maximum: Tensor | None = None

    def add(self, values: Tensor) -> None:
        values = values.detach().float().reshape(-1)
        if not values.numel():
            return
        self.count += values.numel()
        batch_total = values.sum()
        batch_total_sq = torch.square(values).sum()
        batch_min = values.min()
        batch_max = values.max()
        if self.total is None:
            self.total = batch_total
            self.total_sq = batch_total_sq
            self.minimum = batch_min
            self.maximum = batch_max
        else:
            self.total.add_(batch_total)
            assert (
                self.total_sq is not None
                and self.minimum is not None
                and self.maximum is not None
            )
            self.total_sq.add_(batch_total_sq)
            self.minimum.copy_(torch.minimum(self.minimum, batch_min))
            self.maximum.copy_(torch.maximum(self.maximum, batch_max))

    def materialize(self) -> dict[str, float | int]:
        if not self.count or self.total is None:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        assert (
            self.total_sq is not None
            and self.minimum is not None
            and self.maximum is not None
        )
        total = float(self.total.item())
        total_sq = float(self.total_sq.item())
        mean = total / self.count
        variance = max(total_sq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": variance**0.5,
            "min": float(self.minimum.item()),
            "max": float(self.maximum.item()),
        }


@dataclass
class _Exposure:
    count: int = 0
    triggered: Tensor | None = None
    correction: Tensor | None = None

    def add(self, excess: Tensor) -> None:
        excess = excess.detach().float().reshape(-1)
        if not excess.numel():
            return
        self.count += excess.numel()
        triggered = torch.count_nonzero(excess > 0)
        correction = excess.sum()
        if self.triggered is None:
            self.triggered = triggered
            self.correction = correction
        else:
            self.triggered.add_(triggered)
            assert self.correction is not None
            self.correction.add_(correction)

    def materialize(self) -> dict[str, float | int]:
        if not self.count or self.triggered is None or self.correction is None:
            return {
                "count": 0,
                "triggered": 0,
                "trigger_rate": 0.0,
                "correction_sum": 0.0,
                "mean_correction_all_positions": 0.0,
                "mean_correction_triggered_positions": 0.0,
                "mean_correction": 0.0,
            }
        triggered = int(self.triggered.item())
        correction = float(self.correction.item())
        mean_all = correction / self.count
        return {
            "count": self.count,
            "triggered": triggered,
            "trigger_rate": triggered / self.count,
            "correction_sum": correction,
            "mean_correction_all_positions": mean_all,
            "mean_correction_triggered_positions": (
                correction / triggered if triggered else 0.0
            ),
            # Backward-compatible alias used by frozen v6 artifacts. It has always
            # meant the all-position mean, never the triggered-position mean.
            "mean_correction": mean_all,
        }


class AssistantAxisMonitor:
    """Measure pre-cap axis projections and counterfactual cap exposure.

    The monitor must be entered before ``ActivationCapper`` so its forward hooks
    observe each layer output before the cap at that layer. The first call at a
    layer is treated as prompt prefill; later cache calls are treated as decode.
    A batch attention mask removes left-padding from prefill telemetry.
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
        assistant_axis: Tensor,
        *,
        layers: Sequence[int],
        prefill_attention_mask: Tensor,
        reference_caps: Mapping[str, Mapping[int, LayerCap]] | None = None,
    ) -> None:
        if assistant_axis.ndim != 2:
            raise ValueError("assistant_axis must have shape [layers, hidden]")
        self.model = model
        self.axis = assistant_axis
        self.layers = tuple(sorted(set(int(layer) for layer in layers)))
        self.prefill_attention_mask = prefill_attention_mask.detach().bool().cpu()
        self.reference_caps = {
            label: dict(caps) for label, caps in (reference_caps or {}).items()
        }
        self._handles: list[Any] = []
        self._calls = {layer: 0 for layer in self.layers}
        self._axis_cache: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}
        self._cap_cache: dict[tuple[str, int, torch.device, torch.dtype], Tensor] = {}
        self._moments = {
            phase: {layer: _Moments() for layer in self.layers}
            for phase in ("prefill", "decode")
        }
        self._exposure = {
            phase: {
                label: {layer: _Exposure() for layer in self.layers}
                for label in self.reference_caps
            }
            for phase in ("prefill", "decode")
        }

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
        raise ValueError("Could not locate decoder layers")

    @staticmethod
    def _tensor_from_output(output: object) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
            return output[0]
        raise TypeError(f"Unsupported decoder output type: {type(output)}")

    @staticmethod
    def _unit(vector: Tensor, tensor: Tensor) -> Tensor:
        value = vector.to(device=tensor.device, dtype=tensor.dtype)
        return value / torch.linalg.vector_norm(value)

    def _select(self, values: Tensor, phase: str) -> Tensor:
        if phase == "decode":
            return values.reshape(-1)
        mask = self.prefill_attention_mask.to(values.device)
        if tuple(mask.shape) != tuple(values.shape):
            raise ValueError(
                f"Prefill mask {tuple(mask.shape)} does not match projection {tuple(values.shape)}"
            )
        return values[mask]

    def _hook(self, layer: int):
        def monitor(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            tensor = self._tensor_from_output(output)
            if tensor.ndim != 3:
                raise ValueError(f"Expected [batch, sequence, hidden] at layer {layer}")
            phase = "prefill" if self._calls[layer] == 0 else "decode"
            self._calls[layer] += 1

            axis_key = (layer, tensor.device, tensor.dtype)
            axis_vector = self._axis_cache.get(axis_key)
            if axis_vector is None:
                axis_vector = self._unit(self.axis[layer], tensor)
                self._axis_cache[axis_key] = axis_vector
            projections = torch.einsum("bsd,d->bs", tensor, axis_vector)
            self._moments[phase][layer].add(self._select(projections, phase))

            for label, caps in self.reference_caps.items():
                if layer not in caps:
                    continue
                cap = caps[layer]
                cap_key = (label, layer, tensor.device, tensor.dtype)
                cap_vector = self._cap_cache.get(cap_key)
                if cap_vector is None:
                    cap_vector = self._unit(cap.direction, tensor)
                    self._cap_cache[cap_key] = cap_vector
                cap_projection = torch.einsum("bsd,d->bs", tensor, cap_vector)
                excess = torch.clamp(cap_projection - cap.threshold, min=0.0)
                self._exposure[phase][label][layer].add(self._select(excess, phase))

        return monitor

    def __enter__(self) -> AssistantAxisMonitor:
        decoder_layers = self._locate_layers()
        try:
            for layer in self.layers:
                if not 0 <= layer < len(decoder_layers):
                    raise IndexError(f"Layer {layer} outside [0, {len(decoder_layers)})")
                self._handles.append(decoder_layers[layer].register_forward_hook(self._hook(layer)))
        except Exception:
            self.remove()
            raise
        return self

    def __exit__(self, *_exc: object) -> None:
        self.remove()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def materialize(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "definition": (
                "pre-cap decoder-block outputs; prefill excludes attention-masked positions; "
                "decode contains cache steps; batch size 1 is required by the v6 contract"
            ),
            "phases": {},
        }
        for phase in ("prefill", "decode"):
            report["phases"][phase] = {
                "assistant_axis_projection": {
                    str(layer): self._moments[phase][layer].materialize()
                    for layer in self.layers
                },
                "reference_cap_exposure": {
                    label: {
                        str(layer): self._exposure[phase][label][layer].materialize()
                        for layer in self.layers
                    }
                    for label in self.reference_caps
                },
            }
        return report
