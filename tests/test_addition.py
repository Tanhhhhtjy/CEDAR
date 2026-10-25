from __future__ import annotations

import pytest
import torch
from torch import nn

from cedar_m2.interventions.addition import (
    ActivationAddition,
    ActivationProjectionMonitor,
    normalize_direction_float32,
)


class _PassLayer(nn.Module):
    def __init__(self, container: str = "tensor") -> None:
        super().__init__()
        self.container = container

    def forward(self, value: torch.Tensor):
        if self.container == "tuple":
            return (value, "cache")
        if self.container == "list":
            return [value, "cache"]
        return value


class _Inner(nn.Module):
    def __init__(self, container: str) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_PassLayer(), _PassLayer(container), _PassLayer()]
        )


class _DummyModel(nn.Module):
    def __init__(self, container: str = "tensor") -> None:
        super().__init__()
        self.model = _Inner(container)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            output = layer(value)
            value = output[0] if isinstance(output, tuple | list) else output
        return value


def test_float32_normalization_and_invalid_directions() -> None:
    unit, raw_norm = normalize_direction_float32(torch.tensor([3.0, 4.0]))
    assert raw_norm == 5.0
    assert unit.dtype == torch.float32
    assert torch.allclose(unit, torch.tensor([0.6, 0.8]))
    with pytest.raises(ValueError, match="rank 1"):
        normalize_direction_float32(torch.ones(2, 2))
    with pytest.raises(ValueError, match="non-zero"):
        normalize_direction_float32(torch.zeros(2))
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_direction_float32(torch.tensor([float("nan"), 1.0]))


@pytest.mark.parametrize("container", ["tensor", "tuple", "list"])
def test_addition_supports_decoder_output_containers_and_all_positions(container: str) -> None:
    model = _DummyModel(container)
    value = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    addition = ActivationAddition(
        model,
        torch.tensor([1.0, 0.0]),
        2.0,
        layer=1,
        positions="all",
        position_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    assert len(model.model.layers[0]._forward_hooks) == 0
    assert len(model.model.layers[2]._forward_hooks) == 0
    with addition:
        assert addition.active
        assert len(model.model.layers[1]._forward_hooks) == 1
        output = model(value)
    assert not addition.active
    assert len(model.model.layers[1]._forward_hooks) == 0
    assert torch.equal(output, value + torch.tensor([2.0, 0.0]))
    stats = addition.stats.materialize()
    assert stats["calls"] == 1
    assert stats["valid_positions"] == 3
    assert stats["modified_positions"] == 3
    assert stats["projection_shift"] == 2.0
    assert stats["within_tolerance_fraction"] == 1.0
    assert stats["finite"] is True


def test_alpha_zero_is_identity_and_positive_negative_shifts_have_correct_sign() -> None:
    model = _DummyModel()
    value = torch.randn(1, 4, 2)
    outputs = {}
    shifts = {}
    for alpha in (0.0, 7.0, -7.0):
        addition = ActivationAddition(
            model,
            torch.tensor([1.0, 1.0]),
            alpha,
            layer=1,
            positions="all",
        )
        with addition:
            outputs[alpha] = model(value)
        shifts[alpha] = addition.stats.materialize()["projection_shift"]
    assert torch.equal(outputs[0.0], value)
    assert shifts[0.0] == 0.0
    assert shifts[7.0] > 0
    assert shifts[-7.0] < 0


def test_last_position_mode_does_not_modify_earlier_positions() -> None:
    model = _DummyModel()
    value = torch.zeros(1, 3, 2)
    addition = ActivationAddition(
        model,
        torch.tensor([1.0, 0.0]),
        3.0,
        layer=1,
        positions="last",
    )
    with addition:
        output = model(value)
    assert torch.equal(output[:, :2], value[:, :2])
    assert torch.equal(output[:, -1], torch.tensor([[3.0, 0.0]]))
    assert addition.stats.materialize()["modified_positions"] == 1


def test_projection_monitor_is_read_only_and_removes_hook() -> None:
    model = _DummyModel()
    value = torch.randn(1, 3, 2)
    monitor = ActivationProjectionMonitor(
        model,
        torch.tensor([1.0, 0.0]),
        layer=1,
        position_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    with monitor:
        output = model(value)
    assert torch.equal(output, value)
    assert not monitor.active
    stats = monitor.stats.materialize()
    assert stats["projection_shift"] == 0.0
    assert stats["modified_positions"] == 0
    assert stats["valid_positions"] == 3


def test_hidden_size_and_mask_mismatches_fail_closed() -> None:
    model = _DummyModel()
    value = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="hidden size"):
        with ActivationAddition(model, torch.ones(3), 1.0, layer=1):
            model(value)
    with pytest.raises(ValueError, match="Position mask"):
        with ActivationAddition(
            model,
            torch.ones(2),
            1.0,
            layer=1,
            position_mask=torch.ones(1, 3, dtype=torch.bool),
        ):
            model(value)
