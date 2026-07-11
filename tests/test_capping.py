from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from cedar_m2.assets.capping_config import (
    load_axis,
    load_capping_experiment,
    verify_axis_orientation,
)
from cedar_m2.interventions.capping import ActivationCapper, LayerCap, apply_upper_cap


class IdentityBlock(nn.Module):
    def forward(self, activations: torch.Tensor) -> tuple[torch.Tensor, str]:
        return activations, "cache"


class InnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([IdentityBlock(), IdentityBlock()])


class DummyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = InnerModel()


class ApplyUpperCapTest(unittest.TestCase):
    def test_caps_only_projections_above_threshold(self) -> None:
        activations = torch.tensor([[[-1.0, 7.0], [0.5, 8.0], [2.0, 9.0]]])
        output, stats = apply_upper_cap(activations, torch.tensor([1.0, 0.0]), 1.0)

        expected = torch.tensor([[[-1.0, 7.0], [0.5, 8.0], [1.0, 9.0]]])
        torch.testing.assert_close(output, expected)
        self.assertEqual(stats.total_positions, 3)
        self.assertEqual(stats.triggered_positions, 1)
        self.assertAlmostEqual(stats.trigger_rate, 1 / 3)
        self.assertAlmostEqual(stats.projection_correction_sum, 1.0)

    def test_preserves_orthogonal_component(self) -> None:
        activations = torch.tensor([[[4.0, -2.0, 11.0]]])
        direction = torch.tensor([3.0, 4.0, 0.0])
        output, _ = apply_upper_cap(activations, direction, threshold=0.0)

        unit = direction / direction.norm()
        before_orthogonal = activations - torch.einsum(
            "...d,d->...", activations, unit
        ).unsqueeze(-1) * unit
        after_orthogonal = output - torch.einsum(
            "...d,d->...", output, unit
        ).unsqueeze(-1) * unit
        torch.testing.assert_close(after_orthogonal, before_orthogonal)

    def test_last_position_does_not_touch_prefix(self) -> None:
        activations = torch.tensor([[[3.0, 1.0], [4.0, 2.0]]])
        output, stats = apply_upper_cap(
            activations, torch.tensor([1.0, 0.0]), 1.0, positions="last"
        )

        torch.testing.assert_close(output[..., 0, :], activations[..., 0, :])
        torch.testing.assert_close(output[..., 1, :], torch.tensor([[1.0, 2.0]]))
        self.assertEqual(stats.total_positions, 1)
        self.assertEqual(stats.triggered_positions, 1)


class ActivationCapperTest(unittest.TestCase):
    def test_hook_preserves_tuple_and_is_removed(self) -> None:
        model = DummyCausalLM()
        cap = LayerCap(layer=1, direction=torch.tensor([1.0, 0.0]), threshold=1.0)
        activations = torch.tensor([[[2.0, 5.0]]])

        with ActivationCapper(model, {1: cap}) as capper:
            output, cache = model.model.layers[1](activations)
            torch.testing.assert_close(output, torch.tensor([[[1.0, 5.0]]]))
            self.assertEqual(cache, "cache")
            self.assertEqual(capper.stats[1].triggered_positions, 1)

        uncapped, _ = model.model.layers[1](activations)
        torch.testing.assert_close(uncapped, activations)

    def test_rejects_key_layer_mismatch(self) -> None:
        model = DummyCausalLM()
        cap = LayerCap(layer=1, direction=torch.tensor([1.0]), threshold=0.0)
        with self.assertRaisesRegex(ValueError, "does not match"):
            ActivationCapper(model, {0: cap})


class CappingConfigTest(unittest.TestCase):
    def test_load_and_verify_opposite_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis_path = root / "axis.pt"
            config_path = root / "config.pt"
            torch.save(torch.tensor([[-1.0, 0.0], [0.0, -1.0]]), axis_path)
            torch.save(
                {
                    "vectors": {
                        "layer_1/test": {"vector": torch.tensor([0.0, 3.0]), "layer": 1}
                    },
                    "experiments": [
                        {
                            "id": "test-exp",
                            "interventions": [{"vector": "layer_1/test", "cap": -2.5}],
                        }
                    ],
                },
                config_path,
            )

            axis = load_axis(axis_path)
            caps = load_capping_experiment(config_path, "test-exp")
            similarities = verify_axis_orientation(caps, axis)

            self.assertEqual(list(caps), [1])
            self.assertEqual(caps[1].threshold, -2.5)
            self.assertAlmostEqual(similarities[1], -1.0)

    def test_orientation_check_rejects_same_direction(self) -> None:
        caps = {0: LayerCap(0, torch.tensor([1.0, 0.0]), 0.0)}
        with self.assertRaisesRegex(ValueError, "expected <="):
            verify_axis_orientation(caps, torch.tensor([[1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
