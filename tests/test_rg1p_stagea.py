from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/m2_2b_formal/rg1p_stagea"
E2B_DATA = ROOT / "datasets/m2_2b_formal/e2b_fixed_trace_cross"
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (DATA, E2B_DATA, SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_rg1p_stagea import projection_metrics, save_official_hidden_shards  # noqa: E402
from plot_rg1p_stagea import generate_svg  # noqa: E402
from rg1p_common import (  # noqa: E402
    CAP_LAYERS,
    CAPTURE_LAYERS,
    EXPECTED_FORMAL_CELLS,
    EXPECTED_SMOKE_CELLS,
    NOOP_CONDITION,
    RANDOM_CONDITIONS,
    build_random_directions,
    family_bootstrap,
    matrix_cells,
    read_json,
    read_jsonl,
    smoke_prompt_ids,
)
from run_rg1p_stagea import (  # noqa: E402
    OfficialScheduleCapture,
    ScheduleMatchedRandomInjector,
    _native_lens_logits,
    restricted_option_log_probs,
)

from cedar_m2.interventions.capping import ActivationCapper, LayerCap  # noqa: E402


class _Inner(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(64)])
        self.norm = nn.LayerNorm(hidden)
        self.hidden = hidden


class _ToyModel(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.model = _Inner(hidden)
        self.lm_head = nn.Linear(hidden, 40, bias=False)


class RG1PCommonTests(unittest.TestCase):
    def test_random_directions_are_deterministic_unit_and_orthogonal(self) -> None:
        generator = torch.Generator().manual_seed(5)
        axis = torch.randn((64, 32), generator=generator)
        first, first_meta = build_random_directions(axis)
        second, second_meta = build_random_directions(axis)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_meta["tensor_sha256"], second_meta["tensor_sha256"])
        for seed_index in range(first.shape[0]):
            for layer_index, layer in enumerate(CAP_LAYERS):
                direction = first[seed_index, layer_index].double()
                official = axis[layer].double()
                official /= torch.linalg.vector_norm(official)
                self.assertLessEqual(abs(float(torch.dot(direction, official))), 1e-6)
                self.assertAlmostEqual(float(torch.linalg.vector_norm(direction)), 1.0, places=6)

    def test_matrix_counts_and_random_scope(self) -> None:
        traces = read_jsonl(E2B_DATA / "frozen_traces192.jsonl")
        smoke_ids = smoke_prompt_ids(traces)
        smoke = matrix_cells(traces, scope="smoke", smoke_ids=smoke_ids)
        formal = matrix_cells(traces, scope="formal", smoke_ids=smoke_ids)
        self.assertEqual(len(smoke), EXPECTED_SMOKE_CELLS)
        self.assertEqual(len(formal), EXPECTED_FORMAL_CELLS)
        self.assertEqual(
            {
                kind: sum(row["cell_kind"] == kind for row in smoke)
                for kind in {"official", "engineering_noop", "random_reference"}
            },
            {"official": 32, "engineering_noop": 8, "random_reference": 32},
        )
        self.assertEqual(
            {
                kind: sum(row["cell_kind"] == kind for row in formal)
                for kind in {"official", "engineering_noop", "random_reference"}
            },
            {"official": 384, "engineering_noop": 96, "random_reference": 384},
        )
        random_rows = [row for row in formal if row["forward_condition"] in RANDOM_CONDITIONS]
        self.assertEqual(len(random_rows), 384)
        self.assertEqual({row["trace_source"] for row in random_rows}, {"e2a_uncapped"})
        noop_rows = [row for row in formal if row["forward_condition"] == NOOP_CONDITION]
        self.assertEqual(len(noop_rows), 96)
        self.assertEqual({row["trace_source"] for row in noop_rows}, {"e2a_uncapped"})

    def test_restricted_option_softmax_is_normalized(self) -> None:
        logits = torch.linspace(-3, 3, 40)
        log_probs = restricted_option_log_probs(logits, [32, 33, 34, 35])
        self.assertEqual(list(log_probs.shape), [4])
        self.assertAlmostEqual(float(torch.exp(log_probs).sum().item()), 1.0, places=6)

    def test_native_lens_is_final_norm_then_full_lm_head(self) -> None:
        model = _ToyModel(hidden=16)
        states = torch.randn((64, 16), generator=torch.Generator().manual_seed(9))
        expected = model.lm_head(model.model.norm(states)).float()
        observed = _native_lens_logits(model, states)
        self.assertTrue(torch.equal(observed, expected))

    def test_family_bootstrap_uses_twelve_clusters(self) -> None:
        scores = {f"F{index:02d}": float(index - 5) for index in range(12)}
        first = family_bootstrap(scores, draws=500, seed=17)
        second = family_bootstrap(scores, draws=500, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["families"], 12)
        self.assertEqual(first["estimate"], 0.5)

    def test_capture_and_projection_contract(self) -> None:
        self.assertEqual(CAPTURE_LAYERS, tuple(range(64)))
        baseline = torch.tensor([2.0, 0.0, 0.0])
        delta = torch.tensor([3.0, 4.0, 0.0])
        axis = torch.tensor([1.0, 0.0, 0.0])
        values = projection_metrics(delta, baseline, axis)
        self.assertAlmostEqual(values["aligned_norm"], 3.0)
        self.assertAlmostEqual(values["off_direction_norm"], 4.0)
        self.assertAlmostEqual(values["aligned_share"] + values["off_direction_share"], 1.0)

    def test_official_hidden_states_are_split_into_four_shards(self) -> None:
        rows = []
        for trace_source in ("e2a_uncapped", "e2a_full_p25"):
            for condition in ("uncapped", "full_p25"):
                rows.extend(
                    {
                        "cell_id": f"{trace_source}-{condition}-{index}",
                        "trace_source": trace_source,
                        "forward_condition": condition,
                    }
                    for index in range(96)
                )
        states = torch.zeros((384, 64, 2), dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = save_official_hidden_shards(root, rows, states)
            index = read_json(root / "hidden_states.index.json")
            self.assertEqual(result["official_rows"], 384)
            self.assertEqual(len(result["shards"]), 4)
            self.assertEqual(len(index["rows"]), 384)

    def test_panel_d_svg_contract(self) -> None:
        def summary(value: float) -> dict[str, object]:
            return {
                "estimate": value,
                "family_cluster_bootstrap_95ci": [value - 0.01, value + 0.01],
            }

        stats = {
            "panel_a": {
                name: {str(layer): summary(0.2) for layer in range(46, 64)}
                for name in ("official_p25", "r1", "r2", "r3", "r4")
            },
            "panel_b": {
                arm: {str(layer): summary(0.3) for layer in (53, 60, 63)}
                for arm in "NRVF"
            },
            "panel_c": {
                "spearman_rho": 0.2,
                "points": [
                    {
                        "series": f"F{index}",
                        "x_g_off": index / 100,
                        "y_delta_m_general": index / 10,
                    }
                    for index in range(12)
                ],
            },
            "panel_d": {
                "trace_sources": {
                    source: {
                        metric: {str(layer): summary((layer + 1) / 1000) for layer in range(64)}
                        for metric in ("delta_m_general_mean_noninitial", "kl_opt")
                    }
                    for source in ("U", "P")
                }
            },
        }
        svg = generate_svg(stats)
        self.assertIn("D  Layer-resolved emergence", svg)
        self.assertIn("cache-faithful replay", svg)


class RG1PHookTests(unittest.TestCase):
    def test_official_schedule_capture_records_actual_norms(self) -> None:
        hidden = 16
        model = _ToyModel(hidden)
        direction = torch.zeros(hidden)
        direction[0] = 1
        caps = {
            layer: LayerCap(layer=layer, direction=direction, threshold=0.0) for layer in CAP_LAYERS
        }
        capture = OfficialScheduleCapture(model, terminal_call_index=1)
        capper = ActivationCapper(model, caps, positions="all", track_stats=False)
        with capture.before_context(), capper, capture.after_context():
            for positions in (3, 1):
                value = torch.zeros((1, positions, hidden), dtype=torch.bfloat16)
                value[..., 0] = 2
                for layer in CAP_LAYERS:
                    model.model.layers[layer](value)
        schedule, cap_input, cap_output, telemetry = capture.finalize()
        self.assertEqual(list(schedule.shape), [8, 4])
        self.assertTrue(torch.allclose(schedule, torch.full_like(schedule, 2.0)))
        self.assertEqual(list(cap_input.shape), [8, hidden])
        self.assertEqual(list(cap_output.shape), [8, hidden])
        self.assertEqual(set(telemetry["triggered_by_layer"].values()), {4})

    def test_random_injector_consumes_same_schedule_and_norm(self) -> None:
        hidden = 64
        model = _ToyModel(hidden)
        schedule = torch.zeros((8, 4), dtype=torch.float32)
        schedule[:, (0, 2, 3)] = torch.tensor([0.5, 1.25, 2.0])
        directions = {}
        for index, layer in enumerate(CAP_LAYERS):
            direction = torch.zeros(hidden)
            direction[(index + 3) % hidden] = 1
            directions[layer] = direction
        injector = ScheduleMatchedRandomInjector(
            model,
            schedule,
            directions,
            sign=-1,
            terminal_call_index=1,
        )
        with injector:
            for positions in (3, 1):
                value = torch.zeros((1, positions, hidden), dtype=torch.bfloat16)
                for layer in CAP_LAYERS:
                    output = model.model.layers[layer](value)
                    self.assertEqual(output.dtype, torch.bfloat16)
        input_state, output_state, realized_norms, telemetry = injector.finalize()
        self.assertEqual(list(input_state.shape), [8, hidden])
        self.assertEqual(list(output_state.shape), [8, hidden])
        self.assertEqual(list(realized_norms.shape), [8, 4])
        self.assertTrue(torch.equal(realized_norms > 0, schedule > 0))
        self.assertLessEqual(telemetry["requested_norm_max_abs_error"], 1e-5)
        self.assertTrue(telemetry["realized_norm_within_frozen_bf16_tolerance"])
        self.assertEqual(set(telemetry["triggered_by_layer"].values()), {3})


if __name__ == "__main__":
    unittest.main()
