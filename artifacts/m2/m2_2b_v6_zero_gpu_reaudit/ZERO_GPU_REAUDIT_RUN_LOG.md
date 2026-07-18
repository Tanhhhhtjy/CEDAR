# M2.2b v6 zero-GPU re-audit run log

Execution date: 2026-07-18 (Asia/Shanghai)

Execution platform: Linux, Python 3.11.15

Frozen analysis HEAD: `5540cda6e5d2d4b1ddb80ec274a509ea5aaaba0a`

## Safety state

- `CUDA_VISIBLE_DEVICES` was empty for tests, label finalization, and analysis.
- No model was loaded and no new response was generated.
- No p1, new seed, J-Lens, probe, MUSE, or new capping was run.
- The private key was not staged or committed.
- Private-key SHA-256 check passed:
  `aab264ea930ae05ecc5314c34b5cca9b082473c15d9613d35eaaee204c3fb785`.

## Test command

```text
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

Result:

```text
97 passed in 3.62s
```

## Label finalization

Command:

```text
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python -B scripts/finalize_v6_single_human_audit.py
```

Result:

```json
{"status":"complete","records":1152,"parse_counts":{"strict_exact_json":1151,"empty_final_surface":1},"adjudicated_sha256":"c39374b61ec11ade4110de8b21161e1ca0bd78045eaf76500d9df8c55c2d2d2f","private_key_read":false,"gpu_or_model_execution":false}
```

This follows the frozen single-human audit amendment: exact-JSON parsing for all records,
one blind human audit of the fixed 15% sample and all format anomalies. It is not represented
as double independent annotation.

## Reveal and re-analysis

Command:

```text
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python -B scripts/analyze_v6_zero_gpu_reaudit.py
```

Result:

```json
{"status":"passed","route":"Gate A","delivery_audit":"artifacts/m2/m2_2b_v6_zero_gpu_reaudit/delivery_audit.json"}
```

Key checks:

- strict and lenient `Delta_S`: `+13.8889pp`;
- strict/lenient results are identical;
- `initial→target`: 5 pairs; `target→initial`: 1 pair;
- family-clustered 95% CI: `[+5.5556pp, +25.0000pp]`;
- positive contributing families: 5/18;
- after removing the two largest positive families: `+9.3750pp`;
- truncation worst-case sensitivity: `+11.1111pp`;
- telemetry cross-check failures: 0;
- CPU simulation recommended new-family count: 24;
- final terminology remains `S target-follow candidate`, not sycophancy.

## Delivered file SHA-256

```text
a747e8f81332b5eac9f492c8012e477d6f1d5019d73b614d8b4116f25a5ad6ed  delivery_audit.json
772e901f1023f9f0b8de95901093f50845fe490f2e01fb421990235ec3adc809  strict_record_scores.jsonl
8943ed8b0d3b6fffeaf8076492e0e4b657c660eb22e08627d281f7c43081b672  lenient_record_scores.jsonl
a8b36f9ac0b888e6d00848aacb73b9dbb1f6494a24a23e47feaaa21e6d69de2e  behavior_report.json
708ca4f56a54ba8dc083c2914526d50f083fb1b16117ffa87ecbd34454c69b4f  telemetry_report.json
7f9477ba9f6cc2657851bce69a69a2e50a6f520e14390b56e86cdf2cf025a945  family_robustness.json
9e2af75110c19ff5f2460fd5d7a48cf6ded92d70f4a227419d5ffd9e90769dac  family_power_simulation.json
```
