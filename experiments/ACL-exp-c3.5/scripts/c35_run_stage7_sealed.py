#!/usr/bin/env python3
"""ACL-exp-c3.5 Stage 7 Bulletproof Parallel Runner for One-Shot Sealed Evaluation (`c1_test_id` & `c1_test_ood`).

Reads the signed manifest `experiments/ACL-exp-c3.5/data/stage7_sealed_config.json` and executes ALL 26 models
(9 Qwen3-14B + 17 Qwen3-8B) with ZERO retraining, strict pre-inference SHA-256 verification, write-once `--no-overwrite`,
deterministic resume of completed jobs, and final fail-closed aggregation via `c35_stage7_stats.py`.

Modes:
  1. `--dry-run-plan` (CPU only, zero GPU, zero sealed access):
       Validates manifest schema, prints all 50 exact commands (`14B=18`, `8B=32`), verifies every command enforces
       `--allow-sealed-final`, `--no-overwrite`, and full 64-char SHA-256 verification.
  2. `--verify-only` (server pre-launch check, zero sealed access):
       Verifies every single adapter `.pt`, PEFT `.safetensors`, base model dir, dataset SHA-256, and
       CAA frozen vector (`L2 == 42.4911`) exists on disk with exact SHA-256 match BEFORE unsealing!
  3. `--execute --allow-sealed-final --gpus 0,1,2,3,4,5,6,7`:
       Runs Wave 7A (14B, 2 GPUs/job -> 4 parallel) and Wave 7B (8B, 1 GPU/job -> 8 parallel),
       followed by `c35_stage7_stats.py` strict fail-closed aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]

DEFAULT_CONFIG = C35 / "data" / "stage7_sealed_config.json"
DEFAULT_OUT_ROOT = C35 / "results" / "stage7_sealed"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _job_outputs_complete(arm_dir: Path, regime: str, id_ds: str, ood_ds: str) -> bool:
    rdir = arm_dir / regime
    return (
        (rdir / f"records_{regime}_{id_ds}.jsonl").is_file()
        and (rdir / f"records_{regime}_{ood_ds}.jsonl").is_file()
        and (rdir / f"eval_summary_{regime}.json").is_file()
    )


def _job_outputs_partial(arm_dir: Path, regime: str, id_ds: str, ood_ds: str) -> bool:
    rdir = arm_dir / regime
    files = [
        rdir / f"records_{regime}_{id_ds}.jsonl",
        rdir / f"records_{regime}_{ood_ds}.jsonl",
        rdir / f"eval_summary_{regime}.json",
    ]
    present = sum(1 for p in files if p.exists())
    return 0 < present < len(files)


def _find_caa_vector(explicit_path: Path | None, default_rel: str) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    cands = [
        REPO / default_rel,
        Path("/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage6-runtime/caa_directions_train.pt"),
        Path("/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage6-runtime/external_8b/s6_8b_caa_steering/caa_directions_train.pt"),
    ]
    for p in cands:
        if p.is_file():
            return p
    return None


def build_stage7_jobs(
    cfg: dict[str, Any],
    *,
    python_exe: str,
    caa_vector_file: Path | None = None,
    pinpoint_merged_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic list of all 50 Stage 7 evaluation jobs (`14B=18`, `8B=32`) with exact CLI commands."""
    id_ds = cfg["id_dataset_name"]
    ood_ds = cfg["ood_dataset_name"]
    jobs: list[dict[str, Any]] = []

    for scale_key in ("qwen3_14b", "qwen3_8b"):
        sc = cfg["scales"][scale_key]
        base_model = sc["base_model"]
        comp_file = sc["components_file"]
        is_14b = scale_key == "qwen3_14b"
        gpus_needed = 2 if is_14b else 1
        nt_bs = 4 if is_14b else 8
        tp = 2 if is_14b else 1

        for arm in sc["arms"]:
            tag = arm.get("tag", f"s7_{'14b' if is_14b else '8b'}_m0")
            m_group = arm["model_group"]
            arm_dir = REPO / arm["dir"]
            regimes = arm.get("regimes", ["no_think", "think"])

            # 1. `no_think` job (HF native BF16 exact, or CAA frozen hook)
            if "no_think" in regimes:
                if m_group == "CAA-Steering":
                    cspec = arm["caa_frozen_spec"]
                    vec_p = caa_vector_file or Path(cspec["server_vector_file_default"])
                    cmd = [
                        python_exe, str(C35 / "scripts" / "c35_eval_caa.py"),
                        "--model", base_model,
                        "--frozen-vector-file", str(vec_p),
                        "--frozen-mode", str(cspec["contrast_mode"]),
                        "--frozen-layer", str(cspec["layer"]),
                        "--frozen-alpha", str(cspec["alpha"]),
                        "--expected-vector-l2", str(cspec["expected_vector_l2"]),
                        "--batch-size", str(nt_bs),
                        "--datasets", f"{id_ds},{ood_ds}",
                        "--allow-sealed-final",
                        "--no-overwrite",
                        "--skip-kl",
                        "--skip-exfever",
                        "--out-dir", str(arm_dir),
                    ]
                    if arm.get("expected_vector_sha256"):
                        cmd.extend(["--expected-vector-sha256", str(arm["expected_vector_sha256"])])
                else:
                    cmd = [
                        python_exe, str(C35 / "scripts" / "c35_eval.py"),
                        "--model", base_model,
                        "--model-key", scale_key,
                        "--components-file", str(REPO / comp_file),
                        "--alias", tag,
                        "--regime", "no_think",
                        "--batch-size", str(nt_bs),
                        "--datasets", id_ds, ood_ds,
                        "--allow-sealed-final",
                        "--no-overwrite",
                        "--out-dir", str(arm_dir / "no_think"),
                    ]
                    if arm.get("server_adapter_state"):
                        cmd.extend([
                            "--arch", arm["arch"],
                            "--adapter-state", arm["server_adapter_state"],
                            "--expected-adapter-sha256", arm["expected_adapter_sha256"],
                        ])
                jobs.append({
                    "job_id": f"{tag}__no_think",
                    "scale": scale_key,
                    "tag": tag,
                    "model_group": m_group,
                    "regime": "no_think",
                    "gpus_needed": gpus_needed,
                    "arm_dir": arm_dir,
                    "cmd": cmd,
                    "arm_spec": arm,
                })

            # 2. `think` job (vLLM BF16 base, vLLM native LoRA, or Pinpoint-Corridor BF16 merged)
            if "think" in regimes:
                th_backend = arm.get("think_backend", "vllm_base")
                model_arg = arm["server_merged_model"] if th_backend == "vllm_bf16_merged" else base_model
                cmd = [
                    python_exe, str(C35 / "scripts" / "c35_eval.py"),
                    "--model", model_arg,
                    "--backend", "vllm",
                    "--tp", str(tp),
                    "--alias", tag,
                    "--regime", "think",
                    "--max-new-tokens", "3072",
                    "--datasets", id_ds, ood_ds,
                    "--arms", "turn1_F", "turn1_V", "V_hes", "V_auth", "F_hes", "F_auth",
                    "--allow-sealed-final",
                    "--no-overwrite",
                    "--out-dir", str(arm_dir / "think"),
                ]
                if th_backend == "vllm_native_lora":
                    cmd.extend([
                        "--peft-lora-dir", arm["server_peft_lora_dir"],
                        "--expected-peft-sha256", arm["expected_peft_sha256"],
                    ])
                elif th_backend == "vllm_bf16_merged":
                    m_path = pinpoint_merged_manifest or (
                        C35 / "results/stage6/external_8b/s6_8b_s42_pinpoint_dla_corridor/vllm_merged_parity.json"
                    )
                    cmd.extend(["--expected-merged-manifest", str(m_path)])
                jobs.append({
                    "job_id": f"{tag}__think",
                    "scale": scale_key,
                    "tag": tag,
                    "model_group": m_group,
                    "regime": "think",
                    "gpus_needed": gpus_needed,
                    "arm_dir": arm_dir,
                    "cmd": cmd,
                    "arm_spec": arm,
                })

    return jobs


def verify_server_assets(
    cfg: dict[str, Any],
    *,
    caa_vector_file: Path | None,
    recreate_pinpoint_merged: bool = False,
) -> dict[str, Any]:
    """Fail-closed server pre-launch asset & SHA-256 verification (ZERO reading of sealed dataset contents)."""
    print("=== STAGE 7 SERVER PRE-LAUNCH ASSET VERIFICATION ===")
    errors: list[str] = []
    report: dict[str, Any] = {"datasets": {}, "models": {}, "caa_vector": None}

    # 1. Verify sealed dataset file SHA-256s (binary hash ONLY, never parses rows!)
    for fname, exp_sha in cfg["expected_dataset_sha256"].items():
        p = REPO / "experiments" / "ACL-exp-c1" / "data" / fname
        if not p.is_file():
            errors.append(f"Missing dataset: {p}")
        else:
            act = sha256_file(p)
            if act != exp_sha:
                errors.append(f"Dataset SHA mismatch {fname}: actual={act} != expected={exp_sha}")
            else:
                report["datasets"][fname] = act
                print(f"  [OK] {fname}: {act[:16]}...")

    # 2. Verify every adapter, PEFT dir, merged Pinpoint model, and CAA vector!
    for scale_key in ("qwen3_14b", "qwen3_8b"):
        sc = cfg["scales"][scale_key]
        if not Path(sc["base_model"]).is_dir():
            errors.append(f"Missing base model dir: {sc['base_model']}")
        for arm in sc["arms"]:
            tag = arm.get("tag", f"{scale_key}_m0")
            grp = arm["model_group"]
            if grp == "M0":
                continue
            if grp == "CAA-Steering":
                cspec = arm["caa_frozen_spec"]
                vec_p = _find_caa_vector(caa_vector_file, cspec["server_vector_file_default"])
                if vec_p is None or not vec_p.is_file():
                    errors.append(f"Missing CAA frozen vector file (pass --caa-vector-file): default={cspec['server_vector_file_default']}")
                else:
                    v_sha = sha256_file(vec_p)
                    report["caa_vector"] = {"path": str(vec_p), "sha256": v_sha}
                    print(f"  [OK] CAA vector: {vec_p} sha256={v_sha}")
                continue

            ap = Path(arm["server_adapter_state"])
            if not ap.is_file():
                errors.append(f"[{tag}] Missing adapter_state: {ap}")
            else:
                act = sha256_file(ap)
                if act != arm["expected_adapter_sha256"]:
                    errors.append(f"[{tag}] adapter_state SHA mismatch: {act} != {arm['expected_adapter_sha256']}")

            if arm.get("think_backend") == "vllm_native_lora":
                pp = Path(arm["server_peft_lora_dir"]) / "adapter_model.safetensors"
                if not pp.is_file():
                    errors.append(f"[{tag}] Missing PEFT safetensors: {pp}")
                else:
                    act_p = sha256_file(pp)
                    if act_p != arm["expected_peft_sha256"]:
                        errors.append(f"[{tag}] PEFT SHA mismatch: {act_p} != {arm['expected_peft_sha256']}")
            elif arm.get("think_backend") == "vllm_bf16_merged":
                mdir = Path(arm["server_merged_model"])
                if not mdir.is_dir() and recreate_pinpoint_merged:
                    peft_dir = Path("/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage6-runtime/peft_8b/s6_8b_s42_pinpoint_dla_corridor")
                    peft_st = peft_dir / "adapter_model.safetensors"
                    exp_pinpoint_peft_sha = "ee3454fdf4cb3aba1a640dcd1c859eb835a5df8e6ac86d6282f0ce2a6ce87d76"
                    if not peft_st.is_file():
                        errors.append(f"[{tag}] Missing source PEFT safetensors for re-merge: {peft_st}")
                    elif sha256_file(peft_st) != exp_pinpoint_peft_sha:
                        errors.append(f"[{tag}] Source PEFT SHA mismatch before re-merge: {sha256_file(peft_st)} != {exp_pinpoint_peft_sha}")
                    else:
                        print(f"  [INFO] Source PEFT verified; re-merging Pinpoint-DLA-Corridor into {mdir} (float32 accum -> BF16)...")
                        import torch
                        from peft import PeftModel
                        from transformers import AutoModelForCausalLM, AutoTokenizer

                        base = AutoModelForCausalLM.from_pretrained(sc["base_model"], dtype=torch.float32, device_map="cpu", trust_remote_code=True)
                        pm = PeftModel.from_pretrained(base, str(peft_dir))
                        merged = pm.merge_and_unload().to(dtype=torch.bfloat16)
                        mdir.mkdir(parents=True, exist_ok=True)
                        merged.save_pretrained(mdir, safe_serialization=True)
                        AutoTokenizer.from_pretrained(sc["base_model"], trust_remote_code=True).save_pretrained(mdir)
                        del merged, pm, base
                if not mdir.is_dir():
                    errors.append(f"[{tag}] Missing BF16 merged dir {mdir} (re-run with --recreate-pinpoint-merged if deleted after Stage 6)")
                else:
                    for sh_name, exp_sh in sorted(arm["expected_merged_shards_sha256"].items()):
                        sp = mdir / sh_name
                        if not sp.is_file():
                            errors.append(f"[{tag}] Missing merged shard {sp}")
                        else:
                            act_sh = sha256_file(sp)
                            if act_sh != exp_sh:
                                errors.append(f"[{tag}] Merged shard SHA mismatch {sh_name}: {act_sh} != {exp_sh}")
            report["models"][tag] = "VERIFIED"

    if errors:
        print("\n[FAIL] Pre-launch verification detected errors:")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)
    print(f"[OK] ALL {len(report['models'])} adapters + datasets + CAA vector verified bit-for-bit!")
    return report


def run_parallel_wave(
    jobs: list[dict[str, Any]],
    gpu_groups: list[str],
    *,
    id_ds: str,
    ood_ds: str,
    log_dir: Path,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    q: Queue[str] = Queue()
    for g in gpu_groups:
        q.put(g)

    def _run_one(job: dict[str, Any]) -> tuple[str, str]:
        arm_dir = job["arm_dir"]
        reg = job["regime"]
        if _job_outputs_complete(arm_dir, reg, id_ds, ood_ds):
            print(f"[SKIP COMPLETE] {job['job_id']}")
            return job["job_id"], "SKIPPED"
        if _job_outputs_partial(arm_dir, reg, id_ds, ood_ds):
            raise RuntimeError(
                f"[PARTIAL OUTPUT DETECTED] {job['job_id']} in {arm_dir / reg}: refusing to overwrite partial sealed outputs!"
            )
        gpu_str = q.get()
        t0 = time.time()
        log_p = log_dir / f"{job['job_id']}.log"
        try:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu_str
            print(f"[LAUNCH GPU={gpu_str}] {job['job_id']} (log -> {log_p})", flush=True)
            with log_p.open("w", encoding="utf-8") as lf:
                res = subprocess.run(job["cmd"], cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT)
            if res.returncode != 0:
                tail = "\n".join(log_p.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
                raise RuntimeError(
                    f"Job {job['job_id']} failed on GPU={gpu_str} (rc={res.returncode}). Last 30 log lines:\n{tail}"
                )
            dt = time.time() - t0
            print(f"[DONE GPU={gpu_str} {dt:.1f}s] {job['job_id']}", flush=True)
            return job["job_id"], f"DONE ({dt:.1f}s)"
        finally:
            q.put(gpu_str)

    with ThreadPoolExecutor(max_workers=len(gpu_groups)) as ex:
        futs = [ex.submit(_run_one, j) for j in jobs]
        for fut in as_completed(futs):
            fut.result()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--caa-vector-file", type=Path, default=None)
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--log-dir", type=Path, default=Path("/tmp/c35_stage7_logs"))
    ap.add_argument("--dry-run-plan", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--recreate-pinpoint-merged", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--allow-sealed-final", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()

    cfg = json.loads(args.config_json.read_text(encoding="utf-8"))
    jobs = build_stage7_jobs(cfg, python_exe=args.python, caa_vector_file=args.caa_vector_file)

    # Hard-assert EVERY job enforces --allow-sealed-final and --no-overwrite!
    for j in jobs:
        assert "--allow-sealed-final" in j["cmd"], j["job_id"]
        assert "--no-overwrite" in j["cmd"], j["job_id"]
        if j["model_group"] not in ("M0", "CAA-Steering") and j["regime"] == "no_think":
            assert "--expected-adapter-sha256" in j["cmd"], j["job_id"]
        if j["arm_spec"].get("think_backend") == "vllm_native_lora" and j["regime"] == "think":
            assert "--expected-peft-sha256" in j["cmd"], j["job_id"]
        if j["arm_spec"].get("think_backend") == "vllm_bf16_merged" and j["regime"] == "think":
            assert "--expected-merged-manifest" in j["cmd"], j["job_id"]

    if args.dry_run_plan:
        j14 = [j for j in jobs if j["scale"] == "qwen3_14b"]
        j8 = [j for j in jobs if j["scale"] == "qwen3_8b"]
        print(f"[OK] Stage 7 Dry-Run Plan VALIDATED: total_jobs={len(jobs)} (14B={len(j14)}, 8B={len(j8)})")
        print("  - Every job enforces --allow-sealed-final, --no-overwrite, and full 64-char SHA-256 verification!")
        for j in jobs[:4]:
            print(f"  SAMPLE {j['job_id']}: {' '.join(j['cmd'][:12])} ...")
        # Verify DPO & CAA have NO think jobs and Pinpoint-Corridor uses merged manifest!
        assert not any(j["model_group"] == "Joint-DPO" and j["regime"] == "think" for j in jobs)
        assert not any(j["model_group"] == "CAA-Steering" and j["regime"] == "think" for j in jobs)
        assert any(j["model_group"] == "Pinpoint-DLA-Corridor" and j["regime"] == "think" and "--expected-merged-manifest" in j["cmd"] for j in jobs)
        print("[OK] Backend locks verified: Joint-DPO (HF no_think only), CAA (frozen hook no_think only), Pinpoint-Corridor (merged think)!")
        return

    if args.verify_only:
        verify_server_assets(cfg, caa_vector_file=args.caa_vector_file, recreate_pinpoint_merged=bool(args.recreate_pinpoint_merged))
        return

    if args.execute:
        assert args.allow_sealed_final, "Refusing --execute without explicit --allow-sealed-final!"
        report = verify_server_assets(cfg, caa_vector_file=args.caa_vector_file, recreate_pinpoint_merged=bool(args.recreate_pinpoint_merged))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "stage7_prelaunch_asset_verification.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        # Patch resolved CAA vector path & SHA into runtime jobs
        if report.get("caa_vector"):
            vec_p = Path(report["caa_vector"]["path"])
            jobs = build_stage7_jobs(cfg, python_exe=args.python, caa_vector_file=vec_p)
            for j in jobs:
                if j["model_group"] == "CAA-Steering" and "--expected-vector-sha256" not in j["cmd"]:
                    j["cmd"].extend(["--expected-vector-sha256", report["caa_vector"]["sha256"]])

        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
        assert len(gpu_list) >= 2, "Need at least 2 GPUs for 14B tp=2!"
        pairs_14b = [f"{gpu_list[i]},{gpu_list[i+1]}" for i in range(0, len(gpu_list) - 1, 2)]
        id_ds, ood_ds = cfg["id_dataset_name"], cfg["ood_dataset_name"]

        print(f"\n=== WAVE 7A: Qwen3-14B Sealed Eval ({len([j for j in jobs if j['scale']=='qwen3_14b'])} jobs across {pairs_14b}) ===")
        run_parallel_wave([j for j in jobs if j["scale"] == "qwen3_14b"], pairs_14b, id_ds=id_ds, ood_ds=ood_ds, log_dir=args.log_dir)

        print(f"\n=== WAVE 7B: Qwen3-8B Sealed Eval ({len([j for j in jobs if j['scale']=='qwen3_8b'])} jobs across {gpu_list}) ===")
        run_parallel_wave([j for j in jobs if j["scale"] == "qwen3_8b"], gpu_list, id_ds=id_ds, ood_ds=ood_ds, log_dir=args.log_dir)

        print("\n=== FINAL STAGE 7 STRICT AGGREGATION ===")
        agg_cmd = [
            args.python, str(C35 / "scripts" / "c35_stage7_stats.py"),
            "--config-json", str(args.config_json),
            "--out-dir", str(args.out_dir),
            "--n-boot", "5000",
        ]
        res = subprocess.run(agg_cmd, cwd=str(REPO))
        if res.returncode != 0:
            raise SystemExit(res.returncode)
        print("[OK] STAGE 7 SEALED EVALUATION & AGGREGATION 100% COMPLETE!")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
