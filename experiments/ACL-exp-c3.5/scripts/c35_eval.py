#!/usr/bin/env python3
"""ACL-exp-c3.5 Hardened Evaluation Wrapper around frozen `c3_eval.py`.

Upgrades (zero modifications to frozen `ACL-exp-c1/` or `ACL-exp-c3/`):
  1. Registers `c3_blind_confirm_v2/v3`, `c3_blind_think_v2/v3`, and `c3_entity_balanced_v1`.
  2. Physically blocks `c1_test_id / c1_test_ood` unless `--allow-sealed-final` is explicitly passed.
  3. `--no-overwrite` (AUTOMATICALLY enforced whenever any sealed dataset is requested!):
       Refuses to start if `records_<regime>_<ds>.jsonl` or `eval_summary_<regime>.json` already exists!
  4. Fail-closed pre-inference SHA-256 verification:
       - `--expected-adapter-sha256 <hex>` verifies `--adapter-state` BEFORE model loading.
       - `--expected-peft-sha256 <hex>` verifies `--peft-lora-dir/adapter_model.safetensors` BEFORE loading.
       - `--expected-merged-manifest <json>` verifies every shard SHA-256 in a BF16 merged model dir.
  5. Records full `provenance` block (`adapter_state_path`, `adapter_state_sha256`, `peft_adapter_model_sha256`,
     `merged_shards_sha256`, `model_config_sha256`, `tokenizer_config_sha256`, `datasets_sha256`, `git_head`)
     into `eval_summary_<regime>.json` upon completion.

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_eval.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C3 / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-c1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import c3_eval as C3EV  # noqa: E402
import c35_modules as M35  # noqa: E402

C3EV.M3 = M35
C3EV.C3_DATASETS["c3_blind_confirm_v2"] = C35 / "data" / "c3_blind_confirm_v2_bases.jsonl"
C3EV.C3_DATASETS["c3_blind_think_v2"] = C35 / "data" / "c3_blind_think_v2_bases.jsonl"
C3EV.C3_DATASETS["c3_blind_confirm_v3"] = C35 / "data" / "c3_blind_confirm_v3_bases.jsonl"
C3EV.C3_DATASETS["c3_blind_think_v3"] = C35 / "data" / "c3_blind_think_v3_bases.jsonl"
C3EV.C3_DATASETS["c3_entity_balanced_v1"] = C35 / "data" / "c3_entity_balanced_v1_bases.jsonl"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_short() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def check_preflight_guards(
    *,
    out_dir: Path,
    datasets: list[str],
    regime: str,
    allow_sealed_final: bool,
    no_overwrite: bool,
    adapter_state: Path | None = None,
    peft_lora_dir: Path | None = None,
    model_path: str | None = None,
    expected_adapter_sha256: str | None = None,
    expected_peft_sha256: str | None = None,
    expected_merged_manifest: Path | None = None,
) -> dict[str, Any]:
    """Fail-closed preflight verification of sealed gate, no-overwrite, and exact weight SHA-256 hashes."""
    has_sealed = any(ds in C3EV.SEALED_DATASETS for ds in datasets)
    if has_sealed and not allow_sealed_final:
        raise PermissionError(
            f"[SEALED BLOCKED] Refusing to evaluate sealed dataset(s) {[d for d in datasets if d in C3EV.SEALED_DATASETS]} "
            "without explicit --allow-sealed-final!"
        )

    enforce_no_overwrite = bool(no_overwrite or has_sealed)
    if enforce_no_overwrite:
        targets = [out_dir / f"eval_summary_{regime}.json"] + [
            out_dir / f"records_{regime}_{ds}.jsonl" for ds in datasets
        ]
        existing = [str(p) for p in targets if p.exists()]
        if existing:
            raise FileExistsError(
                f"[NO-OVERWRITE BLOCKED] Refusing to overwrite existing output(s): {existing}"
            )

    prov: dict[str, Any] = {
        "schema": "cedar.acl_exp_c35.eval_provenance.v1",
        "git_head": _git_head_short(),
        "allow_sealed_final": bool(allow_sealed_final),
        "no_overwrite_enforced": enforce_no_overwrite,
        "adapter_state_path": str(adapter_state) if adapter_state else None,
        "adapter_state_sha256": None,
        "peft_lora_dir": str(peft_lora_dir) if peft_lora_dir else None,
        "peft_adapter_model_sha256": None,
        "merged_shards_sha256": None,
        "model_config_sha256": None,
        "tokenizer_config_sha256": None,
        "datasets_sha256": {},
    }

    if adapter_state is not None:
        if not adapter_state.is_file():
            raise FileNotFoundError(f"Missing --adapter-state: {adapter_state}")
        actual = sha256_file(adapter_state)
        prov["adapter_state_sha256"] = actual
        if expected_adapter_sha256:
            exp = expected_adapter_sha256.strip().lower()
            if actual != exp:
                raise AssertionError(
                    f"[SHA256 MISMATCH] adapter_state {adapter_state}: actual={actual} != expected={exp}"
                )
    elif expected_adapter_sha256:
        raise AssertionError("--expected-adapter-sha256 was set, but --adapter-state was not provided!")

    if peft_lora_dir is not None:
        st_p = peft_lora_dir / "adapter_model.safetensors"
        if not st_p.is_file():
            raise FileNotFoundError(f"Missing PEFT adapter_model.safetensors: {st_p}")
        actual = sha256_file(st_p)
        prov["peft_adapter_model_sha256"] = actual
        if expected_peft_sha256:
            exp = expected_peft_sha256.strip().lower()
            if actual != exp:
                raise AssertionError(
                    f"[SHA256 MISMATCH] peft adapter {st_p}: actual={actual} != expected={exp}"
                )
    elif expected_peft_sha256:
        raise AssertionError("--expected-peft-sha256 was set, but --peft-lora-dir was not provided!")

    if expected_merged_manifest is not None:
        if not expected_merged_manifest.is_file():
            raise FileNotFoundError(f"Missing --expected-merged-manifest: {expected_merged_manifest}")
        m_doc = json.loads(expected_merged_manifest.read_text(encoding="utf-8"))
        exp_shards = m_doc.get("shards_sha256", m_doc)
        assert isinstance(exp_shards, dict) and exp_shards, f"Invalid shards manifest: {expected_merged_manifest}"
        assert model_path is not None, "--expected-merged-manifest requires --model"
        m_dir = Path(model_path)
        actual_shards: dict[str, str] = {}
        for fname, exp_hex in sorted(exp_shards.items()):
            fp = m_dir / fname
            if not fp.is_file():
                raise FileNotFoundError(f"Missing merged shard {fp}")
            actual_hex = sha256_file(fp)
            if actual_hex != str(exp_hex).strip().lower():
                raise AssertionError(f"[MERGED SHARD MISMATCH] {fp}: actual={actual_hex} != expected={exp_hex}")
            actual_shards[fname] = actual_hex
        prov["merged_shards_sha256"] = actual_shards

    if model_path:
        m_dir = Path(model_path)
        if m_dir.is_dir():
            for key, fn in (("model_config_sha256", "config.json"), ("tokenizer_config_sha256", "tokenizer_config.json")):
                fp = m_dir / fn
                if fp.is_file():
                    prov[key] = sha256_file(fp)

    for ds in datasets:
        if ds not in C3EV.C3_DATASETS:
            raise KeyError(f"Unknown dataset {ds!r}; available={sorted(C3EV.C3_DATASETS)}")
        dp = C3EV.C3_DATASETS[ds]
        if dp.is_file():
            prov["datasets_sha256"][ds] = sha256_file(dp)

    return prov


def run_selftest_tiny() -> None:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import b1_lib as B1L
    import c1_train as C1T
    from render_prompts import load_config

    for ds in ("c3_blind_confirm_v2", "c3_blind_think_v2", "c3_blind_confirm_v3", "c3_blind_think_v3", "c3_entity_balanced_v1"):
        p = C3EV.C3_DATASETS[ds]
        assert p.is_file(), p

    # Test fail-closed preflight guards (sealed block, no-overwrite, SHA256 verification) without touching sealed files!
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        fake_pt = td_p / "adapter_state.pt"
        fake_pt.write_bytes(b"cedar-adapter-test-bytes")
        good_sha = sha256_file(fake_pt)

        # 1. Sealed dataset without --allow-sealed-final MUST fail closed
        try:
            check_preflight_guards(
                out_dir=td_p / "out1", datasets=["c1_test_id"], regime="no_think",
                allow_sealed_final=False, no_overwrite=False,
            )
            raise RuntimeError("Should have blocked sealed dataset!")
        except PermissionError:
            pass

        # 2. Wrong SHA-256 MUST fail closed
        try:
            check_preflight_guards(
                out_dir=td_p / "out1", datasets=["c1_dev"], regime="no_think",
                allow_sealed_final=False, no_overwrite=True,
                adapter_state=fake_pt, expected_adapter_sha256="0" * 64,
            )
            raise RuntimeError("Should have caught SHA-256 mismatch!")
        except AssertionError:
            pass

        # 3. Exact SHA-256 passes
        prov = check_preflight_guards(
            out_dir=td_p / "out1", datasets=["c1_dev"], regime="no_think",
            allow_sealed_final=False, no_overwrite=True,
            adapter_state=fake_pt, expected_adapter_sha256=good_sha,
        )
        assert prov["adapter_state_sha256"] == good_sha and prov["no_overwrite_enforced"] is True

        # 4. Duplicate run with existing file MUST fail closed under --no-overwrite (and automatically on sealed!)
        (td_p / "out1").mkdir(parents=True, exist_ok=True)
        (td_p / "out1" / "records_no_think_c1_dev.jsonl").write_text("{}\n", encoding="utf-8")
        try:
            check_preflight_guards(
                out_dir=td_p / "out1", datasets=["c1_dev"], regime="no_think",
                allow_sealed_final=False, no_overwrite=True,
            )
            raise RuntimeError("Should have blocked overwrite!")
        except FileExistsError:
            pass

    cfg = Qwen3Config(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    tok = C1T._TinyTok()
    a1_cfg = load_config()
    opt_ids = {L: tok.encode(L)[0] for L in C3EV.OPTION_LETTERS}
    model = Qwen3ForCausalLM(cfg).float().eval()
    rows = [json.loads(l) for l in C3EV.C3_DATASETS["c3_blind_confirm_v2"].read_text(encoding="utf-8").splitlines()[:2]]

    orig_enc = B1L.encode_prompt
    def _tiny_enc(t, msgs, pref, enable_thinking=False):
        txt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking) + pref
        return torch.tensor(t.encode(txt), dtype=torch.long)
    B1L.encode_prompt = _tiny_enc
    try:
        recs = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, rows, list(C3EV.CORE_4CELL_ARMS), opt_ids, batch_size=2)
    finally:
        B1L.encode_prompt = orig_enc
    assert len(recs) == 12
    print("[OK] c35_eval self-test PASSED (sealed block, no-overwrite, SHA-256 verification, batched eval)!")


def main() -> None:
    if "--selftest-tiny" in sys.argv[1:]:
        run_selftest_tiny()
        return

    wp = argparse.ArgumentParser(add_help=False)
    wp.add_argument("--no-overwrite", action="store_true")
    wp.add_argument("--expected-adapter-sha256", type=str, default=None)
    wp.add_argument("--expected-peft-sha256", type=str, default=None)
    wp.add_argument("--expected-merged-manifest", type=Path, default=None)
    w_args, rest_argv = wp.parse_known_args(sys.argv[1:])

    # Parse key C3EV arguments non-destructively for preflight checks & post-run provenance stamping
    pp = argparse.ArgumentParser(add_help=False)
    pp.add_argument("--out-dir", type=Path, default=None)
    pp.add_argument("--datasets", nargs="+", default=["c3_blind_confirm"])
    pp.add_argument("--regime", type=str, default="no_think")
    pp.add_argument("--allow-sealed-final", action="store_true")
    pp.add_argument("--adapter-state", type=Path, default=None)
    pp.add_argument("--peft-lora-dir", type=Path, default=None)
    pp.add_argument("--model", type=str, default=None)
    p_args, _ = pp.parse_known_args(rest_argv)

    prov: dict[str, Any] | None = None
    if p_args.out_dir is not None:
        prov = check_preflight_guards(
            out_dir=p_args.out_dir,
            datasets=list(p_args.datasets),
            regime=str(p_args.regime),
            allow_sealed_final=bool(p_args.allow_sealed_final),
            no_overwrite=bool(w_args.no_overwrite),
            adapter_state=p_args.adapter_state,
            peft_lora_dir=p_args.peft_lora_dir,
            model_path=p_args.model,
            expected_adapter_sha256=w_args.expected_adapter_sha256,
            expected_peft_sha256=w_args.expected_peft_sha256,
            expected_merged_manifest=w_args.expected_merged_manifest,
        )

    if "--components-file" not in rest_argv:
        rest_argv = [*rest_argv, "--components-file", str(C35 / "data" / "c35_components_frozen.json")]

    sys.argv = [sys.argv[0], *rest_argv]
    C3EV.main()

    if prov is not None and p_args.out_dir is not None:
        sum_p = p_args.out_dir / f"eval_summary_{p_args.regime}.json"
        if sum_p.is_file():
            doc = json.loads(sum_p.read_text(encoding="utf-8"))
            doc["provenance"] = prov
            sum_p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
            print(f"[c35_eval] stamped provenance into {sum_p}")


if __name__ == "__main__":
    main()
