#!/usr/bin/env bash
set -Eeuo pipefail
set -x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
CACHE_DIR="${C2_V8_CACHE_DIR:-/data1/WuMingrui/HF_HOME/hub}"

test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-parse '@{u}')"

conda run -n cedar python scripts/audit_c2_v8_natural_confirmation.py \
  --stage smoke --track all

conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --scope formal --track direct --condition uncapped --cache-dir "$CACHE_DIR"
conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --scope formal --track direct --condition p25 --cache-dir "$CACHE_DIR"
conda run -n cedar python scripts/audit_c2_v8_natural_confirmation.py \
  --stage formal --track direct

conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --scope formal --track free --condition uncapped --cache-dir "$CACHE_DIR"
conda run -n cedar python scripts/run_c2_v8_natural_confirmation.py \
  --scope formal --track free --condition p25 --cache-dir "$CACHE_DIR"
conda run -n cedar python scripts/audit_c2_v8_natural_confirmation.py \
  --stage formal --track free

conda run -n cedar python scripts/analyze_c2_v8_natural_confirmation.py
