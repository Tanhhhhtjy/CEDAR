#!/usr/bin/env bash
set -euo pipefail

python scripts/run_c3_sycobench_bidirectional.py --scope smoke --condition nohook --stage all
python scripts/run_c3_sycobench_bidirectional.py --scope smoke --condition uncapped --stage all
python scripts/run_c3_sycobench_bidirectional.py --scope smoke --condition p25 --stage all
python scripts/audit_c3_sycobench_bidirectional.py --stage smoke

python scripts/run_c3_sycobench_bidirectional.py --scope formal --condition uncapped --stage all
python scripts/audit_c3_sycobench_bidirectional.py --stage uncapped
python scripts/run_c3_sycobench_bidirectional.py --scope formal --condition p25 --stage all
python scripts/audit_c3_sycobench_bidirectional.py --stage p25
python scripts/analyze_c3_sycobench_bidirectional.py
