# B2 operations · authorized 2026-10-06

The user explicitly authorized B2 after reviewing B1: “可以，你现在开始B2吧”. This supersedes the first-package restriction on starting B2. Scientific protocol remains B_PROTOCOL.md; resource policy remains operations/RESOURCE_AMENDMENT.md. No cumulative runtime cap; count elapsed time. See operations/b2_20261006/AUTHORIZATION.md and its separate manifest.

Remote root: /data4/WuMingrui/TianJinyu/CEDAR. Output entry: /data4/WuMingrui/TianJinyu/CEDAR/arr_next_steps_20261005, resolved to /data1/WuMingrui/TianJinyu/CEDAR/arr_next_steps_20261005. All local artifacts go under D:/CEDAR/arr_next_steps_20261005.

## Canonical execution

Use /home/WuMingrui/miniconda3/envs/cedar/bin/python -u operations/b2_20261006/campaign_b2.py with --root and --out set to the absolute remote paths above, cwd at the remote root. The launcher must use absolute script path, detached session, stdin closed, and timestamped log under output logs. Never restart old campaign.py or the first-package campaign_runtime.py to run B2.

This controller verifies frozen manifests, original completed smoke24/B1 192 and all asset SHA-256s, then uses the unchanged scripts/run_phase_factorial.py --scope B2. It reuses the passed smoke, acquires the frozen 192 B2 cells, audits them, and runs unchanged scripts/analyze_phase_factorial.py for B2 and pooled. B1 is retained unchanged; all prior first-package files are fingerprinted and checked again on completion. C/D/H model matrices are outside this authorization.

## Monitoring and recovery

Use operations/b2_20261006/monitor_b2.py --root ... --out ... with the cedar Python. Verify actual process command/cwd/uid, shared campaign/runner locks, frozen manifests, per-record JSON/NPZ hashes, terminal and cap-call gates, counts, active-cell age, same-file log growth, GPU memory/utilization, output/input free space. A live process alone is not evidence of progress. Server supervision checks every 15 seconds and records snapshots every 5 minutes. Chat heartbeat should run every 30 minutes and remain quiet unless meaningful progress, failure/repair or action is needed.

Retain 90 minutes without transition, 20 GiB output, 2 GiB input and memory/technical gates. Preserve completed/failed/orphan records; a restart skips completed verified cells. If the supervisor is lost while a runner still holds the lock, inspect and recover its identity/accounting before any new launch; never launch duplicates or kill unrelated processes. Never change precision, tokens, cap settings, sample selection, metrics or gates to pass.

Key paths: explore_outputs/campaign_status.json, explore_outputs/gpu_budget.json, explore_outputs/monitor_state.json, explore_outputs/heartbeat_B2_latest.json, explore_outputs/B2_start_*/preflight.json; runs/B2/status.json, results.json and runtime.json; logs/B2_<timestamp>.log (exclude B2_analysis logs). The budget accumulates earlier smoke+B1 and new B2 separately. New B2 failure evidence uses campaign_B2_failure_* or failure_B2_*.

SSH uses the previously authorized host, port 50000 and D:/CEDAR/arr_revision_20260927/known_hosts. Direct IPv6 worked at B2 intake; the optional old local proxy port was not listening. Password only through SSH stdin, never persisted or logged. If recovering from read_thread, keep the raw response in memory and extract only the necessary credential without printing it.

## Completion

Acquire all 192 B2 cells regardless of scientific effect. Keep separate B1/B2 descriptive reports and pooled 384-cell report; pooled six raw EH/ET/I tests are the single prespecified Holm family. Do not rerun completed pooled analysis merely to change presentation. Audit the 24+192+192 record sets and hashes, verify first-package preservation, independently check pooled statistics and scale fits, synchronize evidence to D:/CEDAR, report actual runtime and limitations. Confirm no remaining managed processes and free locks, then pause the existing monitor. Do not automatically start C/D/H.
