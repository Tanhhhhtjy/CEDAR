# B initial package operations

User authorized finalizing the feedback and starting on2026-10-05. Read EXPERIMENT_PLAN.md and B_PROTOCOL.md. Scope: CPU A + H preparation, smoke24, then B1 192 after technical/resource gates. Do not launch B2/C/D/H model inference automatically. Completed historical C1/S1/P1 and blocked c1_20260928_v2 are read-only.

Local root D:/CEDAR/arr_next_steps_20261005.
Remote root /data4/WuMingrui/TianJinyu/CEDAR/arr_next_steps_20261005 is a symlink to /data1/WuMingrui/TianJinyu/CEDAR/arr_next_steps_20261005. Data4 had only17GB free; data1 had286GB. Only new output is redirected. Read weights and original inputs at /data4/WuMingrui/TianJinyu/CEDAR.

Campaign: /home/WuMingrui/miniconda3/envs/cedar/bin/python -u /data4/WuMingrui/TianJinyu/CEDAR/arr_next_steps_20261005/scripts/campaign.py --root /data4/WuMingrui/TianJinyu/CEDAR --out /data4/WuMingrui/TianJinyu/CEDAR/arr_next_steps_20261005

Check explore_outputs/campaign_status.json, runs/smoke/status.json or runs/B1/status.json, monitor_state.json, GPU budget and logs. Compare completed counts and log size; a live PID alone is not progress. Server snapshots every5min; external heartbeat every30min. Validate actual command/cwd/uid and both campaign/runner flock before any resume. Preserve all failures and orphan files. No duplicate if runner remains. Model and cap hashes verified before model load; all completed JSON/NPZ hashes must match on resume.

GPU allocation1,2,4,6,18GiB model maximum per card. Hard cumulative12h loading/inference, smoke3h, no-transition90min; output disk20GiB floor, input disk2GiB floor. Do not relax precision, sample selection, gates or budget to pass. If the first package cannot fit, report and preserve progress.

SSH host and port are the same authorized server and50000; use D:/CEDAR/arr_revision_20260927/known_hosts. Password only through SSH stdin; never in task/code/commands/logs. Recover the credential from this chat only, internally: read_thread's thread preview may contain a credential, so never print the full tool response or save it. Print only selected nonsecret fields. If needed use the original proxy helper; do not assume it is active.

Normal unchanged progress stays quiet. Notify on milestones, actual failure/repair, or action needed. After B1 verify24+192records, all hashes and automatic analysis, sync evidence to D, confirm no active managed processes and pause the new monitor. B2 remains a separate pending budget package with prespecified samples; completion of this first package is not completion of H or the paper.
