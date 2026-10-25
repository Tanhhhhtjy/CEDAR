# C1 status and safe continuation

Local root: D:/CEDAR/arr_c1_20260928. Server root: /data4/WuMingrui/TianJinyu/CEDAR/c1_20260928_v2. All local output belongs on D:.

As of 2026-09-28 13:49 Beijing, campaign PID 1351122 has exited; the server audit found no task process. CPU tests and existing-state diagnostics completed. The norm preflight rejected 14 main cells. **Do not restart campaign.py as a recovery from this scientific-design block.** It will repeat preparation and hit the same gate; there are no interrupted v2 GPU cells to resume.

Read explore_outputs/STATUS_AND_FINDINGS.md and the saved blocked_audit JSON. Preserve original PROTOCOL.md, scripts, prepared/manifest.json, all logs and gates. Signed negative random controls also need correction; their current norm-check pass does not validate the intended signed design.

Next action must use a new prospective protocol and isolated run directory. First audit all proposed signed controls on cached states, then rehash original model/cap assets and verify numerical, no-op, repeat and hook gates on the proposed small smoke. Never lower the 0.5% realized-norm tolerance, change BF16, skip rows, choose new random directions after outcomes, or label a partial matrix complete.

The independent signed-control candidate CPU audit has now passed 3840 proposed cells (maximum relative norm error 0.00063055). This does not release v2 or authorize a new GPU campaign. Review explore_outputs/NEXT_STEP_PROPOSAL.md for the concrete proposed 52-forward technical smoke. Historical runtime is 1.385 hours before loading/overhead; the proposed hard limit is 6 hours. No automatic formal stage is proposed.

Use physical GPUs 1,2,4,6 with 18 GiB budgets and >=22 GiB free per card. Recheck shared disk >=20 GiB and declared time/storage budget. No other user's process may be stopped. Preserve our exclusive locks and inspect process ownership and cwd before launch.

There is currently no running experiment or active Codex monitor. On actual future launch, use the user's 30-minute interval and meaningful-change-only notifications. Stop monitoring on completion or a stable block requiring a scientific decision.

Remote complete audit snapshot: evidence_snapshot.tgz. Verified local metadata snapshot: evidence_metadata_20260928T054910Z.tgz and server_evidence_20260928T054910Z/. The .partial file is a preserved incomplete transfer, not usable evidence. Credentials are not saved in any artifact.
