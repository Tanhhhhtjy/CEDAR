# Resource-policy amendment · 2026-10-05

Authorization: the user questioned the cumulative12-hour budget and requested removing it if no necessary reason existed. It was a conservative estimate-based spending cap (rough6.6h forecast plus margin), not a scientific requirement or hardware restriction.

Effective policy: remove the cumulative12h GPU loading/inference limit, the forecast-based refusal to start within that limit, and the analogous smoke3h cap. Runtime accounting and forecasts remain recorded and never reset. An experiment making verified progress must not be stopped merely for elapsed runtime.

Retain independent failure protections: no state transition90min, output filesystem20GiB floor, original input filesystem2GiB floor, memory availability, all technical/hash/finite-value gates, exclusive locks and task-specific process ownership checks. No changes to model, precision, tokens, caps, dose, samples, metrics, analysis or planned cell counts. This amendment changes resource policy only; it does not authorize automatic B2/C/D/H matrices.

The original frozen B_PROTOCOL.md, EXPERIMENT_PLAN.md, scripts and prepared manifest remain unchanged as historical provenance. This explicit amendment supersedes their runtime-budget clauses. New operational controller operations/campaign_runtime.py uses the original frozen scientific runner and analysis. It may take over the currently verified live runner after replacing only its old supervisor. All completed hashes and the same runner PID/starttime are verified before and after takeover. No completed cell is repeated solely to change a budget.

Do not launch the old scripts/campaign.py after this amendment; it still implements the archived limits. Use the canonical operations command in RUNBOOK.md. Operational controller and amendment hashes are separately recorded in operations/manifest.json.
