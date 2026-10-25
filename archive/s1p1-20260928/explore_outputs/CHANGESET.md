# ARR supplemental acquisition changes

Current research: distinguish general readout scale from reason-related changes under official p25. Remote experiment branch: `explore/2026-09-27-arr-s1-p1`, rooted in the newly isolated CEDAR directory. The historical supporting package and checkouts are input sources.

`run_arr_acquisition.py` imports the submitted cache-faithful runner unchanged. Observation hooks save prompt/answer-boundary states, full-vocabulary logits, normalization inputs/outputs and denominator-swap readouts. `analyze_s1.py` implements non-option fit/test token separation and family-out fits. `analyze_p1.py` implements bidirectional family-held-out probes with training-only preprocessing. `run_campaign.py` orders verified assets, smoke, acquisition, and analysis with explicit failure states.

Rollback: stop only the recorded campaign/child PID belonging to this workspace and retain all generated evidence; historical code and results require no rollback. New code is candidate research instrumentation, not an established contribution.

2026-09-27 engineering repair: a requests ChunkedEncodingError interrupted the weight stream. Preserve initial failure evidence and partial blobs; retry transient network errors up to 16 attempts with bounded backoff, protect the downloader with a process lock, and account for partial downloads in the disk-space check. Integrity errors still stop immediately. This changes asset transport only, not experiment semantics.
