# Comparability anchors and limits

- Inputs: all 192 original traces match submitted E2B prompt/prefix/concatenated-token hashes; 384 historical paired records are retained as anchors.
- Qwen3-32B, pinned revision, BF16, SDPA, Transformers 4.56.1 and Torch 2.6.0+cu124. Cap files are verified against original SHA-256 values. All 17 weight shards must pass LFS SHA-256 verification before loading.
- Physical GPU order and per-device model memory budget follow the archived E2B run. Actual model device mapping, shared-job conditions and replay errors remain explicit.
- New capture hooks were compared against the original runner on a random tiny CPU Qwen3. This validates instrumentation only; real-model equivalence is tested by smoke.
- Fixed-trace option argmax is not free-generation behavior. Decodable V/F labels are not proof of semantic validity or causal use. New-salt data are not structurally independent confirmation.
- No confirmation claim or ARR-readiness claim is authorized by a running process. C1/H1 are pending follow-up evidence.
