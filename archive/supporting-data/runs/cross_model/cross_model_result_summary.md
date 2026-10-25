# Cross-model transport results

## Matched forced-choice geometry

- Qwen-FC overall delta m_general: `+2.075 [+1.372, +2.791]` logits.
- Llama overall delta m_general: `+1.391 [+0.908, +1.865]` logits.
- Qwen-FC V-minus-F continuous selectivity change: `+0.042 [-0.438, +0.573]` logits.
- Llama V-minus-F continuous selectivity change: `+0.115 [-0.365, +0.615]` logits.

Both model-specific configurations shift non-initial answer geometry in the same direction,
while neither continuous V-minus-F interval supports a reliable selectivity improvement.
The magnitude differs across models and is therefore reported separately rather than pooled.

## Discrete behavior and controls

- Qwen-FC: V target-follow 37.5% -> 50.0% (+12.5%); F 16.7% -> 33.3% (+16.7%).
- Qwen-FC: N/R option-argmax stability was 24/24 and 23/24.
- Llama: V target-follow 33.3% -> 50.0% (+16.7%); F 20.8% -> 33.3% (+12.5%).
- Llama: N/R option-argmax stability was 23/24 and 22/24.

The control arms show that a continuous shift can remain largely latent. The paired V/F
rates do not support interpreting the intervention as selectively accepting valid evidence
while rejecting invalid arguments.

## Scope

These estimates describe two audited model-specific configurations under the same forced-choice
regime. They do not define a population-level model effect or establish universality.
