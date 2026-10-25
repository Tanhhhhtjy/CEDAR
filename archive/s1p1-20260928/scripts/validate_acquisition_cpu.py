"""Engineering-only end-to-end check on a tiny random Qwen3; no scientific result."""
from pathlib import Path
import sys
import json
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
import run_arr_acquisition as acq
from cedar_m2.interventions.capping import LayerCap


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    meta = json.loads((acq.ROOT / 'explore_outputs/asset_paths.json').read_text())
    tok = AutoTokenizer.from_pretrained(meta['model'], local_files_only=True)
    tok.padding_side = 'left'
    tok.pad_token = tok.eos_token
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=32,
        intermediate_size=64, num_hidden_layers=64, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id, attention_dropout=0.0)).eval()
    caps = {layer: LayerCap(layer, torch.randn(32), -0.25) for layer in range(46, 54)}
    messages = [{'role': 'user', 'content': 'Choose A or B.'}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    prefix = tok.encode('Answer:', add_special_tokens=False)
    label_ids = {x: tok.encode(x, add_special_tokens=False)[0] for x in 'ABCD'}
    tr = dict(trace_id='engineering-only', source_id='engineering-only', trace_source='uncapped',
        source_condition='uncapped', family_id='synthetic', base_id='synthetic', structure_class='synthetic',
        variant='synthetic', arm='N', initial_answer='A', target_answer='B', gold_answer='A',
        prompt_sha256=acq.historical.object_sha256(prompt), prompt_token_ids_sha256=acq.historical.object_sha256(ids),
        trace_prefix_text_sha256=acq.historical.object_sha256('Answer:'),
        trace_prefix_token_ids_sha256=acq.historical.object_sha256(prefix), trace_prefix_token_ids=prefix,
        teacher_forced_input_token_ids_sha256=acq.historical.object_sha256(ids+prefix),
        generated_label_token_id=label_ids['A'], generated_label='A', candidate_token_ids=label_ids,
        anchor_record_seed=0, anchor_answer_logits={})
    rows = {'engineering-only': {'messages': messages}}
    states = {}
    for condition in ['uncapped', 'full_p25']:
        baseline = acq.historical.run_one(model,tok,acq.transformers,caps,rows,tr,condition)
        capture, original = {}, acq.historical.v1run.diagnostics
        def diag(z, trace, tokenizer):
            capture['logits'] = z.clone()
            return original(z,trace,tokenizer)
        acq.historical.v1run.diagnostics = diag
        try:
            with acq.BoundaryObserver(model) as observer:
                wrapped = acq.historical.run_one(model,tok,acq.transformers,caps,rows,tr,condition)
            assert wrapped['logit_diagnostics'] == baseline['logit_diagnostics'], 'Observation changed readout'
            with torch.inference_mode():
                error = acq.probe_reconstruct(model, observer, capture['logits'])
            assert error < 1e-6, error
            states[condition] = acq.numpy_state(observer,capture['logits'])
            assert len(states[condition]) == 11
            assert all(len(layer._forward_hooks)==0 and len(layer._forward_pre_hooks)==0 for layer in model.model.layers)
        finally:
            acq.historical.v1run.diagnostics = original
    with torch.inference_mode():
        norms, cf = acq.counterfactual_norm(model, states['uncapped'], states['full_p25'],model.model.norm.variance_epsilon)
    assert norms['rms_u'] > 0 and norms['rms_p'] > 0
    assert all(v.shape == (len(tok),) for v in cf.values())
    acq.write(acq.ROOT/'explore_outputs/CPU_ENGINEERING_TEST.json',dict(
        passed=True, model='random Qwen3, hidden_size=32, 64 layers, CPU float32',
        scientific_evidence=False, tests=['observer leaves original paired logits unchanged',
        'cache-faithful forced token replay', 'prompt and answer boundaries captured',
        'postnorm reconstruction', 'hook cleanup', 'RMS denominator counterfactual']))
    print('PASS: CPU engineering checks; this is not a real-model experiment.', flush=True)


if __name__=='__main__':
    main()
