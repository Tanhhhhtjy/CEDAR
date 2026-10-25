"""S1/P1 acquisition using the submitted cache-faithful E2B runner unchanged.

The intervention, prompt, token sequence, precision and attention implementation
come from the archived experiment. New hooks only observe three residual states
and terminal normalization. This is new acquisition, not historical reproduction
until the independent replay comparison has been assessed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'CEDAR_SUPPORTING_DATA'
CODE = PACKAGE / 'code'
for p in [CODE / 'src', CODE / 'scripts']:
    sys.path.insert(0, str(p))
import numpy as np
import torch
import transformers
import run_e2b_fixed_trace_cached as historical
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model

LAYERS = (53, 60, 63)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def snapshots(out, **arrays):
    tmp = out.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out)


class BoundaryObserver:
    def __init__(self, model):
        self.model = model
        self.first, self.last, self.handles = {}, {}, []

    def save(self, key, value):
        if isinstance(value, (tuple, list)):
            value = value[0]
        # Keep on device while decoding: no per-token host synchronization.
        item = value[:, -1:, :].detach().clone()
        self.last[key] = item
        if key not in self.first:
            self.first[key] = item

    def __enter__(self):
        for layer in LAYERS:
            if layer == 53:
                # Observe layer-54 input, after the cap hook on layer 53.
                self.handles.append(self.model.model.layers[54].register_forward_pre_hook(
                    lambda _m, i: self.save('layer_53', i[0])))
            else:
                self.handles.append(self.model.model.layers[layer].register_forward_hook(
                    lambda _m, _i, o, k=f'layer_{layer}': self.save(k, o)))
        self.handles.append(self.model.model.norm.register_forward_pre_hook(
            lambda _m, i: self.save('prenorm', i[0])))
        self.handles.append(self.model.model.norm.register_forward_hook(
            lambda _m, _i, o: self.save('postnorm', o)))
        return self

    def __exit__(self, *_):
        for h in self.handles:
            h.remove()


def numpy_state(observer, logits):
    result = {'logits': logits.detach().float().cpu().numpy()}
    for position, values in [('prompt', observer.first), ('answer', observer.last)]:
        for key, v in values.items():
            result[f'{position}_{key}'] = v.reshape(-1).float().cpu().numpy()
    return result


def probe_reconstruct(model, observer, logits):
    norm = observer.last['postnorm']
    device = model.lm_head.weight.device
    reconstructed = model.lm_head(norm.to(device)).reshape(-1).float()
    return float((reconstructed - logits.to(device).float()).abs().max())


def counterfactual_norm(model, arrays_u, arrays_p, epsilon):
    device, dtype = model.model.norm.weight.device, model.model.norm.weight.dtype
    weight = model.model.norm.weight
    hu = torch.tensor(arrays_u['answer_prenorm'], device=device, dtype=dtype)
    hp = torch.tensor(arrays_p['answer_prenorm'], device=device, dtype=dtype)
    du = (hu.float().square().mean() + epsilon).sqrt()
    dp = (hp.float().square().mean() + epsilon).sqrt()
    result = dict(rms_u=float(du), rms_p=float(dp))
    logits = {}
    for key, h, denominator in [('p_with_u_denominator', hp, du), ('u_with_p_denominator', hu, dp)]:
        post = weight * (h.float() / denominator).to(dtype)
        z = model.lm_head(post.reshape(1, 1, -1).to(model.lm_head.weight.device))
        logits[key] = z.reshape(-1).float().cpu().numpy()
    return result, logits


def main(args):
    out = ROOT / 'runs' / args.scope
    out.mkdir(parents=True, exist_ok=True)
    status = out / 'status.json'
    assets = json.loads((ROOT / 'explore_outputs/assets_status.json').read_text())
    assert assets['status'] == 'ready', 'Weights must finish SHA-256 verification before acquisition'
    traces = readl(ROOT / 'inputs/frozen_traces192.jsonl')
    archived = readl(PACKAGE / 'runs/fixed_trace_replay/formal-fixed-trace-seed0/results.jsonl')
    anchors = {(r['trace_id'], r['forward_condition']): r for r in archived}
    assert len(traces) == 192 and len(anchors) == 384
    for t in traces:
        anchor = anchors[t['trace_id'], 'uncapped']
        for key in ['prompt_sha256', 'trace_prefix_token_ids_sha256', 'teacher_forced_input_token_ids_sha256']:
            assert t[key] == anchor[key], (t['trace_id'], key)
        assert historical.object_sha256(t['trace_prefix_token_ids']) == t['trace_prefix_token_ids_sha256']
    prompts = {r['id']: r for r in readl(PACKAGE / 'data/phase_validity_inputs/answer_inputs96.jsonl')}
    # Smoke is the lexicographically first base, all N/R/V/F, uncapped-source.
    # It is a numerical/engineering check and is excluded from claims of confirmation.
    first_base = sorted(set(t['base_id'] for t in traces))[0]
    if args.scope == 'smoke':
        traces = [t for t in traces if t['base_id'] == first_base and t['source_condition'] == 'uncapped']
        assert len(traces) == 4 and {t['arm'] for t in traces} == set('NRVF')
    else:
        smoke = json.loads((ROOT / 'runs/smoke/gate.json').read_text())
        assert smoke['technical_pass'], 'Formal acquisition requires the paired smoke gate'
    traces.sort(key=lambda t: (t['source_condition'] != 'uncapped', t['source_id']))
    fingerprint = dict(script_sha256=sha(__file__), trace_sha256=sha(ROOT / 'inputs/frozen_traces192.jsonl'),
        supporting_archive_sha256=sha(ROOT / 'CEDAR_SUPPORTING_DATA.zip'),
        prompt_sha256=sha(PACKAGE / 'data/phase_validity_inputs/answer_inputs96.jsonl'),
        protocol_sha256=sha(ROOT / 'PROTOCOL.md'), source_runner_sha256=sha(CODE / 'scripts/run_e2b_fixed_trace_cached.py'),
        model_revision=assets['revision'], torch=torch.__version__, transformers=transformers.__version__,
        scope=args.scope, gpu_memory_gib=args.gpu_memory_gib, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == fingerprint, 'Refusing mixed-provenance resume'
    else:
        write(manifest_path, fingerprint)
    write(status, dict(stage='loading_model', completed=0, traces=len(traces), pid=os.getpid(), started=time.time()))
    caps, cap_metadata = historical.e2run.load_p25_assets(ROOT / 'cache/hub')
    model, tokenizer = load_qwen_model(QwenModelSpec(repo_id=assets['model']),
        max_memory={i: f'{args.gpu_memory_gib}GiB' for i in range(torch.cuda.device_count())}, local_files_only=True)
    assert all(str(d) not in ('cpu', 'disk') for d in model.hf_device_map.values()), 'Unexpected model offload'
    assert len(model.model.layers) == 64
    excluded = set(tokenizer.all_special_ids)
    for label in 'ABCD':
        for text in [label, ' ' + label, '\n' + label, '\t' + label]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                excluded.add(ids[0])
    eps = model.model.norm.variance_epsilon
    write(out / 'runtime.json', dict(device_map=model.hf_device_map, cap_assets=cap_metadata, norm_epsilon=eps,
        excluded_fit_token_ids=sorted(excluded), selected_layers=list(LAYERS),
        hardware=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        torch=torch.__version__, transformers=transformers.__version__))
    np.save(out / 'norm_weight.npy', model.model.norm.weight.detach().float().cpu().numpy())
    results = []
    total = len(traces) * (3 if args.scope == 'smoke' else 2)
    for index, trace in enumerate(traces):
        arrays = {}
        for condition in ['uncapped', 'full_p25'] + (['uncapped_repeat'] if args.scope == 'smoke' else []):
            name = f'{index:04d}_{condition}'
            record_path, array_path = out / (name + '.json'), out / (name + '.npz')
            if record_path.exists() and array_path.exists():
                record = json.loads(record_path.read_text())
                assert record['trace_id'] == trace['trace_id']
                assert record['arrays_sha256'] == sha(array_path)
                arrays[condition] = dict(np.load(array_path))
                results.append(record)
                continue
            base_condition = 'uncapped' if condition == 'uncapped_repeat' else condition
            original_diagnostics = historical.v1run.diagnostics
            captured = {}
            def capture_logits(z, row, tok):
                captured['logits'] = z.detach().clone()
                return original_diagnostics(z, row, tok)
            historical.v1run.diagnostics = capture_logits
            start = time.monotonic()
            try:
                with BoundaryObserver(model) as observer:
                    record = historical.run_one(model, tokenizer, transformers, caps, prompts, trace, base_condition)
                with torch.inference_mode():
                    reconstruction_error = probe_reconstruct(model, observer, captured['logits'])
                values = numpy_state(observer, captured['logits'])
            finally:
                historical.v1run.diagnostics = original_diagnostics
            assert all(np.isfinite(v).all() for v in values.values()), 'Nonfinite captured states'
            record['layer_53_endpoint'] = 'layer_54_input_after_layer_53_cap'
            record['condition'] = condition
            record['reconstruction_max_abs'] = reconstruction_error
            record['seconds'] = time.monotonic() - start
            z_old = anchors[trace['trace_id'], base_condition]['logit_diagnostics']['logits']
            z_new = record['logit_diagnostics']['logits']
            record['historical_option_max_abs'] = max(abs(z_new[k] - z_old[k]) for k in 'ABCD')
            record['peak_gpu_allocated_bytes'] = [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
            snapshots(array_path, **values)
            record['arrays_sha256'] = sha(array_path)
            write(record_path, record)
            arrays[condition] = values
            results.append(record)
            write(status, dict(stage='acquiring', completed=len(results), total=total, pid=os.getpid(),
                last_cell=record['cell_id'], last_seconds=record['seconds'], updated=time.time()))
            print(json.dumps(dict(completed=len(results), total=total, cell=name, seconds=record['seconds'],
                historical_error=record['historical_option_max_abs'], reconstruction_error=reconstruction_error)), flush=True)
        with torch.inference_mode():
            norms, cf = counterfactual_norm(model, arrays['uncapped'], arrays['full_p25'], eps)
        snapshots(out / f'{index:04d}_norm_counterfactual.npz', **cf)
        write(out / f'{index:04d}_norm_counterfactual.json', dict(trace_id=trace['trace_id'], **norms))
    write(out / 'results.json', results)
    if args.scope == 'smoke':
        errors = []
        for i in range(len(traces)):
            u = dict(np.load(out / f'{i:04d}_uncapped.npz'))
            r = dict(np.load(out / f'{i:04d}_uncapped_repeat.npz'))
            errors.append(max(float(np.max(np.abs(u[k] - r[k]))) for k in u))
        reconstruction_max = max(r['reconstruction_max_abs'] for r in results)
        historical_max = max(r['historical_option_max_abs'] for r in results)
        gate = dict(technical_pass=max(errors) <= 1e-4 and reconstruction_max <= 0.25,
            repeat_all_states_max_abs=max(errors), postnorm_logit_reconstruction_max_abs=reconstruction_max,
            historical_option_max_abs=historical_max, historical_exact_reproduction=historical_max <= 1e-4,
            note='A technical pass allows paired acquisition on this runtime; it does not certify historical identity or a mechanism.')
        write(out / 'gate.json', gate)
        assert gate['technical_pass'], gate
    write(status, dict(stage='complete', completed=len(results), total=total, updated=time.time(), pid=os.getpid()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scope', choices=['smoke', 'formal'], required=True)
    p.add_argument('--gpu-memory-gib', type=int, default=18)
    args = p.parse_args()
    try:
        main(args)
    except Exception as e:
        write(ROOT / 'runs' / args.scope / 'failure.json', dict(error=repr(e), traceback=traceback.format_exc(), time=time.time()))
        traceback.print_exc()
        raise
