"""Post-run audit only. Does not modify acquisition or original analysis outputs."""
from pathlib import Path
from collections import Counter, defaultdict
import datetime
import hashlib
import itertools
import json
import math
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'explore_outputs'
SEED = 20260927
checks = Counter()
hashes = {}


def read(p):
    return json.loads((ROOT / p).read_text())


def sha(p):
    p = ROOT / p
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    hashes[str(p.relative_to(ROOT))] = h.hexdigest()
    return h.hexdigest()


def check(ok, label):
    if not ok:
        raise AssertionError(label)
    checks[label] += 1


def close(a, b, label):
    check(bool(np.allclose(a, b, rtol=1e-10, atol=1e-10)), label)


def holm(values):
    out, last = {}, 0.
    for i, (k, p) in enumerate(sorted(values.items(), key=lambda kv: kv[1])):
        last = max(last, min(1., (len(values) - i) * p))
        out[k] = last
    return out


def family_stats(vals):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(SEED)
    boot = vals[rng.integers(len(vals), size=(20000, len(vals)))].mean(1)
    return dict(mean=float(vals.mean()), family_sd=float(vals.std(ddof=1)),
                ci95=np.quantile(boot, [.025, .975]).tolist())


def main():
    check(Path.cwd().resolve() == ROOT, 'correct_working_directory')
    campaign = read('explore_outputs/campaign_status.json')
    check(campaign['stage'] == 's1_p1_complete_c1_h1_pending', 'campaign_finished')
    assets = read('explore_outputs/assets_status.json')
    check(assets['status'] == 'ready', 'assets_previously_verified_ready')
    traces = [json.loads(s) for s in (ROOT/'inputs/frozen_traces192.jsonl').read_text().splitlines()]
    traces.sort(key=lambda t: (t['source_condition'] != 'uncapped', t['source_id']))
    check(len(traces) == 192 and len({t['trace_id'] for t in traces}) == 192, 'frozen_trace_count')
    fp = read('runs/formal/manifest.json')
    paths = dict(script_sha256='scripts/run_arr_acquisition.py',
                 trace_sha256='inputs/frozen_traces192.jsonl',
                 supporting_archive_sha256='CEDAR_SUPPORTING_DATA.zip',
                 prompt_sha256='CEDAR_SUPPORTING_DATA/data/phase_validity_inputs/answer_inputs96.jsonl',
                 protocol_sha256='PROTOCOL.md',
                 source_runner_sha256='CEDAR_SUPPORTING_DATA/code/scripts/run_e2b_fixed_trace_cached.py')
    for key, path in paths.items():
        check(sha(path) == fp[key], 'frozen_provenance_hash_match')
    check(fp['model_revision'] == assets['revision'] == '9216db5781bf21249d130ec9da846c4624c16137', 'pinned_model_revision')
    runtime = read('runs/formal/runtime.json')
    check(runtime['selected_layers'] == [53, 60, 63], 'prespecified_layers')
    check(fp['cuda_visible_devices'] == '1,2,4,6' and fp['gpu_memory_gib'] == 18, 'prespecified_gpu_budget')
    record_meta = {}
    for scope in ('smoke', 'formal'):
        folder = ROOT/'runs'/scope
        selected = traces if scope == 'formal' else [t for t in traces if t['base_id'] == min(t['base_id'] for t in traces) and t['source_condition'] == 'uncapped']
        conditions = ('uncapped', 'full_p25') if scope == 'formal' else ('uncapped', 'full_p25', 'uncapped_repeat')
        rows = read(f'runs/{scope}/results.json')
        check(len(rows) == len(selected)*len(conditions), 'result_count_'+scope)
        check(read(f'runs/{scope}/status.json')['stage'] == 'complete', 'run_complete_'+scope)
        reconstructed, historical, repeat = [], [], []
        for i, t in enumerate(selected):
            arrays = {}
            for j, c in enumerate(conditions):
                stem = f'runs/{scope}/{i:04d}_{c}'
                r = read(stem+'.json')
                check(rows[i*len(conditions)+j] == r, 'summary_equals_individual_record')
                check(r['trace_id'] == t['trace_id'] and r['condition'] == c, 'cell_identity')
                for key in ('prompt_sha256','trace_prefix_token_ids_sha256','teacher_forced_input_token_ids_sha256'):
                    check(r[key] == t[key], 'input_trajectory_hash_match')
                check(sha(stem+'.npz') == r['arrays_sha256'], 'recorded_array_sha256_match')
                sha(stem+'.json')
                with np.load(ROOT/(stem+'.npz'), allow_pickle=False) as z:
                    arrays[c] = {k:z[k] for k in z.files}
                required = {'logits'} | {f'{p}_{k}' for p in ('prompt','answer') for k in ('prenorm','postnorm','layer_53','layer_60','layer_63')}
                check(set(arrays[c]) == required, 'captured_endpoints_complete')
                check(all(np.isfinite(v).all() for v in arrays[c].values()), 'captured_arrays_finite')
                reconstructed.append(r['reconstruction_max_abs'])
                historical.append(r['historical_option_max_abs'])
            cfstem = f'runs/{scope}/{i:04d}_norm_counterfactual'
            cfmeta = read(cfstem+'.json')
            check(cfmeta['trace_id'] == t['trace_id'], 'counterfactual_pair_identity')
            with np.load(ROOT/(cfstem+'.npz'), allow_pickle=False) as cf:
                check(set(cf.files) == {'p_with_u_denominator','u_with_p_denominator'}, 'counterfactual_directions_complete')
                check(all(np.isfinite(cf[k]).all() and cf[k].shape == arrays['uncapped']['logits'].shape for k in cf.files), 'counterfactual_arrays_finite_shape')
            for ext in ('.json','.npz'): sha(cfstem+ext)
            if scope == 'smoke':
                repeat.extend(float(np.max(np.abs(arrays['uncapped'][k]-arrays['uncapped_repeat'][k]))) for k in arrays['uncapped'])
        check(max(reconstructed) <= .25, 'reconstruction_gate_'+scope)
        record_meta[scope] = dict(records=len(rows), traces=len(selected), reconstruction_max_abs=max(reconstructed), historical_option_max_abs=max(historical), historical_mismatch_records=sum(v>1e-4 for v in historical))
        if repeat:
            check(max(repeat) <= 1e-4, 'repeat_gate_rechecked')
            record_meta[scope]['repeat_all_states_max_abs'] = max(repeat)
        for p in ('manifest.json','runtime.json','results.json','status.json','norm_weight.npy'): sha(f'runs/{scope}/{p}')
    check(read('runs/smoke/gate.json')['technical_pass'], 'saved_smoke_gate_passed')
    s1 = read('analysis-output/formal/s1/S1_SUMMARY.json')
    details = read('analysis-output/formal/s1/S1_DETAILS.json')
    splits = read('analysis-output/formal/s1/token_splits.json')
    sel = read('analysis-output/formal/s1/S1_SELECTIVITY.json')
    check(len(details) == s1['trace_pairs'] == 192 and len(sel) == 48, 's1_coverage')
    check({d['trace_id'] for d in details} == {t['trace_id'] for t in traces} == set(splits), 's1_trace_coverage')
    for i, t in enumerate(traces):
        row = read(f'runs/formal/{i:04d}_uncapped.json')
        excluded = set(runtime['excluded_fit_token_ids']) | set(row['candidate_token_ids'].values())
        fits, tests = set(), set()
        for spec in splits[t['trace_id']].values():
            fits.update(spec['fit']); tests.update(spec['test'])
        check(not (fits & tests or fits & excluded or tests & excluded), 's1_token_split_disjoint_excludes_options')
    primary_s1 = s1['aggregate']['e2a_uncapped']['V_minus_F']
    s1p = {}
    for metric, entry in primary_s1.items():
        groups = defaultdict(list)
        for r in sel:
            if r['trace_source'] == 'e2a_uncapped': groups[r['family_id']].append(r[metric])
        fm = {k:float(np.mean(v)) for k,v in sorted(groups.items())}
        check(set(fm) == set(entry['family_means']) and len(fm) == 12, 's1_family_count')
        close(list(fm.values()), list(entry['family_means'].values()), 's1_family_means_recomputed')
        stats = family_stats(list(fm.values()))
        for k in ('mean','family_sd','ci95'): close(stats[k], entry[k], 's1_statistics_recomputed')
        values = np.array(list(fm.values()))
        null = np.array(list(itertools.product([-1.,1.], repeat=12)))@values/12
        s1p[metric] = float((np.abs(null) >= abs(values.mean())-1e-12).mean())
        close(s1p[metric], entry['p_exact_signflip'], 's1_signflip_recomputed')
    for k,p in holm(s1p).items(): close(p, primary_s1[k]['p_holm_four_selectivity_contrasts'], 's1_holm_recomputed')
    p1 = read('analysis-output/formal/p1/P1_SUMMARY.json')
    examples = read('analysis-output/formal/p1/examples.json')
    check(p1['n'] == len(examples) == 48 and p1['families'] == 12 and p1['bases'] == 24, 'p1_coverage')
    check(len(p1['results']) == 24, 'p1_prespecified_configurations')
    families = np.array([r['family_id'] for r in examples]); y = np.array([r['arm']=='V' for r in examples])
    for base in {r['base_id'] for r in examples}:
        pair = [r for r in examples if r['base_id']==base]
        check(len(pair)==2 and {r['arm'] for r in pair}=={'V','F'} and len({r['family_id'] for r in pair})==1, 'p1_vf_pairs_share_fold')
    for key, configs in p1['results'].items():
        check(set(configs)=={'U->U','U->P','P->U','P->P'}, 'p1_four_transfer_directions')
        for direction,r in configs.items():
            z = np.asarray(r['scores'])
            check(z.shape==(48,) and np.isfinite(z).all(), 'p1_scores_finite_complete')
            auc, ba = {}, {}
            for family in sorted(set(families)):
                mask = families==family
                auc[family] = roc_auc_score(y[mask], z[mask])
                ba[family] = balanced_accuracy_score(y[mask], z[mask]>0)
            close(list(auc.values()), list(r['family_auc'].values()), 'p1_family_auc_recomputed')
            close(np.mean(list(auc.values())), r['equal_family_auc'], 'p1_equal_family_auc_recomputed')
            close(np.mean(list(ba.values())), r['equal_family_balanced_accuracy'], 'p1_balanced_accuracy_recomputed')
            close(family_stats(list(auc.values()))['ci95'],r['descriptive_family_bootstrap_ci95'],'p1_descriptive_ci_recomputed')
    p1p, primary_p1 = {}, p1['results'][p1['primary']]
    for direction, null in p1['paired_label_permutation_null'].items():
        check(len(null)==100 and np.isfinite(null).all(), 'p1_100_null_results_per_direction')
        p1p[direction]=(1+sum(v>=primary_p1[direction]['equal_family_auc'] for v in null))/101
        close(p1p[direction], primary_p1[direction]['paired_label_permutation_p'], 'p1_permutation_p_recomputed')
    for k,p in holm(p1p).items(): close(p,primary_p1[k]['holm_p_four_transfer_directions'],'p1_holm_recomputed')
    direction_meta = list((ROOT/'analysis-output/formal/p1/directions').glob('*.json'))
    check(len(direction_meta)==24,'p1_24_candidate_directions')
    for f in direction_meta:
        d=json.loads(f.read_text())
        check(d['heldout_family'] not in d['training_families'] and len(d['training_families'])==11,'exported_direction_train_families_only')
        check(np.isfinite(np.load(f.with_suffix('.npy'),allow_pickle=False)).all(),'exported_direction_finite')
    for folder in ('analysis-output/formal','explore_outputs'):
        paths = (ROOT/folder).rglob('*') if folder.startswith('analysis') else [(ROOT/folder)/n for n in ('assets_status.json','campaign_status.json')]
        for p in paths:
            if p.is_file(): sha(p.relative_to(ROOT))
    for f in ('scripts/analyze_s1.py','scripts/analyze_p1.py','scripts/verify_batch_completion.py','runs/smoke/gate.json'): sha(f)
    primary_rows=[d for d in details if d['trace_source']=='e2a_uncapped']
    descriptive={}
    for source in ('e2a_uncapped','e2a_full_p25'):
        rows=[d for d in details if d['trace_source']==source]
        descriptive[source] = dict(option_argmax_flips=sum(d['option_argmax_flip'] for d in rows),n=len(rows),
            nonpositive_fits=sum(not v['positive_slope'] for d in rows for mode in ('same_prompt_fits','family_out_fits') for v in d[mode].values()))
        for mode in ('same_prompt_fits','family_out_fits'):
            descriptive[source][mode]={}
            for kind in ('identity','scale','affine'):
                groups=defaultdict(list)
                for d in rows: groups[d['family_id']].append(d[mode][kind]['options']['rmse'])
                descriptive[source][mode][kind]=family_stats([np.mean(v) for _,v in sorted(groups.items())])
    report = dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed',checks=dict(checks),
        records=record_meta,s1_primary=primary_s1,p1_primary={k:{a:b for a,b in v.items() if a!='scores'} for k,v in primary_p1.items()},
        descriptive_scale_prediction=descriptive,campaign=campaign,scientific_protocol_changed=False,
        limitations=['Model shard SHA256 verification was performed at acquisition startup; this audit reuses its ready manifest and pinned revision rather than rereading 65GB.',
                    'Counterfactual arrays had no acquisition-time expected SHA256; this audit checks identity/shape/finiteness and establishes post-run hashes.',
                    'This audit recomputes statistics from saved scores and family data; it does not rerun GPU inference or the 100 retrained permutation fits.',
                    'P1 sensitivity configurations use uncapped-source V/F traces; capped-source probe sensitivity and standalone nuisance-label decoding are not provided.',
                    'S1 family-out option predictions are saved; family-out non-option test errors are not separately emitted by the frozen analyzer.'],
        artifact_sha256=hashes)
    destination=OUT/'batch_integrity_20260928.json'
    destination.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],checks=dict(checks),records=record_meta,descriptive_scale_prediction=descriptive,
                         campaign_completed_at_utc=datetime.datetime.fromtimestamp(campaign['updated'],datetime.timezone.utc).isoformat(),
                         hash_entries=len(hashes),audit_file=str(destination)),indent=2),flush=True)


if __name__ == '__main__':
    main()
