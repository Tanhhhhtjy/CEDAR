"""Read-only ARR reanalysis of the frozen CEDAR supporting package.

This does not implement non-option vocabulary prediction, activation probes,
or causal patching. Those require assets omitted from the supporting package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = list("ABCD")
SEED = 20260927
DRAWS = 20000
INPUTS = {}


def read(path):
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    INPUTS[str(path)] = {"sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows)}
    return rows


def indexed(rows, key):
    result = {key(r): r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate record key")
    return result


def summary(rows, metric, cluster):
    groups = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is not None and np.isfinite(v):
            groups[str(r[cluster])].append(float(v))
    if not groups:
        return {"n": 0, "clusters": 0, "mean": None, "ci95": None}
    ordered = [groups[k] for k in sorted(groups)]
    counts = np.array([len(v) for v in ordered])
    sums = np.array([sum(v) for v in ordered])
    means = sums / counts
    rng = np.random.default_rng(SEED)
    estimates, equal_weight = [], []
    for start in range(0, DRAWS, 500):
        indices = rng.integers(0, len(groups), (min(500, DRAWS - start), len(groups)))
        estimates.extend(sums[indices].sum(1) / counts[indices].sum(1))
        equal_weight.extend(means[indices].mean(1))
    return {
        "n": int(counts.sum()), "clusters": len(groups),
        "mean": float(sums.sum() / counts.sum()),
        "ci95": np.quantile(estimates, [.025, .975]).tolist(),
        "equal_cluster_mean": float(means.mean()),
        "equal_cluster_ci95": np.quantile(equal_weight, [.025, .975]).tolist(),
        "cluster_sizes": {"min": int(counts.min()), "max": int(counts.max())},
    }


def vector(row, field):
    obj = row.get(field)
    if not obj or not obj.get("logits"):
        return None
    v = np.array([obj["logits"][k] for k in LABELS], dtype=float)
    return v if np.isfinite(v).all() else None


def margins(v, initial, target):
    i, t = LABELS.index(initial), LABELS.index(target)
    others = [k for k in range(4) if k not in (i, t)]
    return np.mean(np.delete(v, i)) - v[i], v[t] - np.mean(v[others])


def affine(x, y):
    xc = x - np.mean(x)
    denom = float(xc @ xc)
    if denom < 1e-12:
        return None
    slope = float(xc @ (y - np.mean(y)) / denom)
    return slope, float(np.mean(y) - slope * np.mean(x))


def social(package):
    results, detail = {}, []
    for mode in ("direct", "free"):
        u = indexed(read(package / f"runs/social/{mode}-uncapped/results.jsonl"), lambda r: r["id"])
        p = indexed(read(package / f"runs/social/{mode}-p25/results.jsonl"), lambda r: r["id"])
        if u.keys() != p.keys():
            raise ValueError(f"Unmatched Social {mode} record IDs")
        rows = []
        field = "answer_logits" if mode == "direct" else "actual_answer_logits"
        for key in sorted(u):
            a, b = u[key], p[key]
            for k in ("base_id", "arm", "normalized_stem_id", "initial_answer", "target_answer", "prompt_sha256"):
                if a[k] != b[k]:
                    raise ValueError(f"Social pairing drift: {key}/{k}")
            x, y = vector(a, field), vector(b, field)
            r = {"id": key, "mode": mode, "base_id": a["base_id"], "arm": a["arm"], "stem": a["normalized_stem_id"]}
            if x is not None and y is not None:
                mu, su = margins(x, a["initial_answer"], a["target_answer"])
                mp, sp = margins(y, a["initial_answer"], a["target_answer"])
                r.update(delta_broad=float(mp-mu), delta_specific=float(sp-su))
            if mode == "direct":
                if x is None or y is None:
                    raise ValueError("Missing direct logits")
                au, ap = LABELS[int(x.argmax())], LABELS[int(y.argmax())]
            else:
                au, ap = a.get("strict_answer"), b.get("strict_answer")
            r["delta_target_follow"] = float(ap == a["target_answer"]) - float(au == a["target_answer"])
            r["delta_invalid"] = float(ap not in LABELS) - float(au not in LABELS)
            r["delta_initial"] = float(ap == a["initial_answer"]) - float(au == a["initial_answer"])
            r["delta_other"] = float(ap in LABELS and ap not in (a["initial_answer"], a["target_answer"])) - float(au in LABELS and au not in (a["initial_answer"], a["target_answer"]))
            if abs(sum(r[k] for k in ("delta_target_follow", "delta_invalid", "delta_initial", "delta_other"))) > 1e-12:
                raise ValueError("Four-state probabilities do not sum to one")
            rows.append(r)
        metrics = ("delta_broad", "delta_specific", "delta_target_follow", "delta_invalid", "delta_initial", "delta_other")
        results[mode] = {"pairs": len(rows), "arms": {}}
        for arm in ("AO", "S"):
            selected = [r for r in rows if r["arm"] == arm]
            results[mode]["arms"][arm] = {k: summary(selected, k, "stem") for k in metrics}
        paired = defaultdict(dict)
        for r in rows:
            paired[r["base_id"]][r["arm"]] = r
        did = []
        for base, cells in paired.items():
            if set(cells) != {"AO", "S"} or cells["AO"]["stem"] != cells["S"]["stem"]:
                raise ValueError("Social AO/S pairing drift")
            d = {"base_id": base, "stem": cells["S"]["stem"]}
            d.update({k: cells["S"][k]-cells["AO"][k] for k in metrics if k in cells["S"] and k in cells["AO"]})
            did.append(d)
        results[mode]["cap_effect_S_minus_AO"] = {k: summary(did, k, "stem") for k in metrics}
        detail.extend(rows)
    return results, detail


def replay(package):
    rows = read(package / "runs/fixed_trace_replay/formal-fixed-trace-seed0/results.jsonl")
    cells = indexed(rows, lambda r: (r["trace_id"], r["forward_condition"]))
    pairs = []
    for (tid, cond), u in cells.items():
        if cond != "uncapped":
            continue
        p = cells.get((tid, "full_p25"))
        if p is None:
            raise ValueError("Missing replay partner")
        for k in ("source_id", "family_id", "trace_source", "prompt_sha256", "trace_prefix_token_ids_sha256", "candidate_token_ids"):
            if u[k] != p[k]:
                raise ValueError(f"Replay drift: {tid}/{k}")
        x, y = vector(u, "logit_diagnostics"), vector(p, "logit_diagnostics")
        if x is None or y is None:
            raise ValueError("Missing replay logits")
        pairs.append({"trace_id": tid, "family": u["family_id"], "arm": u["arm"], "trace_source": u["trace_source"], "x": x, "y": y})
    detail = []
    for r in pairs:
        x, y = r["x"], r["y"]
        fit = affine(x, y)
        if fit is None:
            raise ValueError("Degenerate four-option fit")
        a, b = fit
        d = {k:v for k,v in r.items() if k not in ("x", "y")}
        d.update(slope=a, train_mae=float(np.abs(y-(a*x+b)).mean()), argmax_flip=float(x.argmax()!=y.argmax()))
        # A leave-one-option-out diagnostic has only three fitting points;
        # it is not the requested non-option-vocabulary confirmation.
        predictions = []
        for k in range(4):
            keep = np.arange(4) != k
            fold = affine(x[keep], y[keep])
            predictions.append(None if fold is None else fold[0]*x[k]+fold[1])
        d["lo_option_valid_folds"] = sum(v is not None for v in predictions)
        d["lo_option_mae"] = float(np.mean([abs(y[k]-v) for k,v in enumerate(predictions) if v is not None])) if any(v is not None for v in predictions) else None
        d["lo_option_complete_mae"] = d["lo_option_mae"] if d["lo_option_valid_folds"] == 4 else None
        # Strict family-out global positive scale. Intercepts cancel after
        # centering. Same trace source and arm; no test-family capped logits fit.
        train = [s for s in pairs if s["family"] != r["family"] and s["trace_source"] == r["trace_source"] and s["arm"] == r["arm"]]
        tx = np.concatenate([s["x"]-s["x"].mean() for s in train])
        ty = np.concatenate([s["y"]-s["y"].mean() for s in train])
        scale = max(1e-8, float(tx @ ty / (tx @ tx)))
        xc, yc = x-x.mean(), y-y.mean()
        d.update(family_out_scale=scale, family_out_centered_mae=float(np.abs(yc-scale*xc).mean()), identity_centered_mae=float(np.abs(yc-xc).mean()))
        d["family_out_minus_identity_mae"] = d["family_out_centered_mae"] - d["identity_centered_mae"]
        detail.append(d)
    results = {}
    for ts in sorted({r["trace_source"] for r in detail}):
        results[ts] = {}
        for arm in ("ALL", "N", "R", "V", "F"):
            chosen = [r for r in detail if r["trace_source"] == ts and (arm == "ALL" or r["arm"] == arm)]
            results[ts][arm] = {k:summary(chosen, k, "family") for k in ("slope", "train_mae", "lo_option_complete_mae", "family_out_centered_mae", "identity_centered_mae", "family_out_minus_identity_mae", "argmax_flip")}
            results[ts][arm]["incomplete_leave_option_rows"] = sum(r["lo_option_valid_folds"] < 4 for r in chosen)
    return results, detail


def propagation(package):
    rows = read(package / "runs/downstream_propagation/propagation_metrics.jsonl")
    groups = defaultdict(list)
    for r in rows:
        if r["layer"] in (53, 60, 63):
            groups[(r["trace_source"], r["treatment_condition"], r["layer"])].append(r)
    results = {}
    for (ts, condition, layer), group in sorted(groups.items()):
        results[f"{ts}/{condition}/L{layer}"] = {k:summary(group, k, "family_id") for k in ("delta_norm", "aligned_norm", "off_direction_norm", "aligned_share", "off_direction_share", "signed_aligned_projection", "relative_delta")}
    return results


def self_check():
    x = np.array([4., 1., -2., 0.])
    y = .6*x + 2
    a,b = affine(x,y)
    assert np.allclose(a*x+b,y) and x.argmax()==y.argmax()
    for k in range(4):
        keep=np.arange(4)!=k
        a,b=affine(x[keep],y[keep])
        assert np.isclose(a*x[k]+b,y[k])
    assert affine(np.ones(3),np.arange(3.)) is None
    result=summary([{"g":"a","v":1},{"g":"a","v":1},{"g":"b","v":-1}],"v","g")
    assert np.isclose(result["mean"],1/3) and result["equal_cluster_mean"]==0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--package",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.package.resolve() == args.output.resolve() or args.package.resolve() in args.output.resolve().parents:
        raise ValueError("Output must be outside the immutable supporting package")
    self_check()
    s,sd=social(args.package)
    r,rd=replay(args.package)
    p=propagation(args.package)
    result={"status":"completed_existing_data_reanalysis_only", "seed":SEED,"bootstrap_draws":DRAWS,"bootstrap":"paired percentile cluster bootstrap; row-weighted primary and equal-cluster sensitivity", "social":s,"fixed_trace_option_only":r,"rg1_absolute_components":p,"input_manifest":INPUTS,
        "limits":["Post-hoc analysis of existing development data, not an independent confirmation.","Option-only leave-out tests do not replace held-out non-option-token prediction.","No full-vocabulary logits, activation probes, patches, or new GPU forwards were generated.","RG1 random controls have different starting alignment; absolute reanalysis does not repair that control design.","Free-generation logit comparisons include trajectory changes; fixed-trace conclusions must use the separate replay assay."]}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"CPU_REANALYSIS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    for name,data in [("social_paired_deltas",sd),("replay_option_predictions",rd)]:
        (args.output/f"{name}.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,allow_nan=False)+"\n" for x in data),encoding="utf8")
    print(json.dumps({"status":result["status"],"social_paired_rows":len(sd),"replay_pairs":len(rd),"output":str(args.output)},ensure_ascii=False))


if __name__=="__main__":
    main()
