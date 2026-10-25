"""Independent synthetic checks for holdout prediction and fold leakage."""
from pathlib import Path
import json
import numpy as np
from analyze_s1 import token_split,sufficient,fit,margin
from analyze_p1 import evaluate


def main():
    rng=np.random.default_rng(71)
    x=rng.normal(size=10000);y=.4*x+2.
    excluded={1,2,3,4}
    split=token_split(x,excluded,'synthetic')
    a,b=fit(sufficient(x,y,split))['affine']
    assert np.allclose([a,b],[.4,2.])
    altered=y.copy();altered[list(excluded)]+=100
    assert np.allclose(fit(sufficient(x,altered,split))['affine'],[a,b]), 'Options contaminated fit'
    ids=[1,2,3,4]
    assert x[ids].argmax()==y[ids].argmax(), 'Positive affine argmax invariance'
    for kind,val in margin(x[ids],'A','B').items():
        assert abs(margin((y[ids]-b)/a,'A','B')[kind]-val)<1e-12
    rows=[dict(family_id=f'f{f}',base_id=f'b{f}_{base}',initial_answer='A',target_answer='B',arm=arm)
        for f in range(3) for base in range(2) for arm in ['V','F']]
    labels=np.array([int(r['arm']=='V') for r in rows])
    u=rng.normal(size=(len(rows),8));u[:,0]+=labels*3
    features=dict(U=u,P=.7*u)
    before=evaluate(features,rows,labels,'raw',True)
    changed=labels.copy();changed[:4]=1-changed[:4]
    after=evaluate(features,rows,changed,'raw',True)
    for key in before:
        assert np.array_equal(before[key]['scores'][:4],after[key]['scores'][:4]), 'Held-out labels entered training'
    out=Path(__file__).resolve().parents[1]/'explore_outputs/CPU_ANALYSIS_TEST.json'
    out.write_text(json.dumps(dict(passed=True,scientific_evidence=False,tests=[
        'known affine map recovered on non-option tokens','altering held-out options cannot affect fit',
        'positive affine argmax invariance','scale-restored margins recover known truth',
        'changing held-out family labels leaves its predictions unchanged']),indent=2)+'\n')
    print('PASS: synthetic holdout and leakage tests; not scientific evidence.')


if __name__=='__main__':main()
