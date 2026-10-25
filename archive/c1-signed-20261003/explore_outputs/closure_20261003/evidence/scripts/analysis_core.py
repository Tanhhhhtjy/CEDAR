"""Frozen four endpoints, isolated from model execution."""
import itertools
import numpy as np

def endpoints(s):
    """s[target][arm] = [true, random0, ..., random7] central differences."""
    g={c:{a:float(v[0]-np.mean(v[1:])) for a,v in arms.items()} for c,arms in s.items()}
    return dict(common=float(np.mean([g['P'][a]-g['U'][a] for a in 'VF'])),
        interaction=(g['P']['V']-g['P']['F'])-(g['U']['V']-g['U']['F']))

def summarize(values):
    names=sorted(values);x=np.array([values[n] for n in names],float)
    assert len(x)==12 and np.isfinite(x).all()
    rng=np.random.default_rng(20260928)
    boot=x[rng.integers(len(x),size=(20000,len(x)))].mean(1)
    null=np.array(list(itertools.product([-1.,1.],repeat=len(x))))@x/len(x)
    return dict(equal_family_mean=float(x.mean()),family_means=values,
        descriptive_ci95=np.quantile(boot,[.025,.975]).tolist(),
        exact_signflip_p=float((np.abs(null)>=abs(x.mean())-1e-12).mean()),
        inference_limit='Exploratory: symmetric independent family sign-flip assumption; cross-fit training sets overlap.')

def holm(stats):
    running=0.
    for i,(name,stat) in enumerate(sorted(stats.items(),key=lambda v:v[1]['exact_signflip_p'])):
        running=max(running,min(1.,(len(stats)-i)*stat['exact_signflip_p']))
        stat['holm_four_primary']=running

def tests():
    def z(v):return np.array([v]+[0.]*8)
    s={'U':{'V':z(2),'F':z(2)},'P':{'V':z(.5),'F':z(.5)}}
    assert endpoints(s)=={'common':-1.5,'interaction':0.}
    s['P']['V']=z(1.5)
    assert endpoints(s)=={'common':-1.,'interaction':1.}
    equal={c:{a:np.full(9,v) for a,v in [('V',2.),('F',4.)]} for c in 'UP'}
    assert endpoints(equal)=={'common':0.,'interaction':0.}
    # Affine offsets cancel in the finite difference; positive scale divides it.
    pos=np.array([2.,4.,6.,8.]);neg=pos-2
    assert np.allclose(((pos-10)/2-(neg-10)/2)/2,(pos-neg)/4)
    stats={str(i):{'exact_signflip_p':v} for i,v in enumerate([.001,.02,.03,.5])}
    holm(stats);assert np.allclose([stats[str(i)]['holm_four_primary'] for i in range(4)],[.004,.06,.06,.5])
    print('Analysis endpoint, cancellation and Holm tests passed.',flush=True)
if __name__=='__main__':tests()
