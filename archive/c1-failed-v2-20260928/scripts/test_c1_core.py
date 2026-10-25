"""Engineering tests for schedule, quantization matching, and cleanup; no model results."""
import numpy as np
import torch
from torch import nn
from c1_core import make_patch, PromptPatch, hook_inventory
class Fake(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.Identity() for _ in range(55)])

def main():
    rng=np.random.default_rng(10928);checked=0;rejected=0
    for scale in [.1,1.,10.,100.]:
        h=torch.tensor(rng.normal(size=5120)*scale,dtype=torch.bfloat16).float().numpy()
        donor=torch.tensor(h+rng.normal(size=5120)*scale*.2,dtype=torch.bfloat16).float().numpy()
        d=rng.normal(size=5120)
        for op,dose in [('component',0.),('dose',-1.),('dose',-.5),('dose',.5),('dose',1.)]:
            y,s=make_patch(h,donor,d,0.2,op,dose)
            for j in range(8):
                try:
                    z,t=make_patch(h,donor,d,0.2,op,dose,rng.normal(size=5120))
                except RuntimeError as exc:
                    if 'BF16 realized norm mismatch' not in str(exc):raise
                    rejected+=1
                    continue
                assert abs(np.linalg.norm(z-h)-np.linalg.norm(y-h))/max(np.linalg.norm(y-h),1e-12)<=.00501
                checked+=1
        no,s=make_patch(h,donor,d,.2,'noop',0)
        assert np.array_equal(no,h)
        repl,s=make_patch(h,donor,d,.2,'replace',0)
        assert np.array_equal(repl,donor)
    model=Fake();inp=torch.tensor(np.tile(h,(1,5,1)),dtype=torch.bfloat16)
    hooks=hook_inventory(model)
    patch=PromptPatch(model,h,repl,5,2)
    with patch:
        out=model.model.layers[54](inp)
        assert torch.equal(out[:,:-1],inp[:,:-1])
        assert np.array_equal(out[0,-1].float().numpy(),repl)
        for _ in range(2):
            token=inp[:,-1:];assert model.model.layers[54](token) is token
    assert patch.calls==3 and patch.applied==1 and hook_inventory(model)==hooks
    try:
        with PromptPatch(model,h,repl,5,0):
            model.model.layers[54](inp[:,:3])
    except RuntimeError:pass
    else:raise AssertionError('wrong prompt length accepted')
    assert hook_inventory(model)==hooks
    try:
        with PromptPatch(model,h+1,repl,5,0):model.model.layers[54](inp)
    except RuntimeError:pass
    else:raise AssertionError('wrong cached state accepted')
    assert hook_inventory(model)==hooks
    assert checked>0 and rejected>0
    print({'status':'cpu_engineering_tests_passed','quantized_matches':checked,'infeasible_matches_rejected':rejected,
        'schedule_and_wrong_boundary_fail_closed':True,'cleanup_on_error':True})
if __name__=='__main__':main()

