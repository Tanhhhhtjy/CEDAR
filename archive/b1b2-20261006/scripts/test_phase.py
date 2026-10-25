"""Meaningful CPU test: identical arithmetic when enabled, no change when disabled."""
import argparse,sys
from pathlib import Path
from common import enabled
from phase_capper import make_capper,BoundaryCheck
def main(root):
    sys.path.insert(0,str(root/'CEDAR_SUPPORTING_DATA/code/src'))
    import torch
    from torch import nn
    from cedar_m2.interventions.capping import ActivationCapper,LayerCap,apply_upper_cap
    class Toy(nn.Module):
        def __init__(self):
            super().__init__();self.model=nn.Module();self.model.layers=nn.ModuleList([nn.Identity(),nn.Identity()])
        def forward(self,x):
            for layer in self.model.layers:x=layer(x)
            return x
    model=Toy();d=torch.tensor([1.,0.,0.,0.]);caps={i:LayerCap(i,d,0.) for i in [0,1]}
    for L in [1,3,7]:
        for phase in ['00','10','01','11']:
            t=dict(trace_prefix_token_ids=[1]*L);tele={};x=torch.tensor([[[2.,1.,0.,0.]]],dtype=torch.bfloat16)
            with make_capper(ActivationCapper,t,phase,tele)(model,caps,positions='all',defer_stats_sync=True):
                for i in range(L+1):
                    y=model(x);expected=apply_upper_cap(x,d,0.)[0] if enabled(i,L,phase) else x
                    assert torch.equal(y,expected),(L,phase,i)
            assert not any(m._forward_hooks for m in model.modules())
    try:
        with make_capper(ActivationCapper,dict(trace_prefix_token_ids=[1]),'11',{})(model,caps):raise RuntimeError('intentional')
    except RuntimeError:pass
    assert not any(m._forward_hooks for m in model.modules())
    class Cache:
        def get_seq_length(self):return 6
    check=BoundaryCheck(dict(trace_prefix_token_ids=[4,5],prompt_tokens=5))
    check(None,(),{});check(None,(),{});check(None,(),dict(input_ids=torch.tensor([[5]]),past_key_values=Cache(),cache_position=torch.tensor([6]),attention_mask=torch.ones(1,7)))
    assert check.terminal['past_cache_length']==6
    print('CPU gate, arithmetic, exception cleanup and terminal-boundary tests passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
