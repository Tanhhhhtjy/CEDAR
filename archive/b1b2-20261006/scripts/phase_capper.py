"""Gate the historical cap hook by forward call, preserving its BF16 arithmetic."""
from common import enabled
def make_capper(original,trace,phase,telemetry):
    class PhaseCapper(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.calls={i:0 for i in self.caps};self.enabled_calls={i:0 for i in self.caps}
        def _hook(self,layer):
            original_hook=super()._hook(layer)
            def hook(module,inputs,output):
                call=self.calls[layer];self.calls[layer]+=1
                active=enabled(call,len(trace['trace_prefix_token_ids']),phase)
                if active:
                    self.enabled_calls[layer]+=1
                    return original_hook(module,inputs,output)
                return output
            return hook
        def __exit__(self,*exc):
            super().__exit__(*exc)
            telemetry.update(calls=self.calls,enabled_calls=self.enabled_calls,phase=phase)
            if exc[0] is None:
                L=len(trace['trace_prefix_token_ids'])
                assert set(self.calls.values())=={L+1}
                assert set(self.enabled_calls.values())=={L*int(phase[0])+int(phase[1])}
    return PhaseCapper

class BoundaryCheck:
    def __init__(self,trace):self.t=trace;self.calls=0;self.terminal=None
    def __call__(self,module,args,kwargs):
        call=self.calls;self.calls+=1;L=len(self.t['trace_prefix_token_ids'])
        assert call<=L
        if call!=L:return
        ids=kwargs['input_ids'];cache=kwargs.get('past_key_values')
        assert ids.shape==(1,1) and int(ids[0,0])==self.t['trace_prefix_token_ids'][-1]
        length=int(cache.get_seq_length());assert length==self.t['prompt_tokens']+L-1,(length,self.t['prompt_tokens'],L)
        pos=kwargs.get('cache_position');mask=kwargs.get('attention_mask')
        assert pos is not None and int(pos[-1])==length
        assert mask is not None and mask.shape[-1]==length+1
        import hashlib
        self.terminal=dict(call_index=call,input_token=int(ids[0,0]),input_shape=list(ids.shape),past_cache_length=length,cache_position=pos.detach().cpu().tolist(),attention_mask_shape=list(mask.shape),attention_mask_sha256=hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest())
