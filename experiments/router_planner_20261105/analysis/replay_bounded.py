"""Bound residency search and sweep count on real input; offline quality ablation."""
from replay_strategies import *

def main():
    out=ROOT/'outputs/replay/bounded.json';rows=[]
    # Fixed settings across all layers, no best-per-layer selection.
    settings=[('reciprocal',256,2),('reciprocal',1024,2),('visited',0,0)]
    for layer in range(8):
        x=np.load(ROOT/f'outputs/trace_audit/layer_{layer:02d}.npz')['topk'].astype(np.int32)
        p=plan(x,E=352,B=8);base=communication(x,p.owner,352);home=np.arange(352)//44
        for method,sample,groups in settings:
            start=time.perf_counter()
            support,diag=select_support(x,p.owner,home,slots=8,method=method,
                                       sample_tokens=sample,groups_per_direction=groups,seed=20261026)
            prep=time.perf_counter()-start
            for rounds in (2,4):
                dst,stats=optimize(x,p.owner,support,rounds=rounds,mode='bundle')
                m,e=audit(x,p.owner,dst,support,8)
                rows.append(dict(layer=layer,method=method,sample=sample,groups=groups,rounds=rounds,
                                 cpu_support_seconds=prep,baseline=base['remote_token_destinations'],
                                 remote=m['remote_token_destinations'],active_extra=m['active_extra_experts'],
                                 audit_pass=True,diagnostics=diag))
            out.write_text(json.dumps({'scope':'CPU quality/complexity ablation, not GPU timing','rows':rows},indent=2)+'\n')
        print('layer',layer,'done',flush=True)

if __name__=='__main__':main()
