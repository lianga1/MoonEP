"""Fixed 256-token, top-2-group residency ablation across three sampling seeds."""
from replay_strategies import *

rows=[]
for layer in range(8):
    x=np.load(ROOT/f'outputs/trace_audit/layer_{layer:02d}.npz')['topk'].astype(np.int32)
    p=plan(x,E=352,B=8);base=communication(x,p.owner,352);home=np.arange(352)//44
    for seed in (20261026,20261027,20261028):
        start=time.perf_counter()
        support,diag=select_support(x,p.owner,home,slots=8,method='reciprocal',sample_tokens=256,groups_per_direction=2,seed=seed)
        prep=time.perf_counter()-start
        dst,stats=optimize(x,p.owner,support,rounds=4,mode='bundle')
        m,e=audit(x,p.owner,dst,support,8)
        rows.append(dict(layer=layer,seed=seed,baseline=base['remote_token_destinations'],remote=m['remote_token_destinations'],
                         cpu_support_seconds=prep,audit_pass=True,reserved_extras=e,active_extras=m['active_extra_experts']))
        if seed==20261026:
            np.savez_compressed(ROOT/f'outputs/replay/layer{layer:02d}_b8_small256_global_4.npz',owner=dst,support=support)
    (ROOT/'outputs/replay/small_sample.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
    print('layer',layer,'done',flush=True)
