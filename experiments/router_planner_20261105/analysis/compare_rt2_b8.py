"""Fresh trace replay: native rt2 versus sampled-support single/bundle exchange.

CPU semantic quality only. Existing sampled-support artifacts are read back and
audited; single-pair exchange is rerun with exactly that same support and seed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from moonep_cpu import plan, communication, retention_audit
from replay_strategies import audit
from vendor.bucket_optimizer import optimize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/rt2_b8_comparison'
OUT.mkdir(exist_ok=True)
rows = []
for layer in range(8):
    inp = ROOT / f'outputs/trace_audit/layer_{layer:02d}.npz'
    x = np.load(inp)['topk'].astype(np.int32)
    p0 = plan(x, E=352, B=8, replica_target=0)
    p2 = plan(x, E=352, B=8, replica_target=2)
    row = {'layer': layer, 'input_sha256': hashlib.sha256(inp.read_bytes()).hexdigest(), 'methods': {}}
    for name, p in [('native_rt0', p0), ('native_rt2', p2)]:
        m = communication(x, p.owner, 352)
        invalid = int((p.physical_topk < 0).sum())
        selected = (p.experts_to_copy >= 0).sum(1)
        valid = invalid == 0 and max(m['active_extra_experts']) <= 8
        assert m['pair_loads'] == [x.shape[1]*x.shape[2]]*8
        row['methods'][name] = dict(metrics=m, selected_extra=selected.tolist(), unsupported_pairs=invalid,
                                    audit_pass=valid, retention=retention_audit(x,p))
        np.savez_compressed(OUT/f'layer{layer:02d}_{name}.npz', owner=p.owner,
                            alloc=p.alloc, physical_topk=p.physical_topk, experts_to_copy=p.experts_to_copy)
    for name, artifact in [('sample256_g2', f'layer{layer:02d}_b8_small256_global_4.npz'),
                           ('sample4096_g4', f'layer{layer:02d}_b8_reciprocal_global_4.npz')]:
        path = ROOT/'outputs/replay'/artifact
        saved = np.load(path)
        support = saved['support']
        for mode in ('bundle', 'single'):
            if mode == 'bundle':
                dst = saved['owner']
            else:
                dst, _ = optimize(x, p0.owner, support, rounds=4, mode='single')
                np.savez_compressed(OUT/f'layer{layer:02d}_{name}_single.npz', owner=dst, support=support)
            m, extras = audit(x, p0.owner, dst, support, 8)
            row['methods'][name+'_'+mode] = dict(metrics=m, reserved_extra=extras, audit_pass=True,
                support_artifact=str(path.relative_to(ROOT)), support_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    base = row['methods']['native_rt0']['metrics']['remote_token_destinations']
    for data in row['methods'].values():
        data['reduction_pct_vs_rt0'] = 100*(1-data['metrics']['remote_token_destinations']/base)
    rows.append(row)
    print(layer, {k: (round(v['reduction_pct_vs_rt0'],4), v['audit_pass']) for k,v in row['methods'].items()}, flush=True)
    (OUT/'results.json').write_text(json.dumps({'scope': 'CPU semantic replay; no new GPU timing',
        'budget_per_rank':8, 'rounds':4, 'sampling_seed':20261026, 'rows':rows},indent=2)+'\n')
totals = {}
for name in rows[0]['methods']:
    total = sum(r['methods'][name]['metrics']['remote_token_destinations'] for r in rows)
    baseline = sum(r['methods']['native_rt0']['metrics']['remote_token_destinations'] for r in rows)
    totals[name] = dict(remote=total, reduction_pct=100*(1-total/baseline),
        audit_pass=all(r['methods'][name]['audit_pass'] for r in rows),
        max_active_extra=max(max(r['methods'][name]['metrics']['active_extra_experts']) for r in rows),
        unsupported_pairs=sum(r['methods'][name].get('unsupported_pairs',0) for r in rows))
(OUT/'summary.json').write_text(json.dumps(totals,indent=2)+'\n')
print(json.dumps(totals,indent=2))
