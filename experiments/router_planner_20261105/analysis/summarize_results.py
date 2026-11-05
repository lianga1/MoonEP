"""Produce compact, reproducible numeric summaries and a scientific figure."""
from pathlib import Path
import json,statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'outputs/replay/strategies.json').read_text())['rows']
summary={'input':{'ranks':8,'tokens_per_rank':16768,'topk':8,'experts':352,'layers':8},'strategies':[],'native':[]}
for b in (8,44):
    rr=[r for r in rows if r['budget']==b]
    base=sum(r['baseline']['remote_token_destinations'] for r in rr)
    for name in [v['name'] for v in rr[0]['variants']]:
        vv=[next(v for v in r['variants'] if v['name']==name) for r in rr]
        summary['strategies'].append(dict(budget=b,name=name,baseline_remote=base,
            final_remote=base-sum(v['remote_saved'] for v in vv),
            reduction_percent=100*sum(v['remote_saved'] for v in vv)/base,
            per_layer_reduction_percent=[v['remote_reduction_percent'] for v in vv],
            mean_active_extra_per_layer=np.mean([sum(v['metrics']['active_extra_experts']) for v in vv]),
            cpu_support_median_ms=1000*statistics.median(v['cpu_support_seconds'] for v in vv)))
for dn in ('gpu_lut_full','gpu_sparse_select'):
    data=json.loads((ROOT/f'outputs/{dn}/summary.json').read_text())
    for b in (8,44):
        rr=[r for r in data['records'] if r['slots']==b]
        variant=data['metadata']['optimized_variant']
        base=statistics.mean(r['median_us']['baseline'] for r in rr)
        changed=statistics.mean(r['median_us'][variant] for r in rr)
        assert all(r['valid_physical_plan'] for r in rr)
        for r in rr:
            for p in r['per_rank']:
                assert all(p['exact_match'].values())
                assert all(p['checks'][x] for x in ['owner_matches_cpu_replay','physical_topk_matches_cpu_replay','experts_to_copy_matches_cpu_replay','global_pair_slots_unique'])
        summary['native'].append(dict(variant=variant,budget=b,mean_layer_baseline_us=base,
            mean_layer_optimized_us=changed,reduction_percent=100*(1-changed/base),all_correct=True))
small=json.loads((ROOT/'outputs/replay/small_sample.json').read_text())['rows']
summary['small_sample']=[]
for seed in sorted({r['seed'] for r in small}):
    rr=[r for r in small if r['seed']==seed]
    summary['small_sample'].append(dict(seed=seed,reduction_percent=100*(1-sum(r['remote'] for r in rr)/sum(r['baseline'] for r in rr)),
        cpu_support_median_ms=statistics.median(r['cpu_support_seconds'] for r in rr)*1000))
summary['bucket_components']=[]
for mode in ('single','local','coordinated'):
    f=ROOT/f'experiments/bucket8/results/{mode}_graph.json'
    if not f.exists():continue
    d=json.loads(f.read_text())
    for method in ('seed','visited','reciprocal'):
        rr=[r for r in d['rows'] if r['case'].endswith('_'+method)]
        if not rr:continue
        assert len(rr)==8 and all(r['audit']['cpu_owner_exact'] for r in rr)
        summary['bucket_components'].append(dict(mode=mode,method=method,
            mean_layer_median_gpu_ms=statistics.mean(r['median_gpu_ms'] for r in rr),
            min_layer_median_gpu_ms=min(r['median_gpu_ms'] for r in rr),
            max_layer_median_gpu_ms=max(r['median_gpu_ms'] for r in rr),
            reduction_percent=100*sum(r['audit']['saved'] for r in rr)/sum(r['audit']['remote_before'] for r in rr)))
(ROOT/'outputs/SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
names=['seed_global_4','visited_global_4','reciprocal_global_4']
labels=['Existing replicas','Simple statistics, B=8','Grouped residency, B=8']
colors=['#4d748a','#d78b30','#4d9169']
for n,l,c in zip(names,labels,colors):
    r=next(r for r in summary['strategies'] if r['budget']==8 and r['name']==n)
    ax[0].plot(range(8),r['per_layer_reduction_percent'],'o-',label=l,color=c)
ax[0].set(xlabel='MoE layer',ylabel='Remote token payload reduction (%)',title='Real router replay: four exchange rounds')
ax[0].legend(fontsize=8,loc='lower right');ax[0].grid(alpha=.2)
if summary['bucket_components']:
    data=[r for r in summary['bucket_components'] if r['method']=='reciprocal']
    xs=['One GPU\nglobal','Eight GPUs\nlocal only','Eight GPUs\nglobal counts']
    values=[next(r for r in data if r['mode']==m)['mean_layer_median_gpu_ms']*1000 for m in ('single','local','coordinated')]
    bars=ax[1].bar(xs,values,color=['#4d748a','#d78b30','#4d9169'])
    ax[1].bar_label(bars,fmt='%.1f us',padding=3)
    ax[1].set(ylabel='Four-round component latency (us)',title='CUDA Graph; fixed seed and residency')
    ax[1].set_ylim(0,max(values)*1.2)
else:
    vals=[r['cpu_support_median_ms'] for r in summary['strategies'] if r['budget']==8 and r['name'] in ('visited_global_4','reciprocal_global_4')]
    vals.append(summary['small_sample'][0]['cpu_support_median_ms'])
    bars=ax[1].bar(['Simple\nstatistics','4096 samples\ntop-4 groups','256 samples\ntop-2 groups'],vals,color=colors)
    ax[1].set_yscale('log');ax[1].set(ylabel='CPU residency construction (ms)',title='Candidate construction remains offline')
fig.suptitle('MoonEP supplied trace | 8 ranks x 16,768 rows | top-8 | E=352',fontsize=12)
fig.savefig(ROOT/'outputs/replay_comparison.png',dpi=180)
fig.savefig(ROOT/'outputs/replay_comparison.svg')
print(json.dumps(summary,indent=2))
