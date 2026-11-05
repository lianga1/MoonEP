from pathlib import Path
import json, numpy as np
root=Path(__file__).resolve().parents[2]
rows=json.loads((root/'outputs/replay/small_sample.json').read_text())['rows']
data=[]
for seed in (20261026,20261027,20261028):
 ids=np.sort(np.random.default_rng(seed).choice(134144,256,replace=False))
 rr=[r for r in rows if r['seed']==seed]
 data.append(dict(seed=seed,ids=ids.tolist(),ranks=np.bincount(ids//16768,minlength=8).tolist(),reduction=100*(1-sum(r['remote'] for r in rr)/sum(r['baseline'] for r in rr))))
summary=json.loads((root/'outputs/rt2_b8_comparison/summary.json').read_text())
text=(Path(__file__).parent/'template.html').read_text().replace('__DATA__',json.dumps(data)).replace('__RESULTS__',json.dumps(summary))
(Path(__file__).parent/'MoonEP_动画讲解.html').write_text(text)
(Path(__file__).parent/'sampling_evidence.json').write_text(json.dumps({'numpy_version':np.__version__,'scope':'exact NumPy sampling reproduction; ids sorted for processing; total=134144, sample=256','seeds':data},indent=2))
print('built',len(text),'characters')
