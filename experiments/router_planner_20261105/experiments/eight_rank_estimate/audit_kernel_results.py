"""Independent retained-sample readback; CPU only, no GPU invocation."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/eight_rank_estimate'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def med(t,metric='cuda_ms'):return statistics.median(s[metric] for s in t['samples'])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=OUT)
    result_dir=parser.parse_args().output_dir
    result_dir.mkdir(parents=True,exist_ok=True)
    paths=[OUT/f'kernel_graph_attempt01_physical{i}.json' for i in range(2)]
    raw=[json.loads(p.read_text()) for p in paths]
    errors=[];timings=[];sources=[];inputs=[];correctness=[]
    def require(ok,message):
        if not ok:errors.append(message)
    executed={f'experiments/eight_rank_estimate/{f}' for f in
              ('benchmark_kernels.py','exchange_prefix.py','exchange_prefix.cu','residency8.py','residency8.cu')}
    executed.update(f'experiments/h100_planner/{f}' for f in
                    ('bundle_lane8.py','bundle_lane8.cu','baseline_virtual.py','baseline_virtual.cu'))
    def walk(value,where):
        if isinstance(value,dict):
            if {'samples','median_cuda_ms','median_host_ms'}.issubset(value):
                ss=value['samples'];require(len(ss)==10,where+': sample count')
                require([x['sample'] for x in ss]==list(range(10)),where+': missing/duplicate sample IDs')
                require(value['execution_mode']=='graph' and value['replays_or_calls_per_sample']==20,where+': timing protocol')
                row=dict(path=where,sample_count=len(ss),replays_per_sample=value['replays_or_calls_per_sample'])
                for metric in ('cuda_ms','host_ms'):
                    vv=[x[metric] for x in ss]
                    require(all(math.isfinite(x) and x>0 for x in vv),where+': nonpositive/nonfinite '+metric)
                    recomputed=statistics.median(vv);delta=abs(recomputed-value['median_'+metric])
                    require(delta<=1e-12,where+': stored median mismatch '+metric)
                    row[metric]=dict(recomputed_median=recomputed,stored_median=value['median_'+metric],
                                     absolute_difference=delta,min=min(vv),max=max(vv))
                timings.append(row)
            for k,v in value.items():walk(v,where+'/'+str(k))
        elif isinstance(value,list):
            for i,v in enumerate(value):walk(v,where+'/'+str(i))
    fixed=dict(logical_ranks=8,global_tokens=134144,tokens_per_source=16768,K=8,E=352,B=8,d=256,g=2,L=4,sample_seed=20261026)
    require(raw[0]['metadata']['uuid']!=raw[1]['metadata']['uuid'],'same physical GPU UUID')
    require(raw[0]['metadata']['source_files']==raw[1]['metadata']['source_files'],'two physical runs used different source manifests')
    manifest=json.loads((OUT/'kernel_inputs/manifest.json').read_text())
    manifests={r['layer']:r for r in manifest['layers']}
    for device,d in enumerate(raw):
        require(d['status']=='complete' and d['failures']==[],f'physical{device}: incomplete/failed')
        require(d['metadata']['physical_process_rank']==device,f'physical{device}: wrong rank')
        for k,v in fixed.items():require(d['metadata'].get(k)==v,f'physical{device}: metadata {k}')
        require([r['layer'] for r in d['layers']]==list(range(8)),f'physical{device}: layer coverage')
        for name,recorded in d['metadata']['source_files'].items():
            p=ROOT/name;exists=p.exists();actual=sha(p) if exists else None
            appledouble=p.name.startswith('._');critical=name in executed
            same=exists and recorded==actual
            if critical:require(same,f'physical{device}: executed source mismatch {name}')
            sources.append(dict(physical=device,path=name,recorded_sha256=recorded,current_sha256=actual,
                exists=exists,exact=same,execution_dependency=critical,appledouble_transfer_metadata=appledouble,
                interpretation=('传输附带的macOS元数据，不是Python/CUDA程序' if appledouble else
                                '执行依赖' if critical else '记录时一并快照的非执行辅助脚本')))
        require(executed.issubset(d['metadata']['source_files']),f'physical{device}: missing executed source hashes')
        for row in d['layers']:
            layer=row['layer'];p=OUT/f'kernel_inputs/layer_{layer:02}.npz';actual=sha(p)
            require(actual==row['input_sha256']==manifests[layer]['sha256'],f'physical{device} layer{layer}: input sha')
            inputs.append(dict(physical=device,layer=layer,path=str(p.relative_to(ROOT)),sha256=actual,
                               raw_and_preparation_manifest_exact=actual==row['input_sha256']==manifests[layer]['sha256']))
            require(row['shape']==[8,16768,8],f'physical{device} layer{layer}: folded/wrong shape')
            require(all(v is True for v in row['correctness'].values()) and row['post_timing_correctness'] is True,
                    f'physical{device} layer{layer}: correctness status')
            require([s['source'] for s in row['source_components']]==list(range(8)),f'physical{device} layer{layer}: source coverage')
            for sr in row['source_components']:
                require(sr['tokens_scanned']==16768 and sr['possible_execution_ranks']==8,f'physical{device} layer{layer} source{sr["source"]}: scan shape')
                require([r['round'] for r in sr['rounds']]==list(range(4)),f'physical{device} layer{layer} source{sr["source"]}: round coverage')
                correctness.append(dict(physical=device,layer=layer,source=sr['source'],rounds=[0,1,2,3],
                    original_tokens=16768,layer_all32_owner_and_count_audit_pass=True,post_timing_layer_audit_pass=True,
                    evidence='hash-verified harness asserts source owner and counts in every round; raw JSON retains layer-level pass flags'))
        walk(d,f'physical{device}')
    # Independently re-read the original source hashes and check every fixture
    # owner's support, load and byte-count invariants; no CUDA owner tensors are
    # persisted, so do not pretend this re-executes GPU-output equality checks.
    fixture_checks=[]
    for layer,m in manifests.items():
        originals=[]
        for source,expected in m['original_sources'].items():
            p=Path(source)
            if p.is_absolute() and "/outputs/" in p.as_posix():
                p=ROOT/"outputs"/p.as_posix().split("/outputs/",1)[1]
            elif not p.is_absolute():
                p=ROOT/p
            ok=p.exists() and sha(p)==expected
            require(ok,f'layer{layer}: original artifact hash {p}')
            originals.append(dict(path=str(p),exact=ok))
        z=np.load(OUT/f'kernel_inputs/layer_{layer:02}.npz',allow_pickle=False)
        x=z['topk'];support=z['support'];owners=z['owner_prefixes'];remotes=[]
        require(owners.shape==(5,8,16768,8),f'layer{layer}: CPU prefix shape')
        for step,o in enumerate(owners):
            require(bool(((o>=0)&(o<8)).all()) and bool(support[o,x].all()),f'layer{layer} step{step}: illegal owner')
            require(np.array_equal(np.bincount(o.ravel(),minlength=8),np.full(8,16768*8)),f'layer{layer} step{step}: load')
            sorted_owner=np.sort(o,axis=-1)
            first=np.concatenate((np.ones((8,16768,1),bool),sorted_owner[:,:,1:]!=sorted_owner[:,:,:-1]),axis=-1)
            remotes.append(int((first&(sorted_owner!=np.arange(8)[:,None,None])).sum()))
        require(np.array_equal(np.array(remotes[:-1])-remotes[1:],z['stats'][:,3]),f'layer{layer}: gain vs deduplicated remote')
        require(np.array_equal(z['sample_ids'],np.sort(np.random.default_rng(20261026).choice(134144,256,replace=False))),f'layer{layer}: sample seed')
        fixture_checks.append(dict(layer=layer,originals=originals,all_five_owner_states_physical_and_balanced=True,
            remote_destinations=remotes,round_gains=z['stats'][:,3].tolist(),exact_sampling_seed=True))
    layers=[]
    for layer in range(8):
        globals_us=[med(d['layers'][layer]['global_components']['residency']['timing'])*1000 for d in raw]
        ss=[]
        for source in range(8):
            entry=dict(source=source)
            for mode in ('sort_L4_conditional','prefix_L4_conditional'):
                vv=[med(d['layers'][layer]['source_components'][source][mode])*1000 for d in raw]
                entry[mode]=dict(per_physical_median_us=vv,max_of_two_physical_medians_us=max(vv))
            entry['reduction_pct']=100*(1-entry['prefix_L4_conditional']['max_of_two_physical_medians_us']/entry['sort_L4_conditional']['max_of_two_physical_medians_us'])
            ss.append(entry)
        slow=max(s['sort_L4_conditional']['max_of_two_physical_medians_us'] for s in ss)
        fast=max(s['prefix_L4_conditional']['max_of_two_physical_medians_us'] for s in ss)
        layers.append(dict(layer=layer,global_residency_per_physical_median_us=globals_us,
            global_residency_max_physical_median_us=max(globals_us),
            old_sort_max_source_and_physical_median_us=slow,prefix_max_source_and_physical_median_us=fast,
            prefix_reduction_pct=100*(1-fast/slow),sources=ss))
    avg_res=statistics.mean(r['global_residency_max_physical_median_us'] for r in layers)
    avg_old=statistics.mean(r['old_sort_max_source_and_physical_median_us'] for r in layers)
    avg_new=statistics.mean(r['prefix_max_source_and_physical_median_us'] for r in layers)
    summary=dict(status='passed' if not errors else 'failed',scope_zh='两张物理H200独立测原始八逻辑rank维度；每来源实际扫描16768个token，未折叠路由、未将全量时间除以8。',
        aggregation_zh='每个配置先取10个原始样本的中位数；来源计算再取8来源及2张物理卡的最大中位数；最后对8层取算术平均。两物理卡是独立微基准副本，不是八卡同步采样。',
        units='microseconds',layers=layers,
        eight_layer_mean=dict(global_residency_us=avg_res,old_sort_source_L4_us=avg_old,prefix_source_L4_us=avg_new,
                              prefix_reduction_pct=100*(1-avg_new/avg_old),prefix_speedup=avg_old/avg_new),
        measured_scope=dict(global_residency='完整R8驻留选择的GPU计算；所有来源数据已在同一GPU，含runtime样本索引读取，不含跨卡收集和发布',
            source_L4='固定四轮global count表已可用条件下：本来源cache准备、seed恢复、四轮提议/桶计数、本GPU全局计数折叠及配额、提交。',
            old_sort='已有R8 CUB排序交换实现；不是原生MoonEP planner',prefix='R8缓存与稳定桶前缀交换端口'),
        not_measured=['原生MoonEP八卡planner时间','真实八卡样本/计数集合通信及等待','expert plan跨卡发布','最终地址重建及发布','八卡端到端训练时间'],
        total_planner_time_measured=False,global_positive_benefit_guaranteed=False,
        raw_results=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths])
    exceptions=[r for r in sources if not r['exact']]
    audit=dict(status='passed_with_provenance_notes' if not errors and exceptions else 'passed' if not errors else 'failed',
        errors=errors,raw_results=summary['raw_results'],raw_failures=[d['failures'] for d in raw],
        coverage=dict(physical_devices=2,layers_per_device=8,sources_per_layer=8,source_layer_records=len(correctness),
            source_round_correctness_coverage=len(correctness)*4,timing_blocks=len(timings),raw_samples=sum(t['sample_count'] for t in timings),
            recomputed_medians=len(timings)*2),
        correctness_records=correctness,fixture_checks=fixture_checks,input_hash_checks=inputs,
        source_hash_checks=sources,source_hash_exceptions=exceptions,
        executed_source_hashes_all_exact=not any(r['execution_dependency'] and not r['exact'] for r in sources),
        medians_recomputed_from_raw_samples=True,timing_readback=timings,
        limitations_zh=['GPU owner张量未保存在原始JSON；离线审计核对已保存的逐层校验标志及其源码覆盖范围，未重新执行GPU。',
            '._文件是传输附带的macOS元数据，本地缺失；build_estimate.py是测后可更新的非执行辅助脚本，哈希差异单列保留，不冒充全部源码完全一致。',
            '样本全部保留，无删除、重试筛选或将跨设备独立样本当作八卡同步样本。'])
    for name,value in [('kernel_independent_readback.json',audit),('kernel_summary.json',summary)]:
        with (result_dir/name).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=audit['status'],errors=errors,coverage=audit['coverage'],means=summary['eight_layer_mean']),ensure_ascii=False))

if __name__=='__main__':main()
