"""Render a standalone Chinese report from paired two-rank raw measurements.

Every displayed timing/byte value is derived from supplied artifacts. The
analysis must name the exact raw SHA256. Existing reports are not modified.
Optional subsequent measurement sections have a reserved insertion marker.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics as stats
from urllib.parse import quote


MODES={
    'native_cold_serial':'MoonEP：冷启动，先 token 后权重',
    'native_cold_early':'MoonEP：冷启动，提前传权重',
    'optimized_cold_serial':'当前算法：冷启动，先 token 后权重',
    'optimized_cold_early':'当前算法：冷启动，提前传权重',
    'native_hot':'MoonEP：权重已就绪',
    'optimized_hot':'当前算法：权重已就绪',
}
COMPONENTS={
    'native_planner':('MoonEP 原生 planner','生成均衡分配与执行元数据'),
    'sampled_residency':('GPU 选择专家副本','解码原生执行位置、刷新样本、汇集两卡样本、统计候选并选择副本'),
    'exchange_sort':('交换：原排序路径','按提议类别排序，得到稳定顺序和匹配名额'),
    'exchange_prefix':('交换：复用小表和分块计数','复用可执行任务集合，用各块计数的前缀和替代排序'),
    'materialize':('生成最终执行地址','把新执行位置转成 MoonEP 需要的物理槽、地址与去重信息'),
    'materialize_sort':('执行地址：重新排序','按最终专家位置重新排序，再生成地址'),
    'materialize_reuse':('执行地址：复用原有顺序','复用原生 planner 的排序和顺序信息生成地址'),
    'optimized_full_planner':('当前算法：完整 planner','起始 MoonEP 计划、选副本、全量交换与最终地址'),
    'native_weights':('MoonEP：单独传权重','按原生计划真实复制权重'),
    'optimized_weights':('当前算法：单独传权重','按优化计划真实复制权重'),
}


def esc(value):
    return html.escape(str(value))


def signed(value,places=3):
    return f'{value:+.{places}f}'


def rows_table(headers,rows):
    return ('<div class="table"><table><thead><tr>'+''.join(f'<th>{esc(x)}</th>' for x in headers)+
            '</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+
            '</tbody></table></div>')


def load_cohort(raw_path,analysis_path):
    raw_bytes=raw_path.read_bytes();raw=json.loads(raw_bytes)
    analysis=json.loads(analysis_path.read_text())
    digest=hashlib.sha256(raw_bytes).hexdigest()
    if analysis['source_sha256']!=digest:
        raise ValueError('analysis refers to a different raw measurement file')
    if raw['status']!='complete' or raw.get('failures'):
        raise ValueError('main report requires a complete cohort; retain failures as separate evidence')
    meta=raw['metadata'];layers=sorted(raw['layers'],key=lambda r:r['layer'])
    if len({r['layer'] for r in layers})!=len(layers):
        raise ValueError('duplicate layer IDs')
    if meta['world']!=2:
        raise ValueError('this report requires two physical ranks')
    for row in layers:
        if not all(row['correctness'].values()) or not row.get('post_timing_checked'):
            raise ValueError('correctness audit incomplete')
        for mode,samples in row['measurements'].items():
            if len(samples)!=meta['samples']:
                raise ValueError(f'incomplete {mode} cohort in layer {row["layer"]}')
            for sample in samples:
                if len(sample['per_rank'])!=2:
                    raise ValueError('missing rank')
                for metric in ('cuda_ms','host_ms'):
                    values=[r[metric] for r in sample['per_rank']]
                    if any(not math.isfinite(v) or v<0 for v in values):
                        raise ValueError('invalid duration')
                    stored=sample.get('max_'+metric)
                    if stored is not None and abs(stored-max(values))>1e-9:
                        raise ValueError('stored rank maximum inconsistent')
    return raw,analysis,digest


def render_timeline(layers,source_path,link):
    rows=[];detail=[]
    for row in layers:
        for d in row.get('timeline_diagnostic',[]):
            if not isinstance(d,dict):continue
            rank=d.get('rank','?');layer=row['layer'];events=d.get('gpu_event_ms',{})
            diagnostic_mode=d.get('diagnostic_mode',d.get('execution_mode','eager'))
            mode_label='CUDA Graph 重放' if diagnostic_mode=='graph' else '逐次发起（eager）'
            prof=d.get('profiler',{});overlap=prof.get('measured_kernel_overlap_ms')
            classified=prof.get('kernel_classification_complete',False)
            kernel_text=f'{overlap:.4f}' if classified and isinstance(overlap,(float,int)) else '未取得可分类的 kernel 证据'
            phase_overlap=d.get('weight_token_stream_interval_overlap_ms')
            lead=d.get('expert_copy_lead_over_token_plan_ms')
            tail=d.get('expert_copy_exposed_after_dispatch_ms')
            fmt=lambda x:f'{x:.4f}' if isinstance(x,(int,float)) else '未记录'
            trace=Path(d.get('trace_path',''))
            local=source_path.parent/trace.name
            trace_html=link(local,'原始时间线') if trace.name and local.is_file() else '原始时间线尚未导回'
            rows.append([f'L{layer:02d} / 卡{rank}',
                         '当前' if d.get('optimized') else 'MoonEP',
                         '提前' if d.get('early') else '串行',mode_label,fmt(lead),fmt(phase_overlap),kernel_text,fmt(tail),trace_html])
            names=[]
            for key,label in [('prefetch_kernel_names','专家权重程序'),('dispatch_kernel_names','token 发送程序')]:
                names.append(f'<p>{label}：<code>{esc("; ".join(prof.get(key,[])) or "未识别")}</code></p>')
            bands=[];end=max(events.values(),default=0)
            if end>0:
                for label,start,stop,color in [('专家权重队列','weight_stream_start','weight_stream_end','#cb7539'),
                                               ('token 发送队列','token_stream_start','token_stream_end','#2c82a5')]:
                    if start in events and stop in events:
                        a,b=events[start],events[stop]
                        bands.append(f'<div>{label}：{a:.4f}—{b:.4f} ms</div><div style="position:relative;height:18px;background:#edf2f6;border-radius:4px"><div style="position:absolute;left:{100*a/end:.4f}%;width:{100*max(0,b-a)/end:.4f}%;height:18px;background:{color};border-radius:4px"></div></div>')
            detail.append(f'<details><summary>L{layer:02d} / 卡{rank} 的事件范围与 kernel 分类</summary>'+
                          '<p>下图画的是队列事件之间的时间段，内部可能含等待和空隙；不是 kernel 实际占用的图。</p>'+''.join(bands)+''.join(names)+
                          '<pre class="formula">'+esc(json.dumps(events,ensure_ascii=False,indent=2))+'</pre></details>')
    if not rows:
        return '<p>本轮未记录提前传输的时间线诊断；仅凭完整用时不能认定两类传输已经同时执行。</p>'
    return ('<p>这部分是额外插入观察工具的冷调用，未混入正式延迟样本。事件节点和观察工具可能改变调度，绝对用时不能替代主表。提前量为正，表示专家传输队列在 token 计划完成前已经开始；它不等于提前量全部被隐藏。</p>'+
            rows_table(['层／卡','算法','方式','诊断发起方式','权重早于 token 计划 / ms','两条队列时间段相交 / ms','实际 kernel 执行相交 / ms','token 发完后权重剩余 / ms','证据'],rows)+
            '<p>“队列时间段相交”来自 CUDA 事件，两端之间包含发起间隙、等待和同步；只能说明两段活动的时间范围重叠。“实际 kernel 执行相交”来自性能记录中专家预取与 token 发送程序的持续时间交集，证据更强，但仍不能证明 NVLink 在每个时刻同时传两类字节。程序按名字分类，排除发送收尾程序，具体名字保留如下。</p>'+''.join(detail))


def render_component_cohorts(source_path,layers,link):
    """Show exact independent component cohorts without folding them into main timings."""
    expected={r['input_sha256']:r['layer'] for r in layers}
    labels={'base':'原实现','tops':'各类别并行选前两组','warp':'32线程协作选择',
            'combined':'并行前两组＋协作选择','fused':'统计次数与前两组筛选合并',
            'fused_warp':'合并统计筛选＋协作选择'}
    sections=[]
    def truth_tree(x):
        if isinstance(x,dict):return bool(x) and all(truth_tree(v) for v in x.values())
        if isinstance(x,list):return bool(x) and all(truth_tree(v) for v in x)
        return x is True
    def checked_median(samples,count):
        if len(samples)!=count:raise ValueError('incomplete independent component cohort')
        values=[]
        for s in samples:
            if len(s['per_rank'])!=2:raise ValueError('missing component rank')
            value=max(r['cuda_ms'] for r in s['per_rank'])
            if not math.isfinite(value) or value<0:raise ValueError('invalid component timing')
            if 'max_cuda_ms' in s and abs(value-s['max_cuda_ms'])>1e-9:raise ValueError('component rank maximum differs')
            values.append(value)
        return stats.median(values)
    paths=sorted(source_path.parent.glob('residency_variants*.json'))+sorted(source_path.parent.glob('residency_fused*.json'))
    for path in paths:
        data=json.loads(path.read_text())
        if data.get('status')!='complete' or data.get('failures'):
            sections.append('<p>独立副本选择测试未完整通过，保留记录：'+link(path,path.name)+'</p>');continue
        meta=data['metadata'];rows=[];input_matches=[];all_audits=True
        for x in data.get('real_layers',[]):
            matched=x.get('input_sha256') in expected;input_matches.append(matched)
            checked=(len(x.get('per_rank_audits',[]))==2 and truth_tree(x['per_rank_audits']) and x.get('post_graph_exact') is True)
            all_audits=all_audits and checked
            values={key:checked_median(samples,meta['samples']) for key,samples in x.get('timings',{}).items()}
            for variant in meta.get('variants',[]):
                select=values.get(variant+'/select');gather=values.get(variant+'/gather_select')
                if select is None or gather is None:continue
                base=values['base/gather_select']
                rows.append([esc(x['name']),esc(labels.get(variant,variant)),f'{select:.4f}',f'{gather:.4f}',
                             f'{100*(1-gather/base):+.1f}%', '一致' if matched else '未匹配本轮输入',
                             '逐项相同' if checked else '未通过'])
        synthetic=data.get('synthetic_checks',[])
        synthetic_pass=sum(len(x.get('per_rank_audits',[]))==2 and truth_tree(x['per_rank_audits']) for x in synthetic)
        sections.append('<h3>'+esc(path.stem)+'</h3><p>'+link(path,'独立原始记录')+
            f'。每项 {meta["samples"]} 次重复；先取两卡较慢时间再取中位数。真实输入指纹全部匹配本轮：{bool(input_matches) and all(input_matches)}；'+
            f'真实输入的完整中间结果与输出核验全部通过：{all_audits}。另有 {synthetic_pass}/{len(synthetic)} 个合成边界输入通过，这些输入只用于正确性检查。</p>'+
            rows_table(['输入','实现','仅选择 / ms','刷新样本＋汇集＋选择 / ms','相对本批原实现下降','输入指纹','结果核验'],rows))
    for path in sorted(source_path.parent.glob('materializer_components*.json')):
        data=json.loads(path.read_text())
        if data.get('status')!='complete' or data.get('failures'):
            sections.append('<p>独立地址生成测试未完整通过，保留记录：'+link(path,path.name)+'</p>');continue
        rows=[]
        for x in data.get('layers',[]):
            measurements=x.get('measurements',{})
            if not all(k in measurements for k in ('materialize_sort','materialize_reuse')):continue
            old=checked_median(measurements['materialize_sort'],data['metadata']['samples'])
            new=checked_median(measurements['materialize_reuse'],data['metadata']['samples'])
            checked=all(x.get('correctness',{}).values()) and x.get('correctness',{}).get('materializers_exact') is True and x.get('post_timing_checked') is True
            rows.append([f'L{x["layer"]:02d}',f'{old:.4f}',f'{new:.4f}',f'{100*(1-new/old):.1f}%',
                         '一致' if x.get('input_sha256') in expected else '未匹配本轮输入',
                         '逐项相同' if checked else '未通过'])
        sections.append('<h3>复用原生顺序生成地址</h3><p>'+link(path,'独立地址生成记录')+
                        '。执行位置在组件计时前已准备；下表不包含完整 planner 和数据传输。</p>'+
                        rows_table(['层','重新排序 / ms','复用原顺序 / ms','用时下降','输入指纹','地址与数据来源核验'],rows))
    if not sections:return ''
    return ('<h3>独立组件实验：定位改进来自哪里</h3><p>这些是分别运行的测量批次，适合比较同一批中的原实现和候选。它们不替代主表的完整路径，也不能把各批节省相加成整段收益。</p>'+
            '<p>“各类别并行选前两组”把不同类别交给不同线程组；“32线程协作选择”让一个固定线程小组协作比较候选；“合并统计筛选”把组出现次数和前两名筛选放在同一次 GPU 程序里，并在组内复用样本。所有这些版本要求保留原始选择结果；最终执行方案是否等价由表中的逐项核验给出。</p>'+''.join(sections))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True,help='current formal raw cohort')
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--comparison',nargs=2,type=Path,metavar=('PREVIOUS_RAW','PREVIOUS_ANALYSIS'),
                   help='retain one preceding iteration in a separate comparison section')
    args=p.parse_args()
    raw,analysis,digest=load_cohort(args.input,args.analysis)
    meta=raw['metadata'];layers=sorted(raw['layers'],key=lambda r:r['layer'])
    outdir=args.output.resolve().parent
    def link(path,label):
        path=Path(path).resolve()
        return f'<a href="{quote(os.path.relpath(path,outdir),safe="/._-")}">{esc(label)}</a>'
    def median(row,mode,metric='cuda_ms'):
        return stats.median(max(s['per_rank'][r][metric] for r in range(2))
                            for s in row['measurements'][mode])
    def average(mode,metric='cuda_ms'):
        return stats.mean(median(row,mode,metric) for row in layers)
    def comparison_layer(layer):
        return next(x for x in analysis['analysis']['cuda_ms']['layers'] if x['layer']==layer)
    comparison_html='';retention_html=''
    if args.comparison:
        previous,previous_analysis,previous_hash=load_cohort(*args.comparison)
        previous_layers=sorted(previous['layers'],key=lambda r:r['layer'])
        def previous_average(mode):
            return stats.mean(median(row,mode) for row in previous_layers)
        changes=[]
        for key in ('world','gpu','torch','cuda','nccl','hidden','intermediate','rounds','num_sms',
                    'prefetch_sms','timing_mode','samples','source_mapping','replica_target','replica_max',
                    'materializer','residency'):
            old=previous['metadata'].get(key);new=meta.get(key)
            if old!=new:changes.append(f'{key}: {old!r} → {new!r}')
        same_inputs=({r['layer']:r['input_sha256'] for r in previous_layers}==
                     {r['layer']:r['input_sha256'] for r in layers})
        if meta.get('rounds')==1 and previous['metadata'].get('rounds')==4 and same_inputs:
            old_by_layer={r['layer']:r for r in previous_layers};retention_rows=[];one_total=four_total=0
            same_experts=True
            for row in layers:
                previous_row=old_by_layer[row['layer']];c=row['communication'];oldc=previous_row['communication']
                same_experts=same_experts and all(c[k]==oldc[k] for k in ('remote_before_by_rank','native_copies','optimized_copies'))
                one=sum(c['remote_before_by_rank'])-sum(c['remote_after_by_rank'])
                four=sum(oldc['remote_before_by_rank'])-sum(oldc['remote_after_by_rank'])
                one_total+=one;four_total+=four
                retention_rows.append([f'L{row["layer"]:02d}',f'{one:,}',f'{four:,}',f'{100*one/four:.2f}%' if four else '无节省可比'])
            if same_experts and four_total>0:
                retention_html=('<h3>交换做一轮还是四轮</h3><p>这里 L 表示全量交换的轮数。两轮实验使用相同路由输入和起始远端份数，记录的原生／优化专家副本数量也一致。'+
                    f'本轮 L=1 共减少 {one_total:,} 份远端 token；前一轮 L=4 减少 {four_total:,} 份。通信节省保留比例为 <strong>{one_total:,} ÷ {four_total:,} = {100*one_total/four_total:.4f}%</strong>。这不是总通信下降百分比，而是一轮取得了四轮节省量的多少。</p>'+
                    rows_table(['层','L=1 少发份数','L=4 少发份数','节省保留比例'],retention_rows)+
                    '<p>只做一轮降低了扫描成本，也放弃了后面几轮才出现的少量机会。两个版本的完整时间来自不同测量批次，时间差还包含副本选择等实现变化；不能全部归因于轮数。</p>')
            else:
                retention_html='<p>L=1 与 L=4 的输入虽相同，但起始通信或副本数量不同，本报告不把节省比例全部归因于轮数。</p>'
        compare_rows=[]
        for mode in list(COMPONENTS)+list(MODES):
            if not all(mode in x['measurements'] for x in previous_layers+layers):continue
            old=previous_average(mode);new=average(mode);delta=old-new
            label=COMPONENTS[mode][0] if mode in COMPONENTS else MODES[mode]
            compare_rows.append([esc(label),f'{old:.4f}',f'{new:.4f}',
                                 f'<span class="{"good" if delta>0 else "bad"}">{delta:+.4f}</span>'])
        comparison_html=('<h3>上一轮保留记录与本轮对照</h3><p>上一轮：'+link(args.comparison[0],args.comparison[0].name)+' · '+
            link(args.comparison[1],'上一轮收益分析')+'。当前主表和启用结论始终使用本轮记录。两次测量是不同批次，下面的差值用于观察迭代变化，不作为跨批次配对显著性结论。</p>'+
            f'<p>输入指纹逐层一致：{same_inputs}。上一轮冷／热收益判定：'+
            ('通过' if previous_analysis['cold_measured_gate_pass'] else '不通过')+'／'+
            ('通过' if previous_analysis['hot_measured_gate_pass'] else '不通过')+'。</p>'+
            rows_table(['组件或路径','上一轮 / ms','本轮 / ms','上一轮减本轮 / ms'],compare_rows)+
            '<p class="muted">记录的协议字段差异：'+esc('; '.join(changes) if changes else '上述字段没有差异')+
            '。缺失字段表示旧记录未写入，不能自动解释为配置变化。上一轮原始指纹：<code>'+previous_hash+'</code>。</p>')
    timeline_html=render_timeline(layers,args.input,link)
    cohort_html=render_component_cohorts(args.input,layers,link)
    cold_native=min(('native_cold_serial','native_cold_early'),key=average)
    cold_opt=min(('optimized_cold_serial','optimized_cold_early'),key=average)
    cold_base=average(cold_native);cold_candidate=average(cold_opt)
    hot_base=average('native_hot');hot_candidate=average('optimized_hot')
    prefix=average('exchange_prefix');sort=average('exchange_sort')
    prefix_saved=100*(1-prefix/sort)
    before=sum(sum(r['communication']['remote_before_by_rank']) for r in layers)
    after=sum(sum(r['communication']['remote_after_by_rank']) for r in layers)
    reduction=100*(1-after/before)
    byte_rows={r['layer']:r for r in analysis['bytes']}
    saved_bytes=sum(byte_rows[r['layer']]['saved_dispatch_combine_payload_bytes'] for r in layers)
    extra_bytes=sum(byte_rows[r['layer']]['incremental_expert_bytes'] for r in layers)
    MiB=1<<20;GiB=1<<30
    cold_pass=analysis['cold_measured_gate_pass'];hot_pass=analysis['hot_measured_gate_pass']
    no_win=not cold_pass and not hot_pass
    title='两卡实测未取得整体正收益，默认不启用动态副本' if no_win else '两卡 planner 与通信综合测量'
    decision='默认禁用' if not analysis['default_enable_dynamic_replication'] else '满足本次测量的启用条件'
    def saving_html(delta,places=3):
        return f'<span class="{"good" if delta>0 else "bad"}">{signed(delta,places)}</span>'
    cold_change=100*(cold_candidate/cold_base-1);hot_change=100*(hot_candidate/hot_base-1)
    headline=('本轮完整规划／通信段未通过正收益判定，动态副本默认禁用。'
              if no_win else '本轮存在通过测量判定的使用场景；冷热状态、逐层数据与适用范围分别列出。')
    mode_rows=[]
    for mode,label in MODES.items():
        reference=hot_base if 'hot' in mode else cold_base
        value=average(mode)
        change=(value/reference-1)*100
        mode_rows.append([esc(label),f'{value:.3f}',f'{average(mode,"host_ms"):.3f}',
            f'<span class="{"bad" if change>0 else "good"}">{signed(change,2)}%</span>'])
    component_rows=[]
    for mode,(label,meaning) in COMPONENTS.items():
        if mode in layers[0]['measurements']:
            component_rows.append([esc(label),f'{average(mode):.4f}',esc(meaning)])
    per_layer=[];traffic=[];reuse=[];confidence=[]
    for row in layers:
        l=row['layer'];b=byte_rows[l]
        nb=min(median(row,'native_cold_serial'),median(row,'native_cold_early'))
        ob=min(median(row,'optimized_cold_serial'),median(row,'optimized_cold_early'))
        nh=median(row,'native_hot');oh=median(row,'optimized_hot')
        per_layer.append([f'L{l:02d}',f'{nb:.3f}',f'{median(row,"optimized_cold_serial"):.3f}',
            f'{median(row,"optimized_cold_early"):.3f}',saving_html(nb-ob),
            f'{nh:.3f}',f'{oh:.3f}',saving_html(nh-oh)])
        cb=sum(row['communication']['remote_before_by_rank']);co=sum(row['communication']['remote_after_by_rank'])
        traffic.append([f'L{l:02d}',f'{cb:,} → {co:,}',f'{100*(1-co/cb):.2f}%',
            f'{b["baseline_copy_count"]} → {b["optimized_copy_count"]}',
            f'{b["saved_dispatch_combine_payload_bytes"]/MiB:.2f}',
            f'{b["incremental_expert_bytes"]/MiB:.2f}',
            saving_html(b['cold_net_saved_payload_bytes']/MiB,2)])
        c=comparison_layer(l);rr=c.get('reuse_analysis',{})
        threshold=rr.get('strict_positive_min_same_version_reuses')
        reuse.append([f'L{l:02d}',f'{rr.get("cold_net_saving_ms",float("nan")):+.3f}',
            f'{rr.get("hot_net_saving_ms",float("nan")):+.3f}',
            '没有有限次数能回本' if threshold is None else str(threshold)])
        for x in c['comparisons']:
            if x['baseline']=='nativecoldearly' and x['candidate']=='optcoldearly':
                lo,hi=x['mean_delta_bootstrap_95pct_ms']
                confidence.append([f'L{l:02d}',f'{x["mean_delta_ms"]:+.3f}',
                    f'[{lo:+.3f}, {hi:+.3f}]',f'{x["observed_positive_pairs"]}/{x["n"]}',
                    '通过' if x['measured_positive_gate_pass'] else '不通过'])
    timing_mode='CUDA Graph' if meta.get('timing_mode')=='graph' else '普通逐次调用'
    raw_link=link(args.input,'本轮全部原始样本')
    analysis_link=link(args.analysis,'配对分析与收益判定')
    source_dir=Path(__file__).resolve().parent
    evidence=[raw_link,analysis_link]
    for filename,label in [('protocol.json','固定实验口径'),('inputs/manifest.json','真实路由来源与两卡转换'),
                           ('provenance/source_archives.json','各轮源码归档与指纹'),
                           ('work_status.json','最终工作状态与设备清理记录'),
                           ('independent_readback.json','独立读取复核结果')]:
        path=args.input.parent/filename
        if path.exists():evidence.append(link(path,label))
    gate_path=args.input.with_name(args.input.name.replace('benchmark_','gate_',1))
    if gate_path!=args.input and gate_path.exists():evidence.append(link(gate_path,'本轮默认禁用配置与收益条件'))
    for filename,label in [('exchange_cuda.cu','扫描与分块计数实现'),('residency_cuda.cu','GPU 驻留选择'),
                           ('residency_cuda_fused.cu','最终融合驻留选择内核'),
                           ('kernel_design.txt','扫描优化机制与复杂度说明'),
                           ('preplanning_gate.py','运行前收益选择规则'),
                           ('moonep_materialize_reuse.cu','复用原有顺序生成地址'),
                           ('moonep_split.py','MoonEP 提前传输适配'),('deepep_v1_adapter.py','DeepEP v1 适配'),
                           ('deepep_trace_smoke.py','DeepEP 真实路由核验入口'),
                           ('deepep_v1_source_manifest.json','DeepEP v1 官方版本与原始文件指纹'),
                           ('deepep-v1-disable-nvshmem.patch','DeepEP v1 仅限机内的构建补丁'),
                           ('deepep-v1-intranode-dlink.patch','DeepEP v1 设备链接补丁'),
                           ('deepep-v1-notify-256.patch','DeepEP v1 通知启动容量补丁'),
                           ('lb-current-stream.patch','DeepEP v2.5 未测可选补丁')]:
        path=source_dir/filename
        if path.exists():evidence.append(link(path,label))
    for path in sorted(args.input.parent.glob('report_attempt*.html')):
        if path.resolve()!=args.output.resolve():evidence.append(link(path,'保留报告：'+path.name))
    first=args.input.parent/'benchmark_graph_attempt01.json'
    if first.exists() and first.resolve()!=args.input.resolve():evidence.append(link(first,'第一轮原始样本'))
    # Only link evidence that has actually been downloaded. No guessed paths.
    failed=[];deep_results=[];hybrid_passes={}
    for path in sorted(args.input.parent.glob('*.json')):
        if path.resolve() in (args.input.resolve(),args.analysis.resolve()):continue
        if '_timeline_' in path.name:continue
        if 'profile' in path.name and not path.name.endswith('_summary.json'):continue
        try:data=json.loads(path.read_text())
        except (ValueError,OSError):continue
        if not isinstance(data,dict):continue
        if data.get('failures') or data.get('status') in ('failed','error','blocked'):
            failure=esc(str(data.get('failures') or data.get('error') or data.get('status'))[:800])
            failed.append([link(path,path.name),esc(data.get('status','见记录')),failure])
        if 'deepep' in path.name.lower() and ('status' in data or 'checks' in data):
            dm=data.get('metadata',{});expert_backend=dm.get('expert_backend',data.get('expert_backend','未记录'))
            hidden=dm.get('hidden',data.get('shape_and_controls',{}).get('hidden','未记录'))
            deep_results.append([link(path,path.name),esc(data.get('status','见记录')),esc(hidden),esc(expert_backend),
                esc(data.get('metadata',{}).get('scope',data.get('workload','以原始记录口径为准')))])
            if data.get('status')=='passed' and not data.get('failures') and expert_backend=='MoonEP one-sided NVLink prefetch':
                exact=all(r.get('checks') and all(c.get('identity_combine_exact') is True and c.get('weight_check')
                          for c in r['checks']) for r in data.get('layers',[]))
                if exact and data.get('layers') and hidden==7168 and data.get('rank') in (0,1):
                    name=path.stem.rsplit('_rank',1)[0]
                    hybrid_passes.setdefault(name,{})[data['rank']]=tuple(sorted(r['layer'] for r in data['layers']))
    failures_html=(rows_table(['原始记录','状态','保留的失败信息'],failed) if failed else
        '<p>当前输出目录没有可列出的独立失败 JSON；本轮正式测量的 failures 为空。此前失败不据此记为成功，导回对应记录后再补链接。</p>')
    hybrid_ok=[name+'（'+','.join(f'L{x}' for x in ranks[0])+'）' for name,ranks in hybrid_passes.items()
               if set(ranks)=={0,1} and ranks[0]==ranks[1]]
    deep_html=(rows_table(['验证记录','状态','H','专家传输实现','范围'],deep_results) if deep_results else
        '<p>当前输出目录尚无 DeepEP 验证结果 JSON，本报告不把适配代码存在当作运行通过，也不填入 DeepEP 性能数字。</p>')
    if hybrid_ok:
        deep_html=('<p class="note">已导回的双卡记录确认：H=7168 的真实路由回放通过，token 发送／返回使用 DeepEP v1 加本地容量补丁，专家预取使用 MoonEP 单边复制。检查包含独特 token 数据、逐 token 专家槽、八份结果精确合并和权重副本。通过的批次：'+
                   esc('、'.join(hybrid_ok))+'。这是混合适配方案的正确性验证，未测专家矩阵乘法，也不是 DeepEP 原生专家预取的性能结论。</p>'+deep_html)
    audit_count=sum(len(r['correctness']) for r in layers)
    hot_hashes=all(s.get('hot_prepared_map_sha256') for r in layers
                  for mode in ('native_hot','optimized_hot') for s in r['measurements'][mode])
    all_no_reuse=all(x.get('reuse_analysis',{}).get('strict_positive_min_same_version_reuses') is None
                     for x in analysis['analysis']['cuda_ms']['layers'])
    reuse_text=('本次每一层在“不再传权重”的热状态下仍然变慢，因此仅靠多次复用权重也无法回本。'
                if all_no_reuse else '复用次数只在权重版本和副本位置都保持不变时有意义，逐层估算如下。')
    reuse_tail=('当前每层热状态也为负，把权重成本摊到更多次调用仍不能变成正收益。'
                if all_no_reuse else '存在有限回本次数也只是同输入、同权重、同副本表下的估算，不是变化输入下的测量保证。')
    baseline_note=('本轮原生对照设置 rt=0，即关闭根据流行度主动增加的专家副本；仍保留均衡负载所需的副本。'+
                   f'额外槽上限为每卡 {meta.get("extra_slots",8)} 个。因此结果比较的是这一原生配置，不能推广成优于所有 MoonEP 参数选择。'
                   if meta.get('replica_target')==0 else '原生对照参数以原始记录和固定协议为准；未记录的参数不作推断。')
    materializer_label={'sort':'重新排序','reuse':'复用原生 planner 已有顺序'}.get(meta.get('materializer'),'旧记录未单列')
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>两卡 MoonEP planner 优化与完整通信成本</title><style>
:root{{--ink:#183145;--muted:#52697a;--line:#d6e0e8;--bg:#f3f6f8;--blue:#1e618b;--bad:#9c3628;--good:#226d4d}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.8 system-ui,-apple-system,"PingFang SC",sans-serif}}header{{background:#163c55;color:white;padding:48px max(24px,calc((100vw - 1160px)/2))}}h1{{font-size:34px;line-height:1.4;margin:12px 0}}header p{{color:#c6dde9}}main{{max-width:1232px;padding:24px;margin:auto}}section{{background:white;border:1px solid var(--line);padding:30px;border-radius:16px;margin:22px 0;scroll-margin-top:18px}}h2{{font-size:26px;line-height:1.4;margin-top:0}}h3{{font-size:20px}}nav{{display:flex;flex-wrap:wrap;gap:10px}}nav a{{background:white;border:1px solid var(--line);border-radius:24px;padding:5px 15px;text-decoration:none}}a{{color:var(--blue)}}p{{margin:13px 0}}.notice{{padding:18px 22px;border-left:5px solid var(--bad);background:#fff1ed;border-radius:8px}}.note{{padding:16px 20px;border-left:4px solid var(--blue);background:#eef5fa}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{padding:18px;border:1px solid var(--line);border-radius:10px;background:#f6fafc}}.card strong{{display:block;font-size:27px;color:var(--blue)}}.bad{{color:var(--bad);font-weight:650}}.good{{color:var(--good);font-weight:650}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:15px;margin:18px 0}}th,td{{border-bottom:1px solid var(--line);text-align:left;vertical-align:top;padding:11px 12px}}th{{background:#edf3f7}}td:first-child{{font-weight:600}}code{{background:#edf2f6;padding:2px 6px;border-radius:4px;overflow-wrap:anywhere}}.formula{{background:#edf4f7;border-left:4px solid #477e83;padding:18px;border-radius:8px;white-space:pre-wrap;font:15px/1.7 ui-monospace,monospace}}.muted{{font-size:14px;color:var(--muted)}}details{{border-top:1px solid var(--line);padding:15px 0}}summary{{cursor:pointer;font-weight:600}}.flow{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}}.flow span{{padding:8px 12px;background:#eaf2f7;border-radius:7px}}footer{{font-size:13px;color:var(--muted);padding:12px;text-align:center}}@media(max-width:720px){{main{{padding:12px}}section{{padding:22px 15px}}header{{padding:32px 24px}}h1{{font-size:27px}}.cards{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}header{{background:white;color:black;padding:20px}}header p{{color:#333}}section{{border:0;border-bottom:1px solid #ccc;border-radius:0;padding:20px 0}}nav{{display:none}}}}
</style></head><body><header><div>2026-11-05 · test-mtp-retrieval-copy · 两张物理 H200</div><h1>{esc(title)}</h1><p>真实路由的两卡派生回放。把规划、专家权重复制、token 发送与结果返回放在同一段里实际测量。</p></header><main>
<nav><a href="#outcome">结论</a><a href="#components">kernel 分项</a><a href="#layers">逐层结果</a><a href="#bytes">通信与副本成本</a><a href="#prefetch">提前传输</a><a href="#gate">收益判定</a><a href="#protocol">范围与证据</a></nav>
<section id="scope-correction"><h2>评估范围修正：目标是八卡，两卡仅用于校准</h2>
<p class="notice">用户指定的设计目标是保留原始八卡路由、44 个原属专家／卡，以及每卡 8 个额外槽。本文此前把八个来源折成两卡重新规划，所以本文的 3.07% 和完整路径负收益仅适用于这个两卡派生实验，<strong>不能用于否定八卡方案，也不能与原始八卡的 24.98% 搭配使用</strong>。</p>
<p>八卡方案改为：从原始八卡回放取得每张卡的发送／接收量，用两张物理 H200 测量 kernel 和传输带宽，再按最忙卡估算八卡时间，并计入训练中的权重刷新和副本梯度归并。见 <a href="../eight_rank_estimate/report.html">八卡路由与两卡校准估算报告</a>。这里保留原始两卡数据供复核。</p></section>
<section id="outcome"><h2>本轮结果：{esc(decision)}</h2>
<p class="notice">{esc(headline)} <strong>不能把分项提速当作整层或模型提速。</strong></p>
<div class="cards"><div class="card"><strong>{prefix_saved:.1f}%</strong>交换组件用时下降：{sort:.4f} → {prefix:.4f} ms；两条路径的执行位置和统计逐项相同。</div><div class="card"><strong class="{'bad' if cold_change>0 else 'good'}">{cold_base:.3f} → {cold_candidate:.3f} ms</strong>冷状态下，各方法取本轮整体平均最快的调度方式，当前算法{'慢' if cold_change>0 else '快'} {abs(cold_change):.2f}%。</div><div class="card"><strong class="{'bad' if hot_change>0 else 'good'}">{hot_base:.3f} → {hot_candidate:.3f} ms</strong>权重已经就绪时，当前算法{'慢' if hot_change>0 else '快'} {abs(hot_change):.2f}%。</div></div>
<p>这里“冷状态”表示本次调用确实复制所需专家权重；“热状态”表示所需专家已在准确的物理槽中、权重版本未变化，因此本次不再复制。两种状态都重新执行 planner。planner 是决定任务位置并生成执行地址的程序。</p>
{rows_table(['完整路径','GPU 时间 / ms','主机观察时间 / ms','相对同冷热最快 MoonEP'],mode_rows)}
<p class="muted">先对每次重复取两张卡中较慢者，再取各层中位数，最后对 {len(layers)} 层等权平均。“主机观察时间”从发起该段到确认 GPU 完成；不含段外的统一同步。所有完整路径直接计时，不由组件相加。</p></section>

<section id="components"><h2>扫描优化确实降低了交换组件成本</h2>
<p>kernel 指一次在 GPU 上执行的计算程序。原路径为每个 token（输入的一条数据）的交换提议排序。新路径先保存每个 token 哪些任务能在每张卡执行，用一个整数的二进制位表示任务集合；后续交换复用这些小表，本轮调用做 {meta['rounds']} 轮交换。随后按固定大小的数据块统计各类提议数量，把前面各块的数量相加，就知道本块从哪个名额开始，保留原先的稳定顺序。</p>
<p>这同时减少重复读取、重复判断和全量排序。当前仍为每个 token 最多保留一个提议，再按反方向、同样任务数匹配；没有改变这批输入的最终结果。</p>
{rows_table(['组件','GPU 时间 / ms','计入内容'],component_rows)}
<p class="note">完整 planner 为 MoonEP {average('native_planner'):.3f} ms、当前算法 {average('optimized_full_planner'):.3f} ms。本轮最终地址生成方式：{esc(materializer_label)}。单独测得的各组件不能直接相加成完整路径：发起间隙、同步、缓存状态和并行执行会改变实际用时。</p>
<!-- BEGIN subsequent-measurements -->
<div id="subsequent-measurements">{comparison_html}{cohort_html}</div>
<!-- END subsequent-measurements -->
</section>

<section id="layers"><h2>逐层结果，保留全部负收益</h2><p>“节省”统一定义为 MoonEP 用时减去当前算法用时；负数表示变慢。冷状态的参照取该层两种 MoonEP 调度中较快者，候选也取该层两种调度中较快者。最终启用判定还要求同一种方式通过全部层的配对检查。</p>
{rows_table(['层','MoonEP 冷最快 / ms','当前冷串行 / ms','当前冷提前传 / ms','冷最好节省 / ms','MoonEP 热 / ms','当前热 / ms','热节省 / ms'],per_layer)}
<details><summary>配对样本的波动范围：双方均提前传权重</summary>
<p>每轮使用同一层输入，并轮换测试顺序。“95% 区间”通过反复重采样配对差值估计平均节省的范围；它不是下一次调用的保证。完整每轮、每卡样本见 {raw_link}。</p>
{rows_table(['层','平均节省 / ms','平均节省 95% 区间 / ms','观察到正收益的重复','判定'],confidence)}</details></section>

<section id="bytes"><h2>减少的 token 数据与额外权重复制</h2>
<p>{len(layers)} 层合计，远端 token 份数从 <strong>{before:,}</strong> 降到 <strong>{after:,}</strong>，下降 <strong>{reduction:.2f}%</strong>。同一个 token 发到同一张卡执行多个专家只算一份，称为“去重后的远端份数”。</p>
<p>按照本轮隐藏维度 H={meta['hidden']}、专家中间维度 I={meta['intermediate']}、BF16（每个数 2 字节），三组专家矩阵合计 <strong>{meta['expert_bytes']/MiB:.0f} MiB／专家</strong>。这些形状是测试设定，router trace 本身不证明真实模型的参数尺寸。实验确实分配并通过 NVLink 复制了这份字节量，内容使用可核验的标记。</p>
<div class="formula">每个专家字节数 = 3 × H × I × 2
token 发送与返回合计节省 = 减少的远端份数 × H × 2 字节 × 2 个方向
额外专家成本 = 当前算法的复制字节 − MoonEP 原本也要复制的字节</div>
{rows_table(['层','远端份数：前 → 后','份数下降','两卡合计专家副本：前 → 后','token 发送＋返回节省 / MiB','额外专家复制 / MiB','冷状态净少传 / MiB'],traffic)}
<p>全 {len(layers)} 层少传 token 数据 <strong>{saved_bytes/GiB:.3f} GiB</strong>，却额外传专家权重 <strong>{extra_bytes/GiB:.3f} GiB</strong>。MiB=2²⁰ 字节，GiB=2³⁰ 字节。表中未计通信协议、路由附加信息、对齐填充和训练梯度；字节差也不能直接换算成时间收益。</p>
<p>两卡拓扑每个 token 最多访问一张远端卡，减少这次远端访问必须把其远端任务全部留回本卡。这与此前八逻辑卡回放的机会数量不同，不能沿用此前约 25% 的下降比例。</p>{retention_html}</section>

<section id="prefetch"><h2>提前传权重：执行顺序与实际重叠</h2>
<div class="flow"><span>专家副本表确定</span><span>→ 独立执行流传专家</span><span>同时继续全量 token 规划</span><span>→ token 计划确定即发 token</span><span>→ 两边完成后才能算专家</span></div>
<p>执行流是 GPU 的一条有序任务队列。MoonEP 适配器用独立队列提前复制专家权重，token 通信等待自己的地址计划，不等待权重；真正的专家计算必须等待两边都完成。权重表、物理槽和源权重版本在使用期间保持稳定。</p>
<p>本轮同时测试 MoonEP 和当前算法的提前传输方式。当前算法提前传输的平均用时为 {average('optimized_cold_early'):.3f} ms，串行为 {average('optimized_cold_serial'):.3f} ms。提前发起不保证更快：两类传输仍共享 NVLink、显存带宽和 GPU 执行资源。具体瓶颈需要性能分析，不能仅凭这组总时间指定原因。</p>
<p>原包 MoonEP 的 planner 把内部步骤合在一个 kernel 内，外部只能在整个 planner 完成后拿到事件。当前算法的专家表在全量交换前已确定，因此可在这个位置发起预取。若要进一步在原生 planner 内部更早开始，需要额外拆分或发布就绪信号，不能把目前接口说成已经做到。</p>
<h3>本轮时间线诊断</h3>{timeline_html}
<h3>DeepEP 的版本与验证范围</h3>
<p>目标环境为 CUDA {esc(meta['cuda'])}、NCCL {'.'.join(map(str,meta['nccl']))}。兼容测试选择官方 DeepEP v1.2.1，固定版本 <code>9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee</code>。专家先转换为“目标卡＋本地物理槽”的编号，再交给原有 dispatch（发送 token）和 combine（返回并合并结果）。该版本的接收数据按 token／目标卡去重，真实专家矩阵计算还需要额外整理数据，不能直接用原始通信时间对比 MoonEP 的完整专家布局。</p>
<p>v1 的两个本地补丁仅调整构建：显式关闭本轮不用的跨机依赖，并补齐 GPU 设备代码的链接步骤；没有改通信 kernel，也没有替换用户现有 CUDA／NCCL 环境。补丁文件和原始指纹均保留。构建、导入、两卡正确性是不同阶段，只有实际结果文件能证明对应阶段通过。</p>
<p>导入成功后的首次通信仍被上游形状限制拒绝：每卡有 176 个原专家和 8 个副本槽，合计 184 槽，通知程序原先仅启动 128 个线程。另一个容量补丁让 129—256 槽使用 256 个线程，保留原来的专家编号与通信数据算法。这项改动已属于本地 kernel 启动适配，因此后续结果必须标作“v1.2.1 加本地容量补丁”；是否通过仍以新的两卡结果文件为准。</p>
<p>H=128 的缩小诊断形状曾在返回结果时失败，记录保留。代码审阅发现该形状使返回程序中的一轮处理只覆盖半个 32 线程小组，而循环内部要求整个小组同步；这是待验证的原因。交付入口已保守地要求 BF16 的 H 为 256 的倍数，默认 H=7168。实际通过范围按 H=7168 的结果文件限定，其他符合倍数要求的尺寸也需要各自验证。</p>
{deep_html}
<p>DeepEP v2.5 已提供原生专家预取，但要求 CUDA 13.1 及以上、NCCL 2.32.3 及以上，与本机不一致。其默认专家预取和 token 通信共用一条通信队列；提供的独立队列补丁还使用第二个 EPBuffer（通信缓冲对象）隔离同步信号和临时存储。<strong>这份 v2.5 可选补丁尚无本机测量，不能列为已验证加速。</strong></p></section>

<section id="gate"><h2>收益判定：{esc(decision)}</h2>
<p>冷状态判定：<strong>{'通过' if cold_pass else '不通过'}</strong>；热状态判定：<strong>{'通过' if hot_pass else '不通过'}</strong>。同一个调度方式必须在 GPU 与主机两种时间下都优于两种原生对照，不能分别挑不同方式拼出一个虚构胜者。</p>
<p>{esc(reuse_text)}</p>
{rows_table(['层','首次冷调用节省 / ms','之后每次热调用节省 / ms','保持同版本／同副本表的回本次数'],reuse)}
<p>回本次数来自“第一次冷调用、之后均为热调用”的估算，双方都享有复用。跨权重更新或副本位置变化时，必须重新传对应权重。{esc(reuse_tail)}</p>
<p class="notice">运行完选副本和交换后再退回 MoonEP，已经花掉的规划时间不会消失。因此“收益不好就回退”本身不能保证零损失。当前判定是这组已测工作负载的启用结论，不能承诺任意未来输入都有正收益。</p></section>

<section id="protocol"><h2>测量范围、正确性和原始证据</h2>
<p class="note">{esc(baseline_note)}</p>
<p>设备为指定机器上的两张物理 H200，经 NVLink（GPU 之间的高速连接）互通。保留原始八层路由记录的全部行，把原来源卡 0—3 合并到物理卡 0，4—7 合并到物理卡 1，专家按 176 个／卡划分。每层共有 {layers[0]['shape'][0]*layers[0]['shape'][1]:,} 个 token，每个 token 选 8 个专家；这是一份两卡派生拓扑，不是原始生产两卡路由记录。</p>
<p>每层每种方式预热 {meta['warmup']} 次，保留 {meta['samples']} 次配对重复；本报告使用 {timing_mode}。CUDA Graph 是预先记录一串固定 GPU 操作后整体重放，减少主机逐个发起的间隙。图捕获、编译、内存分配、CPU 参考核验、热状态预填不计入该段。局部专家直方图与随机样本编号已准备；驻留选择仍在每次调用中从 GPU 路由和执行位置刷新样本并汇集两卡数据。</p>
<p>本轮 {len(layers)} 层共 {audit_count} 项记录的正确性检查全部通过，包括原生 CPU 对照、GPU 驻留选择、排序／分块交换结果一致、计算负载守恒、专家驻留合法、实际传输数据与权重副本校验。热调用逐样本准确预填标记为 {esc(meta.get('hot_exact_map_prefill_per_sample',False))}，原始样本保留映射指纹：{'完整' if hot_hashes else '不完整，热状态不可据此认证'}。</p>
<p><strong>没有专家 GEMM（矩阵乘法）、反向传播、梯度合并和优化器更新。</strong>发送后的专家输出直接沿用输入数值，验证真实返回通信和每 token 的八份结果归并。因此所有完整时间只代表规划／通信段，不是完整模型或训练加速。</p>
<p>{' · '.join(evidence)}</p>
<p class="muted">原始测量 SHA256：<code>{digest}</code>。分析文件已核对这一指纹。原始样本还保存输入与源码指纹；链接到的工作区源码后续可能继续修改，应以本轮保存的指纹为准。</p>
<details><summary>保留的失败与不完整尝试</summary>{failures_html}</details></section>
</main><footer>所有表格均从给定原始文件重新计算。负收益、失败与未测边界保留。</footer></body></html>'''
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(doc)
    print(json.dumps(dict(output=str(args.output),raw_sha256=digest,layers=len(layers),
                         cold_gate=cold_pass,hot_gate=hot_pass),ensure_ascii=False))


if __name__=='__main__':
    main()
