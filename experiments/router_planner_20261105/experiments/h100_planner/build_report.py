"""Summarize accepted GPU samples and render the offline Chinese report."""
from pathlib import Path
import argparse,json,statistics as stats,html
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args();raw=json.loads(a.input.read_text());assert not raw['failures'],raw['failures']
rows=[r for r in raw['rows'] if 'scope_timings_ms' in r];assert len(rows)==8 and {r['layer'] for r in rows}==set(range(8));assert all(all(r['graph_audit'].values()) for r in rows)
scopes={}
for f,count in [('scope_timings_ms',10),('host_timings_ms',10),('graph_timings_ms',20)]:
 scopes[f]={}
 for name in rows[0][f]:
  assert all(len(r[f][name])==count for r in rows)
  med=[stats.median(r[f][name])*1000 for r in rows]
  scopes[f][name]=dict(mean_layer_median_us=stats.mean(med),layer_medians_us=med,min_layer_median_us=min(med),max_layer_median_us=max(med))
quality={}
for r in raw['rows']:
 if 'quality' not in r:continue
 assert r['exact_cpu'] and r['materialized'] and r['quality']['audit_pass']
 q=quality.setdefault(r['variant'],dict(base=0,remote=0,layers=0));q['base']+=r['base_remote'];q['remote']+=r['quality']['remote'];q['layers']+=1
for q in quality.values():assert q['layers']==8;q['reduction_pct']=(1-q['remote']/q['base'])*100
out=ROOT/'outputs/h100_planner';summary=dict(source=a.input.name,protocol='Arithmetic mean of eight per-layer medians; 10 eager, 20 Graph samples/layer. All 56 route audits and all Graph audits pass.',quality=quality,scopes=scopes)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
def us(name,graph=True):return scopes['graph_timings_ms' if graph else 'scope_timings_ms'][name]['mean_layer_median_us']
def ms(name,graph=True):return f'{us(name,graph)/1000:.3f}'
labels=dict(bundle='原成组交换：16 线程',bundle8='成组交换：8 线程',aggregate_aos='完整状态聚合：原布局',aggregate_soa='完整状态聚合：连续字段布局',subset_exact='共享子集：顺序处理 28 对卡',subset_parallel='共享子集：同时提议、扫描模板',subset_indexed='共享子集：同时提议、直接查表')
def component(name):return name+'_prepare_exchange' if name.startswith('subset') else name
table='<table><thead><tr><th>方案</th><th>通信份数下降</th><th>普通调用：准备＋交换 / ms</th><th>Graph：准备＋交换 / ms</th></tr></thead><tbody>'
for n,label in labels.items():
 key=component(n);table+=f'<tr><td>{label}</td><td>{quality[n]["reduction_pct"]:.2f}%</td><td>{ms(key,False)}</td><td>{ms(key) if key in scopes["graph_timings_ms"] else "未实现 Graph"}</td></tr>'
table+='</tbody></table><p class="sub">所有方案都使用同一真实 trace、同一份驻留表、每卡最多 8 个额外专家、四轮交换。准备＋交换包含共享模板统计／选取和索引建表、聚合的状态分组；不包含选择驻留位置。聚合两版只能用普通调用列比较。</p>'
scope='<table><thead><tr><th>范围／方案</th><th>普通调用 / ms</th><th>Graph / ms</th></tr></thead><tbody>'
for n,label in [('moonep_port_full','MoonEP R8 语义移植：起始分配与输出'),('seed_only','只生成起始 owner'),('materialize_only','只生成最终执行地址等元数据')]+[(n+'_seed_exchange_output',label+'：起点＋准备／交换＋输出') for n,label in labels.items()]:
 scope+=f'<tr><td>{label}</td><td>{ms(n,False)}</td><td>{ms(n) if n in scopes["graph_timings_ms"] else "未实现 Graph"}</td></tr>'
scope+='</tbody></table><p class="sub">这是固定驻留下的计划输出路径。各段单独测量之和不必等于整体路径：CPU 发起间隙可与前段 GPU 工作重叠，缓存与调度也不同。整体路径已经直接计时，没有用分段之和估算。</p>'
reduce8=100*(1-us('bundle8')/us('bundle'));reduceidx=100*(1-us('subset_indexed_prepare_exchange')/us('subset_parallel_prepare_exchange'))
conclusion=f'''<p class="lead"><strong>可以避免在多个候选名单中反复扫同一个 token；但这份 trace 上，综合通信收益与规划时间，8 线程成组交换更合适。</strong>查表版把候选扫描改成从 token 自己的单／双专家组合出发直接查询，并保留同时版的全部选择结果。</p><div class="grid"><div class="card"><b>{ms('bundle8')} ms · 24.98%</b>8 线程成组的四轮交换。相对 16 线程版本快 {reduce8:.2f}%，通信结果完全相同。</div><div class="card"><b>{ms('subset_indexed_prepare_exchange')} ms · 25.29%</b>共享子集查表的准备＋交换。相对扫描模板版快 {reduceidx:.2f}%，两版通信结果完全相同。</div><div class="card"><b>{ms('subset_exact_prepare_exchange')} ms · 26.82%</b>保留原卡对顺序的共享子集。收益更高，但有 112 次匹配／提交阶段。</div></div><p>索引同时版比成组多降低约 <strong>0.31 个百分点</strong>的通信份数，准备＋交换却约为其 <strong>{us('subset_indexed_prepare_exchange')/us('bundle8'):.2f} 倍</strong>。完整聚合的连续字段布局普通调用约 {ms('aggregate_soa',False)} ms；当前精确混合匹配中的顺序依赖不能靠改内存排列解决。</p><p>这里的百分比是从最终分配重新计算的去重远端 token 份数，相对原生 rt0 分配；不是模型层加速百分比。上面三张卡片均为 CUDA Graph 实测，单位是毫秒。</p>'''
notes=f'''<h3>测量与诊断证据</h3><p>设备为指定 h100-omni 的单张 H100 80GB HBM3，PyTorch {html.escape(raw['metadata']['torch'])}，CUDA {html.escape(raw['metadata']['cuda'])}。未锁频、未改功率；前后检查均无其他 GPU 计算进程。每层保留 10 次普通调用与 20 次 Graph 样本，按轮换顺序测试；先取每层中位数，再对八层等权平均。编译、Graph 捕获、输入统计和正确性检查不计入延迟。</p><p><a href="{a.input.name}">本表逐层原始样本与审计</a> · <a href="summary.json">汇总 JSON</a> · <a href="environment_before_attempt02.txt">运行前设备状态</a> · <a href="environment_after_attempt02.txt">运行后设备状态</a> · <a href="baseline_smoke_02.json">起始分配／成组逐元素核对</a>。</p><p>独立诊断只取第 0 层、三次执行，并启用了 profiler，<strong>诊断时间不替代上表</strong>。完整聚合中，动态规划 solve_dp 占 kernel 时间和的 71.46%，类别前缀扫描占 20.67%；两者合计约 92.13%。这解释了仅改代表状态的读取布局为何没有明显改善。</p><p>首次查表实现中，单线程累加 768 类的起始位置占 kernel 时间和的 15.15%，约 29.2 μs／轮。后续改为 256 线程分三段计算前缀，保持输出不变。对应诊断降为 2.44 μs／轮，约占 1.47%；<a href="profile_prefix_02/summary.json">优化后诊断</a>与<a href="../h100_subset_validation/subset_smoke_results_03.json">16 项受影响结果核验</a>另行保留。首次完整计时、源码快照与诊断均保留：<a href="benchmark_attempt01.json">第一次计时</a>、<a href="summary_attempt01.json">第一次汇总</a>、<a href="profile_prepare_exchange_01/summary.json">诊断明细</a>。</p><p>这些诊断没有采集 Nsight Compute 的缓存命中率、内存事务或寄存器溢出指标；不能据此声称已达到带宽上限，或把每项收益都归因于 HBM 访存减少。</p>'''
native='''<p><strong>原包原生延迟：本次没有测得。</strong>保留原源码的单 rank 初始化在 <code>cuMulticastCreate</code> 返回 <code>invalid argument</code>，发生于 <code>nvl_shared_buffer.cuh:578</code>。尚未进入 planner kernel，不能记为 0 μs 或用语义移植版本代替。<a href="native_status.json">状态记录</a> · <a href="native_r1_compile_only_failed.log">原始失败日志</a>。</p>'''
text=(out/'report_template.html').read_text()
for marker,value in dict(__CONCLUSION__=conclusion,__GPU_TABLE__=table,__SCOPE_TABLE__=scope,__TIMING_NOTES__=notes,__NATIVE__=native).items():text=text.replace(marker,value)
assert '__GPU_' not in text and '__CONCLUSION__' not in text
(out/'report.html').write_text(text)
print('Built',out/'report.html','from',a.input)
