"""Standalone Chinese report: original R8 traffic, measured R2 calibration.

The displayed communication costs are model outputs, never relabeled eight-GPU
measurements. Independent kernel components are kept separate from full planner.
"""
from pathlib import Path
import argparse,html,json,statistics,hashlib

ROOT=Path(__file__).resolve().parents[2]
LABELS={'native_rt0':'MoonEP rt=0','native_rt2':'MoonEP rt=2','sample256_g2_bundle':'成组交换方案'}
GIB=2**30

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--estimate',type=Path,required=True)
    ap.add_argument('--kernels',type=Path,nargs='*',default=[])
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/eight_rank_estimate/report.html')
    args=ap.parse_args();j=json.loads(args.estimate.read_text())
    assert j['model']['H']==7168 and j['model']['I']==2048 and j['model']['microbatches_per_update']==1, 'This narrative is pinned to H7168 I2048 M1'
    assert j['model'].get('weight_refreshes_per_layer',1)==1
    t=json.loads((ROOT/'outputs/eight_rank_estimate/traffic.json').read_text())
    s=j['scenarios'][0];agg=s['aggregate'];diff=s['optimized_vs']['native_rt0'];byte=j['traffic_payload']['optimized_vs']['native_rt0']
    assert (t['protocol']['N'],t['protocol']['R'],t['protocol']['L'],t['protocol']['sample_seed'])==(134144,8,4,20261026)
    pinned_rates={'dispatch':156.32264378892432,'combine':182.01002257010302,'weight':330.3146385351925,'gradient':278.1711250126369}
    assert all(abs(s['effective_endpoint_GBs'][k]-v)<1e-9 for k,v in pinned_rates.items()), 'Revise the pinned narrative before using a different calibration'
    source=j['inputs']['traffic'];assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256']
    compare=[]
    for base in ('native_rt0','native_rt2'):
        a=t['aggregate'][base]['totals'];b=t['aggregate']['sample256_g2_bundle']['totals'];v=j['traffic_payload']['optimized_vs'][base]
        compare.append([LABELS[base],f"{a['remote_token_destinations']:,} → {b['remote_token_destinations']:,}",
            f"{100*(1-b['remote_token_destinations']/a['remote_token_destinations']):.2f}%",
            f"{a['expert_copies_per_full_prefetch']} → {b['expert_copies_per_full_prefetch']}",
            str(v['extra_expert_copies']),f"{v['extra_weight_GiB']:.2f}",f"{v['extra_gradient_GiB']:.2f}",f"{v['saved_training_token_GiB']:.2f}"])
    costs=[]
    phases=[('training_token_ms','前向＋反向 token'),('weight_ms','BF16 权重刷新'),('gradient_ms','FP32 副本梯度归并'),('training_M1_serial_ms','三项相加，未计 planner')]
    for key,label in phases:
        costs.append([label,*[f'{agg[m][key]:.3f}' for m in LABELS],f"{agg['sample256_g2_bundle'][key]-agg['native_rt0'][key]:+.3f}"])
    per=[]
    for row in s['layers']:
        l=row['layer'];raw=t['layers'][l]['methods'];base=raw['native_rt0'];opt=raw['sample256_g2_bundle']
        maxcopies=lambda m:max([x['expert_tx_copies'] for x in m['per_rank']]+[x['expert_rx_copies'] for x in m['per_rank']])
        v=row['optimized_vs']['native_rt0']
        per.append([l,f"{base['totals']['expert_copies_per_full_prefetch']} → {opt['totals']['expert_copies_per_full_prefetch']}",
            f'{maxcopies(base)} → {maxcopies(opt)}',f"{v['serial']['saved_ms_before_extra_planner']:.3f}",
            f"{v['ideal_all_weights_hidden']['saved_ms_before_extra_planner']:.3f}"])
    cal=[]
    for key,label in [('dispatch','token 发送'),('combine','结果返回'),('weight','权重复制'),('gradient','梯度归并')]:
        c=j['calibration']['selected'][key];rg=j['calibration']['shape_direction_ranges'][key]
        cal.append([label,f"{c['effective_endpoint_GBs']:.2f}",f"{c['median_max_rank_cuda_ms']:.3f}",
            f"{c['payload_endpoint_bytes']/2**20:.1f}",f"{rg['min_case_effective_endpoint_GBs']:.1f}–{rg['max_case_effective_endpoint_GBs']:.1f}"])
    sensitivity=[]
    for scenario in j['scenarios']+j.get('independent_token_bandwidth_scenarios',[]):
        r=scenario['effective_endpoint_GBs'];v=scenario['optimized_vs']
        sensitivity.append([html.escape(scenario['name']),'/'.join(f'{r[x]:.0f}' for x in ('dispatch','combine','weight','gradient')),
            f"{v['native_rt0']['serial']['saved_ms_before_extra_planner']:+.2f}",
            f"{v['native_rt0']['ideal_all_weights_hidden']['saved_ms_before_extra_planner']:+.2f}",
            f"{v['native_rt2']['serial']['saved_ms_before_extra_planner']:+.2f}",
            f"{v['native_rt2']['ideal_all_weights_hidden']['saved_ms_before_extra_planner']:+.2f}"])
    kernels='<p>原始八卡形状的组件测试尚未导入本报告。此前两卡版本约 170 μs 的增量不作为八卡 planner 时间。</p>'
    kernelrows=[];kerneldata=[]
    for path in args.kernels:
        q=json.loads(path.read_text());assert q['status']=='complete' and not q['failures']
        assert q['metadata']['logical_ranks']==8 and q['metadata']['tokens_per_source']==16768
        kerneldata.append(q)
    if kerneldata:
        for layer in range(8):
            rows=[next(x for x in q['layers'] if x['layer']==layer) for q in kerneldata]
            for row in rows:assert row['post_timing_correctness'] and all(row['correctness'].values())
            res=max(row['global_components']['residency']['timing']['median_cuda_ms'] for row in rows)
            old=max(z['sort_L4_conditional']['median_cuda_ms'] for row in rows for z in row['source_components'])
            new=max(z['prefix_L4_conditional']['median_cuda_ms'] for row in rows for z in row['source_components'])
            kernelrows.append([layer,f'{res*1000:.2f}',f'{old*1000:.2f}',f'{new*1000:.2f}',f'{(1-new/old)*100:+.1f}%'])
        means=[statistics.mean(float(r[i]) for r in kernelrows) for i in (1,2,3)]
        kernels='<p>两张物理 H200 各自测原始八逻辑卡维度。每个来源实际扫描 16,768 个 token，方向仍为八卡。逐层先取每个来源的重复中位数，再取来源与物理卡中最大值；不把全量耗时除八。</p>'+f'<p>八层均值：四轮交换从 <strong>{means[1]:.2f} μs</strong> 降到 <strong>{means[2]:.2f} μs</strong>，下降 <strong>{100*(1-means[2]/means[1]):.1f}%</strong>；新增完整八卡驻留选择组件约 <strong>{means[0]:.2f} μs</strong>。可见 170 μs 不能沿用为八卡新增 planner 总成本。</p>'+table(['层','全局驻留选择 / μs','原排序交换四轮 / μs','前缀交换四轮 / μs','交换节省'],kernelrows)+'''<p>驻留选择包括 256 个样本的读取、28 对卡的组计数、前两组筛选以及共享预算的贪心选择。交换复用“每个 token 可在哪些卡执行”的位图，用分块计数及前缀和代替排序；每个 token 仍只提交一个最佳提议。</p>
<p class="warning">这些是组件实测。所有来源先在 GPU 上联动，逐轮验证实时计数、配额和最终执行卡；来源计时阶段则条件化使用已验证的全局计数。尚不包含真实八卡统计汇总、样本收集、专家表广播和最终地址发布，也不是完整八卡原生 MoonEP planner 比较。</p>'''
    payload=json.dumps({'traffic':t['layers'],'rates':s['effective_endpoint_GBs']},separators=(',',':')).replace('</','<\/')
    doc='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>八卡 MoonEP：原路由、两卡校准与训练成本估算</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:#193349;font:17px/1.8 system-ui,-apple-system,"PingFang SC",sans-serif}header{background:#173b54;color:white;padding:40px max(24px,calc((100vw - 1160px)/2))}header p{color:#d6e4ed}h1{font-size:32px;line-height:1.4}main{max-width:1208px;margin:auto;padding:24px}section{background:white;border:1px solid #d8e2e9;border-radius:14px;padding:27px;margin:20px 0}h2{font-size:25px;margin:0 0 14px}h3{font-size:20px}.note,.warning{padding:16px 20px;border-left:4px solid #2377a2;background:#eef6fc}.warning{border-color:#b35c36;background:#fff3ec}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card{padding:20px;background:#eef5f9;border-radius:10px}.card strong{display:block;font-size:28px;color:#195b83}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:15px}td,th{text-align:left;padding:12px;border-bottom:1px solid #d8e2e9}th{background:#edf3f7;white-space:nowrap}a{color:#17658f}code{background:#eef2f6;padding:2px 5px;word-break:break-word}pre{background:#eef4f7;padding:18px;white-space:pre-wrap;border-radius:8px;font:15px/1.7 ui-monospace,monospace}.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:end}.controls label{display:flex;flex-direction:column;font-size:14px}.controls input{width:108px}.controls input,.controls select{padding:8px;border:1px solid #b6c9d5;border-radius:6px;font:inherit}.muted{color:#607486;font-size:14px}.result{font-size:22px;font-weight:650;color:#914b2d}nav{display:flex;gap:15px;flex-wrap:wrap}details{margin:15px 0}summary{cursor:pointer;font-weight:600}@media(max-width:700px){main{padding:12px}section{padding:20px 14px}.cards{grid-template-columns:1fr}h1{font-size:26px}header{padding:30px 22px}}
</style></head><body><header><div>2026-11-05 · 原始八卡路由 · 两张 H200 校准</div><h1>八卡通信收益，按八卡流量重新计算</h1><p>八卡的来源、专家归属、预算和执行计划全部保留。两卡只测计算组件与传输带宽。</p></header><main>
<nav><a href="#answer">结论</a><a href="#counts">流量</a><a href="#model">时间怎么算</a><a href="#time">时间账</a><a href="#whatif">调整假设</a><a href="#kernels">kernel</a><a href="#evidence">证据</a></nav>
<section id="answer"><h2>先修正此前的比较范围</h2><p class="warning">此前把原始八个来源合成两卡后得到的 3.07% 和两卡完整路径负收益，不能用于评价八卡算法。这里重新用原始八卡 trace 的 24.98% 计算，所有八卡传输时间均明确标为估算。</p>
<div class="cards"><div class="card"><strong>24.98%</strong>相对 rt=0，八层去重远程 token 份数减少</div><div class="card"><strong>+436 份</strong>八层额外专家副本；71 → 507</div><div class="card"><strong>+35.77 GiB</strong>八层额外 BF16 权重；另有 71.53 GiB FP32 梯度</div></div>
<p>主训练场景为每次更新只有一个微批，即每层执行一次前向和反向、一次 FP32 副本梯度归并，并先按一次权重刷新计算。这要求副本从前向保留到反向；若跨层共享槽被覆盖，需要再刷新，成本另列。H=7168、I=2048、激活与权重 BF16；这些是约定的测试尺寸，router dump 没有模型权重尺寸。</p>
<p>按两卡实测有效带宽外推、逐层按最忙卡计费，八层 token 通信估计节省 <strong>51.642 ms</strong>，但额外权重约 <strong>25.866 ms</strong>，额外梯度约 <strong>61.428 ms</strong>。不计额外 planner 时，串行通信合计增加 <strong>35.652 ms</strong>。这是当前假设下的通信估算，不是八卡训练实测。</p>
<p>即使双方权重传输都能完全隐藏，当前模型相对 rt=0 仍增加约 <strong>9.787 ms／八层</strong>。相对 rt=2 则不同：这个最乐观场景留下 <strong>6.584 ms／八层</strong> 的 planner 预算；它不能证明实际预取已经取得正收益。</p></section>
<section id="counts"><h2>24.98% 是真的，但只表示 token 流量</h2><p>一个 token（输入的一行向量）在同一张远程卡执行多个专家，只传一份向量。这次减少的是这样的“token—远程目的卡”份数，原有计算任务仍恰好执行一次、每张卡计算对总数仍相等。</p>'''+table(['参照','远程份数：原生 → 优化','降幅','副本数：原生 → 优化','多复制','多权重/GiB','多梯度/GiB','少训练 token/GiB'],compare)+'''
<pre>每专家参数个数 = 3 × H × I = 44,040,192
BF16 权重：每数 2 字节 → 84 MiB／份
FP32 副本梯度：每数 4 字节 → 168 MiB／份
一次训练的 token：前向发出、前向返回、反向发出、反向返回，共 4 次
节省 token 字节 = 少去的远程目的卡份数 × H × 2 × 4</pre>
<p>相对 rt=0，八层少传 token 67.42 GiB，却多传权重＋梯度 107.30 GiB。只看前向时，新增权重是 35.77 GiB，省下的 token 是 33.71 GiB：比值约 1.06，之前两卡折叠实验的 9.83 倍不适用于这里。</p>
<p>每卡 8 个额外槽限制的是<strong>接收端</strong>。例如第 3 层，优化后某个专家原属卡要向多个目的卡合计送出 21 份，而原生最忙卡是 4 份。因此总流量平均分给八卡会低估权重和梯度热点。</p></section>
<section id="model"><h2>时间按每张卡的收发量计算</h2><p>D[s,d] 表示原在卡 s 的 token 有多少份需去卡 d，本地项置零。C[h,d] 表示原属卡 h 要向卡 d 复制多少份专家。先从真实计划建立这两个 8×8 表，再为每种传输换算字节。返回方向把行和列对调。</p>
<pre>TX[i] = 第 i 行之和：卡 i 要发出的字节
RX[i] = 第 i 列之和：卡 i 要收到的字节
该阶段时间 ≈ max(所有 TX，所有 RX) ÷ 该类传输的单向有效带宽
八层时间 = 每层先这样算，再把八层时间相加</pre>
<p>默认假设八卡在同一台 NVSwitch 服务器，八卡能并行收发。不能拿全系统字节除单卡带宽，也不能先把八层流量汇总再平均。这里的模型忽略协议头和对齐填充；两卡有效带宽包含实际 kernel 的本地读写及同步，但不保证八路并发仍有同样效率。</p>
<p><a href="https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/">NVIDIA Hopper 官方说明</a>中的 900 GB/s 是每 GPU 双向带宽。本报告使用实测单向有效值，没有把 900 GB/s 代入单向公式。</p>'''+table(['两卡实测操作','单向有效 GB/s','较慢卡中位时间/ms','端点负载/MiB','不同规模／方向 GB/s'],cal)+'''
<p class="muted">校准共 16 个组合，每组合 10 次，保留每次、每卡的 CUDA 与主机时间，检查真实权重、副本梯度和 token 内容。token 校准的两个目的卡与原八卡的数据分布不同；这些速度是外推参数，并非八卡协议的实测速度。首轮图捕获事件失败记录也保留。</p></section>
<section id="time"><h2>八层通信时间账：先令额外 planner 为零</h2>'''+table(['通信部分／ms','MoonEP rt=0','MoonEP rt=2','成组交换','相对 rt=0 增加'],costs)+'''
<p>如果额外 planner 花费 P 毫秒，就在上述“成组交换”列的总时间再加 P。此前两卡实现约 170 μs 的增量不代入这里。共同的模型计算、优化器和未测原生八卡 planner 都不在此表；这里不是完整训练迭代时间。</p>
<p class="warning">一次刷新是保留副本的假设。原包的额外权重槽跨层共享，若前向后被覆盖，反向再次使用该专家时可能需重新复制。把每层都刷新两次作为敏感性情景，八层三方案串行估算变为 268.651／310.087／330.169 ms，优化相对 rt=0 增加 61.518 ms。实际哪些层需重拷取决于执行顺序及缓存；不能默认所有层恰好两次。</p>
<p>预取最多能把权重藏在后续规划和 token 发送期间。理想就绪时刻为 <code>max(权重时间，剩余规划＋token 发送时间)</code>；实际双方还会争用 NVLink、显存带宽和计算资源。不能因为两个 stream 同时运行就把权重成本清零。</p>'''+table(['层','专家副本数','最忙端点的副本数','串行：节省 ms','理想权重全隐藏：节省 ms'],per)+'''
<p class="muted">逐层表相对 rt=0，正数才是节省，均未计额外 planner。理想隐藏一列同时免去双方全部权重开销，梯度仍全计，也未假定可被未知模型计算隐藏。</p>
<details><summary>带宽敏感性：估计不是物理八卡结论</summary>'''+table(['场景','发送/返回/权重/梯度 GB/s','对 rt0 串行节省/ms','对 rt0 理想隐藏/ms','对 rt2 串行节省/ms','对 rt2 理想隐藏/ms'],sensitivity)+'''<p>不同通信类型可以有不同效率。统一下降只拉长所有阶段；单独下降 token 或梯度速度会改变净收益。正值只表示该假设下有剩余预算。固定权重与梯度带宽、发送和返回使用相同速度时，相对 rt=0 的串行回本门槛约为 token 低于 99.50 GB/s；相对 rt=2 约为低于 142.96 GB/s，且还未收取 planner 成本。</p></details></section>
<section id="whatif"><h2>自己调整带宽和 planner 成本</h2><p>以下实时计算仍使用原八卡逐层矩阵。带宽为每 GPU 单向有效 GB/s；planner 输入是优化方案相对所选 MoonEP 的<strong>每层额外时间</strong>。</p><div class="controls">
<label>对比基线<select id="base"><option value="native_rt0">MoonEP rt=0</option><option value="native_rt2">MoonEP rt=2</option></select></label>
<label>权重处理<select id="schedule"><option value="serial">全部串行</option><option value="ideal">理想全部隐藏权重</option><option value="shared">共享端点服务近似</option></select></label>
<label>每层权重刷新<select id="refresh"><option value="1">1 次：保留到反向</option><option value="2">2 次：反向重拷</option></select></label>
<label>token 发送<input id="dispatch" type="number" min="1" value="156.322644"></label><label>结果返回<input id="combine" type="number" min="1" value="182.010023"></label><label>权重复制<input id="weight" type="number" min="1" value="330.314639"></label><label>梯度归并<input id="gradient" type="number" min="1" value="278.171125"></label><label>额外 planner μs/层<input id="planner" type="number" min="0" value="0"></label></div>
<p id="interactive" class="result"></p><p class="muted">共享端点服务近似把同一卡的权重服务时间和前向 token 发送服务时间相加再取最大值。它考虑端点重叠，但不是经过硬件验证的并发模拟器。</p></section>
<section id="kernels"><h2>kernel 测量保留八卡维度</h2>'''+kernels+'''
<p>目前不能宣称所有层都取得完整收益。M=1 的基准估算相对 rt=0 在额外 planner 为零时仍亏；降低扫描时间有价值，但不能独自消除权重和梯度的这笔成本。后续选择专家时应同时约束新增副本量、原属卡发送热点，以及每份副本实际消除多少远程发送。</p></section>
<section><h2>更多微批能否摊薄成本</h2><p>若每次更新做多个微批，并能持续保留每层相同版本、相同位置的权重和累积梯度，可以让权重刷新及梯度归并每次更新只做一次；当前模型的总字节在重复相同流量两次时可能回本。但 MoonEP 的额外槽池跨层复用，不能只把微批数改为 2 就假定副本一直在。需要额外存储、调整执行顺序或明确的版本缓存，映射变化时还须先归并旧梯度。这里不把它计作已实现收益。</p></section>
<section id="evidence"><h2>复现与证据</h2><p>原始参数：N=134144，R=8，K=8，E=352，B=8，d=256，每类保留 2 组，抽样种子 20261026，交换 L=4 轮。八层 CPU 全量重放、必需副本、抽样支持表和最终 owner 与旧结果逐项核对。</p><ul>
<li><a href="traffic.json">原始八卡流量、逐卡计数和输入 SHA256</a></li>
<li><a href="bandwidth_attempt02.json">两卡带宽正式原始记录</a>；<a href="bandwidth_attempt01.json">保留的首次失败</a></li>
<li><a href="'''+html.escape(args.estimate.name)+'''">逐层估算、独立复算和敏感性</a></li>
<li><a href="../../experiments/eight_rank_estimate/model_notes.txt">模型完整定义</a>；<a href="../../experiments/eight_rank_estimate/build_estimate.py">估算代码</a></li>
<li><a href="kernel_inputs/manifest.json">逐轮 CPU 验证输入及来源</a></li>
<li><a href="residency_gpu_validation_attempt01.json">驻留中间表、决策历史和 Graph 独立核验</a></li>
<li><a href="estimate_independent_readback03.json">字节与模型时间独立复算</a></li>
'''+''.join('<li><a href="'+html.escape(p.name)+'">八卡维度 kernel 原始记录：'+html.escape(p.name)+'</a></li>' for p in args.kernels)+'''
<li><a href="../two_gpu_overlap/report.html">保留的两卡派生实验（已修正适用范围）</a></li></ul><p class="muted">未测八张物理卡完整 planner 或完整训练；未测 GEMM、真正的权重更新、DP/TP 通信、重计算与跨节点链路。估算与组件实测分别列出。</p></section></main>
<script id="data" type="application/json">'''+payload+'''</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
function endpoints(a){let tx=a.map(r=>r.reduce((s,x)=>s+x,0));let rx=a[0].map((_,j)=>a.reduce((s,r)=>s+r[j],0));return {tx,rx,max:Math.max(...tx,...rx)}}
function calc(m,r,schedule,refresh=1){const d=endpoints(m.token_remote_source_to_destination),c=endpoints(m.expert_home_to_replica);let td=d.max*14336/r.dispatch/1e6,tc=d.max*14336/r.combine/1e6,tw=c.max*88080384/r.weight/1e6,tg=c.max*176160768/r.gradient/1e6;if(schedule==='ideal')return 2*td+2*tc+tg;if(schedule==='shared'){const tx=c.tx.map((x,i)=>x*88080384/r.weight/1e6+d.tx[i]*14336/r.dispatch/1e6);const rx=c.rx.map((x,i)=>x*88080384/r.weight/1e6+d.rx[i]*14336/r.dispatch/1e6);return refresh*Math.max(...tx,...rx)+(2-refresh)*td+2*tc+tg}return refresh*tw+2*td+2*tc+tg}
function update(){let r={};for(const k of ['dispatch','combine','weight','gradient']){r[k]=Number(document.getElementById(k).value);if(!(r[k]>0))return}let b=document.getElementById('base').value,schedule=document.getElementById('schedule').value,refresh=Number(document.getElementById('refresh').value),p=Number(document.getElementById('planner').value)/1000;if(!(p>=0))return;let before=0,after=0;for(const layer of data.traffic){before+=calc(layer.methods[b],r,schedule,refresh);after+=calc(layer.methods.sample256_g2_bundle,r,schedule,refresh)+p}const save=before-after;document.getElementById('interactive').textContent=`八层：基线 ${before.toFixed(3)} ms，优化 ${after.toFixed(3)} ms；估算${save>=0?'节省':'增加'} ${Math.abs(save).toFixed(3)} ms（每层平均 ${Math.abs(save/8).toFixed(3)} ms）。`}
document.querySelectorAll('input,select').forEach(el=>el.addEventListener('input',update));update();
</script></body></html>'''
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(doc)
    print(json.dumps(dict(output=str(args.output),estimate=str(args.estimate),kernel_files=len(args.kernels),bytes=len(doc.encode()))))

if __name__=='__main__':main()
