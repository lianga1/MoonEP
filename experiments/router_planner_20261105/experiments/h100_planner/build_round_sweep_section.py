"""Build an audited HTML fragment for the bundle complexity/round-count addendum.

Does not modify the main report, its template, or benchmark JSON.  All table
values are recomputed from raw layer samples, not copied from rounded prose.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import statistics
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'outputs/h100_planner/round_sweep_20261103_attempt01.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/h100_planner/round_sweep_section.html')
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    assert raw['status'] == 'complete' and not raw['failures']
    rows = sorted(raw['layers'], key=lambda row: row['layer'])
    assert len(rows) == 8 and [row['layer'] for row in rows] == list(range(8))
    assert raw['metadata']['groups'] == 10 and raw['metadata']['graph_repeats'] == 20
    assert raw['metadata']['physical_gpus_used'] == 1 and raw['metadata']['logical_ranks'] == 8
    assert all(row['shape'] == [8, 16768, 8] for row in rows)
    for row in rows:
        assert row['gpu_four_round_prefixes_exact_cpu']
        assert all(row['graph_audit'].values())
        for count in (1, 2, 4):
            audit = row['correctness'][str(count)]
            assert audit['exact_cpu_prefix'] and audit['materialized_unique_provenance_exact']
            if count == 4:
                assert audit['exact_saved_l4']
            quality = row['qualities'][str(count)]
            assert quality['audit_pass'] and quality['all_pairs_executable'] and quality['one_owner_per_input_pair']
            assert quality['loads'] == [134144] * 8
            assert len(audit['gpu_round_stats']) == count
        for field, samples in (('scope_timings_ms', 10), ('graph_timings_ms', 20)):
            for values in row[field].values():
                assert len(values) == samples and all(math.isfinite(v) and v > 0 for v in values)

    base = sum(row['base_remote'] for row in rows)
    remote = {count: sum(row['qualities'][str(count)]['remote'] for row in rows) for count in (0, 1, 2, 4)}
    reductions = {count: 100 * (1 - value / base) for count, value in remote.items()}
    gain = {count: base - value for count, value in remote.items()}
    marginal = [sum(row['cpu_rounds'][step]['communication_reduction'] for row in rows) for step in range(4)]
    assert sum(marginal) == gain[4] and marginal[0] == gain[1] and sum(marginal[:2]) == gain[2]
    assert all(row['cpu_rounds'][step]['quality']['remote'] >= row['cpu_rounds'][step + 1]['quality']['remote'] for row in rows for step in range(3))
    timings = {}
    for field in ('scope_timings_ms', 'graph_timings_ms'):
        timings[field] = {name: statistics.mean(statistics.median(row[field][name]) for row in rows)
                          for name in rows[0][field]}
    graph = timings['graph_timings_ms']
    eager = timings['scope_timings_ms']
    retain = {count: 100 * gain[count] / gain[4] for count in (1, 2, 4)}
    exchange_saving = {count: 100 * (1 - graph[f'bundle8_L{count}'] / graph['bundle8_L4']) for count in (1, 2)}
    full_saving = {count: 100 * (1 - graph[f'bundle8_L{count}_seed_exchange_output'] / graph['bundle8_L4_seed_exchange_output']) for count in (1, 2)}
    measurement_date = dt.datetime.fromtimestamp(raw['metadata']['started_unix'], ZoneInfo('Asia/Shanghai')).date().isoformat()
    gpu_row = raw['metadata']['environment_before'][0]['stdout'].strip().splitlines()[0]
    current_uuid = gpu_row.split(',')[1].strip()
    prior_environment = (ROOT / 'outputs/h100_planner/environment_before_attempt02.txt').read_text()
    previous_uuid = prior_environment.splitlines()[1].split(',')[1].strip()
    assert current_uuid != previous_uuid
    json_href = html.escape(args.input.name, quote=True)

    main_rows = []
    eager_rows = []
    for count in (1, 2, 4):
        main_rows.append(f'<tr><td>{count}</td><td>{remote[count]:,}</td><td>{reductions[count]:.2f}%</td><td>{retain[count]:.1f}%</td><td>{graph[f"bundle8_L{count}"]:.3f}</td><td>{graph[f"bundle8_L{count}_seed_exchange_output"]:.3f}</td></tr>')
        eager_rows.append(f'<tr><td>{count}</td><td>{eager[f"bundle8_L{count}"]:.3f}</td><td>{eager[f"bundle8_L{count}_seed_exchange_output"]:.3f}</td></tr>')
    marginal_rows = ''.join(f'<tr><td>第 {step + 1} 轮</td><td>{value:,}</td><td>{100 * value / base:.2f} 个百分点</td></tr>' for step, value in enumerate(marginal))
    layer_rows = []
    for row in rows:
        values = [f'<td>{row["layer"]}</td>']
        values.extend(f'<td>{row["qualities"][str(count)]["reduction_pct"]:.2f}%</td>' for count in (1, 2, 4))
        values.extend(f'<td>{statistics.median(row["graph_timings_ms"][f"bundle8_L{count}"]):.3f}</td>' for count in (1, 2, 4))
        layer_rows.append('<tr>' + ''.join(values) + '</tr>')

    fragment = f'''<!-- BEGIN bundle-round-sweep: generated from {json_href} -->
<div id="bundle-round-sweep" class="placeholder">
<h3>成组交换：R、K 分别花在哪一步？</h3>
<p>这里讨论“每个 token 由 8 个线程协作”的成组交换。它的组是<strong>同一个 token 的若干专家计算任务</strong>。先固定专家驻留位置，每轮为每个 token 最多选一条搬运提议，再按相反方向、相同任务数量匹配；16 线程版与 8 线程版遵循同一规则。</p>
<p><strong>N</strong> 是一层全部 token 数，本次为 134,144；<strong>K</strong> 是每 token 选中的专家数，本次为 8；<strong>R</strong> 是逻辑卡数，本次为 8；<strong>L</strong> 是交换轮数；<strong>p</strong> 是整数排序扫描数据的趟数。下面另用 <strong>U<sub>l</sub></strong> 表示第 l 轮真正改写执行位置的计算对数量，一个计算对就是“一个 token 交给一个专家算一次”，所以 0≤U<sub>l</sub>≤NK。</p>
<div class="table"><table><thead><tr><th>步骤</th><th>一轮工作量</th><th>这个乘数从哪里来</th></tr></thead><tbody>
<tr><td>1. 建立当前位置和可执行位置的小表</td><td><code>O(NKR)</code></td><td>每个 token 的 K 个专家任务，都要判断它对应 R 张卡中的哪些卡。一个 token 有 8 个专家、8 张卡时，共有 8×8 个“专家任务／卡”检查位置。</td></tr>
<tr><td>2. 选择搬运提议</td><td><code>O(4NR²)</code></td><td>每 token 检查 R² 个方向位置；自身到自身跳过，本次有 56 个有效有向卡对。每方向最多考虑前 1、2、4 个可搬任务和原卡上的全部任务。每个候选复用小表做整数位运算。</td></tr>
<tr><td>3. 按类别整理提议、决定接受数量</td><td><code>O(pN + R²K)</code></td><td>N 条记录按类别排序；计数和接受数量按“方向 × 收益 × 搬几个任务”保存。最后一项的大小范围是 1…K，因此有 K，不是每个 token 再与 K 个 token 比较。</td></tr>
<tr><td>4. 写回被接受的执行位置</td><td><code>O(N + U<sub>l</sub>)</code><br>上界 <code>O(NK)</code></td><td>扫描 N 条提议，跳过未接受的；接受的提议搬几个任务，就改写几个位置。一个 token 最多有 K 个位置，因而最坏上界有 K。</td></tr>
</tbody></table></div>
<p><strong>第一步乘 R，表示检查工作量，不表示把 token 从显存重复读取 R 遍。</strong>当前 kernel 的 8 个线程先各读一个专家和执行位置，再循环检查 8 张卡。每张卡的“哪些任务现在归我”和“哪些任务我能执行”压成一个整数：每一位表示一个专家任务是否在集合中。后续方向比较复用这些整数，使用线程间交换与位运算。这样减少重复读取，但检查逻辑本身仍存在；这也不是缓存命中率或显存事务的实测结论。</p>
<p><strong>第三步为什么不是只有 1、2、4 三种大小？</strong>因为还有“把原卡上的任务全部搬走”这个候选。例如某个 token 有 3 个任务在 A，且 B 都能执行，那么大小 3 的全组可以参加；5、6、7、8 也同理。因此配额表按 1…K 预留类别。本次只需给 28 对卡×8 种大小分别算数量，共 224 项；每项是固定次数的整数计算。它不做全体 token 两两比较，也不搜索“2 个任务与 1+1 个任务”的混合配对。</p>
<p>例如 A→B 有 100 条大小 2、收益 1 的提议，B→A 有 120 条大小 2、收益 1 的提议，直接接受双方各 100 条即可，不需要做 100×120 次比较。排序保证同一类别里按原 token 顺序选取。这里的排序只处理类别编号的 <strong>11 个二进制位</strong>，不对 token 编号逐项比较；当前 p 是固定的小常数，所以是 <code>O(pN)</code>，不写成比较排序的 <code>O(N log N)</code>。</p>
<div class="formula">每轮：O(NKR + 4NR² + pN + R²K + N + U_l)
L 轮最坏上界：O(L · [N(KR + R² + p) + R²K])
若把一次驻留位图准备也计入，再加 O(RE)，E=352 是专家总数。
当前 R=K=8、p 固定：O(LN)；L 也固定时：O(N)。</div>
<p>上述是总工作量，不是把每一步串行执行所需的时间。GPU 可以让许多 token 同时处理，但不能据此把整个式子直接除以 8。当前实现把 R、K 写死为 8，且集合装在单个整数中；不能不改实现就将此式外推到任意大的 R、K。16→8 线程改变执行效率和常数，不改变复杂度阶数。第四步实际只改 U<sub>l</sub> 项，比直接写 NK 更能解释后几轮搬得少时的成本。</p>

<h3>{measurement_date} 补测：只做 1、2 轮，会损失多少收益、节约多少时间？</h3>
<p>在指定 h100-omni 重新启动后，使用单张 H100 80GB、同一八层真实 trace、同一驻留表和同一起始分配，重新比较 L=1、2、4。<strong>这是本次同一张卡上互相比较的一组新测量。</strong>本次物理 GPU UUID 与旧主表不同，所以 L=4 也重测；旧主表和旧原始样本保留，不混用两次测量来计算加速比例。</p>
<div class="table"><table><thead><tr><th>轮数 L</th><th>八层远端份数</th><th>相对起点下降</th><th>保留 L=4 的收益</th><th>Graph：交换 / ms</th><th>Graph：起点＋交换＋输出 / ms</th></tr></thead><tbody>{''.join(main_rows)}</tbody></table></div>
<p>起点 L=0 的八层远端份数是 <strong>{base:,}</strong>；这里只作为通信参考，没有把 L=0 记成一次实测 planner 延迟。“份数”按每个 token 需要发送到几张不同的远端卡计数，同一 token 在同一卡交给多个专家只算一份；它不是网络时间或模型加速百分比。</p>
<p><strong>L=2 保留 L=4 约 {retain[2]:.1f}% 的通信收益，同时少花 {exchange_saving[2]:.1f}% 的交换时间、{full_saving[2]:.1f}% 的起点＋交换＋输出时间。</strong>L=1 保留约 {retain[1]:.1f}% 的收益，交换时间减少 {exchange_saving[1]:.1f}%，起点＋交换＋输出时间减少 {full_saving[1]:.1f}%。这些比例都来自本次 Graph 实测，按未四舍五入的数据计算。</p>
<p>轮数增加后，前一轮改过的执行位置会产生新的可交换机会，同分时优先考虑的组大小和卡对也随轮次轮换。每次实际接受的交换都要求组合收益为正，所以沿同一执行前缀增加轮数不会增加远端份数；但后面的增益不保证在所有数据上递减。本次 trace 确实逐轮减少：</p>
<div class="table"><table><thead><tr><th>新增一轮</th><th>这一轮新减少的远端份数（八层合计）</th><th>相对同一起点的新增下降</th></tr></thead><tbody>{marginal_rows}</tbody></table></div>
<p>时间也不会随 L 严格等比例缩放。每轮都要重新检查 token、整理提议、计算配额；轮数减少会减少这部分工作。起始分配与最终输出只做一次，减少轮数也省不掉它们，因此整条路径的时间下降比例小于交换阶段。Graph 的 L=1、2、4 交换实测为 {graph['bundle8_L1']:.3f}、{graph['bundle8_L2']:.3f}、{graph['bundle8_L4']:.3f} ms，而非直接将 L=4 除以 4 或除以 2。</p>
<details><summary>普通调用延迟与逐层结果</summary>
<div class="table"><table><thead><tr><th>轮数 L</th><th>普通调用：交换 / ms</th><th>普通调用：起点＋交换＋输出 / ms</th></tr></thead><tbody>{''.join(eager_rows)}</tbody></table></div>
<div class="table"><table><thead><tr><th>层</th><th>L1 下降</th><th>L2 下降</th><th>L4 下降</th><th>L1 Graph 交换 / ms</th><th>L2 Graph 交换 / ms</th><th>L4 Graph 交换 / ms</th></tr></thead><tbody>{''.join(layer_rows)}</tbody></table></div>
</details>
<p>CUDA Graph 是把固定的一串 GPU 操作预先组织起来，随后整串重放，减少逐次发起的开销；普通调用则逐次发起这些操作。</p>
<p><strong>计时口径：</strong>每层每个方法保留 10 次普通调用与 20 次 CUDA Graph 样本，每轮改变方法执行顺序，执行预热后计时；先取各层中位数，再对八层等权平均。通信下降按八层份数相加后计算。“交换”包含恢复起始执行位置和 L 轮交换；“起点＋交换＋输出”包含 MoonEP 语义移植版生成起点、恢复起点、L 轮交换和最终执行地址等输出，整条路径直接计时。它不是原包原生 MoonEP 的完整延迟。</p>
<p><strong>排除项：</strong>两列时间均不含专家驻留选择、驻留位图打包、输入统计、编译、内存分配、正确性检查、Graph 捕获，也不含真实 token／专家权重通信或专家计算。因此目前只能据此选择规划时间与通信份数之间的取舍，不能直接宣布 L=1、2 或 4 哪个端到端更快。</p>
<p><strong>核验：</strong>8 层×L=1、2、4 共 24 组结果与 CPU 对应前缀逐元素一致，L=4 与原保存结果一致；逐轮搬动数与预测收益一致，负载守恒、专家驻留合法、输出地址唯一且来源可恢复，Graph 计时前后结果均通过核验。另逐轮检查了 GPU 四轮过程的所有前缀。失败记录为空，源码与输入文件哈希保留在原始 JSON。</p>
<details><summary>设备身份与证据</summary><p>本次设备：<code>{html.escape(current_uuid)}</code>；旧主表设备：<code>{html.escape(previous_uuid)}</code>。型号均为 H100 80GB HBM3，本次 PyTorch {html.escape(raw['metadata']['torch'])}，CUDA {html.escape(raw['metadata']['cuda'])}。设备空闲状态、频率和功耗读数见原始 JSON；不把未锁频环境下的细小差别解释成算法变化。</p></details>
<p><a href="{json_href}">本次八层原始样本、逐轮收益与审计</a> · <a href="../../experiments/h100_planner/round_sweep.py">复现实验脚本</a> · <a href="../../experiments/h100_planner/bundle_lane8.cu">成组交换 kernel</a></p>
</div>
<!-- END bundle-round-sweep -->
'''
    assert '<html' not in fragment.lower() and '<body' not in fragment.lower()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    print(f'Built {args.output} from {args.input}; 8 layers, 24 prefix comparisons, no failures')


if __name__ == '__main__':
    main()
