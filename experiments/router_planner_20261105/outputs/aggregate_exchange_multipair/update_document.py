from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
r=json.loads((root/'outputs/aggregate_exchange_multipair/results.json').read_text())['rows']
old=json.loads((root/'outputs/rt2_b8_comparison/results.json').read_text())['rows']
table=''.join(f"<tr><td>{a['layer']}</td><td>{a['variants']['mixed_sizes']['history'][0]['unique_states']:,}</td><td>{b['methods']['sample256_g2_bundle']['reduction_pct_vs_rt0']:.2f}%</td><td>{a['variants']['equal_sizes']['reduction_pct']:.2f}%</td><td>{a['variants']['mixed_sizes']['reduction_pct']:.2f}%</td></tr>" for a,b in zip(r,old))
section='''<section id="aggregate"><h2>10 / 新实验：先合并同类 token，再按数量成批交换</h2>
<p class="lead">这是根据追问新增实现的 CPU 仿真，不是前面原有成组交换的另一种叫法。原有方法逐 token 产生一个提议；新方法对相同完整状态只评估代表，再按这一类有多少 token 分配交换数量。两种方法的选择规则不同，结果不要求相同。</p>
<h3>为什么只记“{e1,e2} 在 A 有多少 token”还不够？</h3>
<p>u、v 的 {e1,e2} 都在 A，也都来自 C。u 已经需要访问 B，v 尚不需要访问 B：同样把完整组从 A 移到 B，u 少发一份，v 只是把远端 A 换成 B，收益为零。如果 token 来自 A 或 B，收益还可能再次变化。只合并专家组会丢掉这些区别。</p>
<p>此外，一个 token 可能同时属于 A 上的组和 B 上的组。如果把两份组计数完全独立使用，就可能在同轮让同一个 token 做两个互相影响的交换，原先算好的收益不再可靠。</p>
<p>本次采用保守、可核验的合并条件：<strong>来源卡相同、完整专家集合相同、每个专家当前执行卡相同。</strong>把专家编号排序后比较，所以不因同一组的列顺序不同而拆开。我们称一类这样的 token 为“相同状态类”。它包含足够信息来算准确收益，也能维护一类还有多少 token 未使用。这个条件不是最粗的可合并条件，后续还有继续压缩的空间。</p>
<h3>新算法具体怎样运行？</h3><ol>
<li>每轮开始，把全量 token 按完整状态合并，记录代表状态、数量，以及原始 token 属于哪一类。</li>
<li>遍历卡对，例如 A/B。每一类只检查一次：A 上的完整组能否移到 B，B 上的完整组能否移到 A。目标必须已有专家，收益不能为负；同一类在这个卡对只选择一个方向，优先正收益，再优先较小组，其他平局按固定规则处理。</li>
<li>按方向、完整组大小、收益汇总数量。不同专家组只要这些条件相同，在配平任务数时就可以合并计数，实际提交时仍保留各自专家信息。</li>
<li>选择两个方向各移动多少 token，使<strong>计算对总数</strong>相等。第一种对照只匹配相同组大小；第二种允许不同大小凑数。</li>
<li>将接受数量分给不同状态类。同一类有 100 个 token，可以先拿 30 个参与 A/B，剩余 70 个稍后参与其他卡对，并非整类必须一次搬完。每个 token 每轮至多移动一个完整组；本轮没有接受的才继续参与后续卡对。</li>
<li>所有数量决定好，再展开成原始 token 的具体分配。下一轮根据新分配重新合并，避免沿用已经分裂的旧状态。</li></ol>
<h3>不同大小怎么凑？</h3><pre>A → B：u 的 {e1,e2,e3}，1 个 token × 3 个任务，收益 1
B → A：v 的 {f1}，      1 个 token × 1 个任务，收益 0
B → A：w 的 {g1,g2}，   1 个 token × 2 个任务，收益 0

正向 3 个任务 = 反向 1 + 2 个任务
A 和 B 的计算量都不变；合计少发 1 份远端 token 数据。
两边的 token 数不必相同，必须相同的是计算对数。</pre>
<p>这正是“找多个专家组来凑”的情况，本次已作为检查用例执行。对这种混合大小匹配，仅取两个名单数量的最小值已经不够。</p>
<p>新实现用一张整数任务量表求配平：对每个方向，计算“恰好移动 w 个计算对，最多能省几份通信”，w 从 0 到两个方向可供任务量的较小值 W。表中有每种组大小 1～K、收益 0/1 的数量上限。然后在两边都能实现的 w 中挑总收益最高者，再反推出各类该取几个 token。这叫动态规划，也就是复用已算好的较小问题答案，避免把所有数量组合重新枚举。代码使用滑动范围内保留最好候选的队列，将每种大小的表更新压到 O(W)。</p>
<p>这里的最优只针对<strong>当前卡对、已经产生的完整组候选和剩余数量</strong>。卡对按顺序处理，前面用掉的 token 后面不能再用；每轮会轮换卡对起点。它不是所有卡、所有任务的全局最优算法。</p>
<h3>新方案的复杂度</h3><p>G 表示本轮不同完整状态的数量，可能远小于 T，也可能几乎等于 T。W<sub>ab</sub> 表示卡对 a/b 可匹配的计算对数上界。它是任务数量，不是专家组类型数量。</p>
<div class="formula">一轮，按当前实现计：
构造与合并状态：O(T×K×log T) 的保守比较排序上界
按代表检查卡对、汇总和分发数量：O(G×R²×K)
仅同大小配平：O(R²×K)
不同大小混合配平：O( K × Σ卡对 W_ab )
把数量展开成具体 token 分配：O(T×K + G×R²)
四轮就对各轮成本求和，因为 G 和 W 会变化。</div>
<p>这里合并使用 NumPy 的精确去重排序；专家集合的列排序只在开始时做一次。如果以后换成完整键校验的哈希表，合并可以争取平均 O(TK)，但那不是当前已经测过的 GPU 实现。内存主要是 O(TK + GR² + K×max W_ab)：原始数组、每状态的方向数量，以及整数配平的回溯表。</p>
<p class="note">所以不能宣称“整体从 O(T) 降成 O(G)”。减少的是重复的方向评估；读输入、重新分组、展开结果仍需碰全量 token。混合大小配平还引入了依赖 W 的成本。本次 W 的最大值为 15,660；并非只扫描几种专家组就能完成所有精确凑数。</p>
<h3>真实 trace 的仿真结果</h3>
<p>固定前面的 256/g2、种子 20261026 选出的同一驻留表，起点都是原生 rt0，每卡最多 8 个额外专家，均执行四轮；没有为新算法重新挑更有利的专家副本。通信降幅仍相对 rt0。</p>
<table><tr><th>方案</th><th>八层远端份数</th><th>下降</th></tr>
<tr><td>现有逐 token 单计算对交换</td><td>3,825,459</td><td>24.31%</td></tr>
<tr><td>现有逐 token 成组交换</td><td>3,791,665</td><td>24.98%</td></tr>
<tr><td>聚合同类，分批参与卡对，只配同大小</td><td>3,803,904</td><td>24.74%</td></tr>
<tr><td><strong>聚合同类，分批参与卡对，不同大小凑数</strong></td><td><strong>3,708,225</strong></td><td><strong>26.63%</strong></td></tr></table>
<p>新混合版比旧成组版多下降约 1.65 个百分点，剩余通信再少 2.20%；并非每一层都更好。它同时改变了代表状态选择、卡对调度和组大小匹配，不能把差异全部归因于“压缩计数”本身。</p>
<table><tr><th>层</th><th>首轮状态类 G</th><th>旧成组</th><th>聚合，同大小</th><th>聚合，混合大小</th></tr>__ROWS__</table>
<p>每层 T=134,144。第 0 层 G=126,984，只合并掉约 5.34%；第 7 层 G=15,226，合并掉约 88.65%。整个新混合版四轮、八层共 1,628,333 个状态代表，对应原来 4,292,608 次 token 轮次。但每个代表仍会检查多个卡对，不能把这个比例直接叫作加速比例。</p>
<p>还保留了一个较弱的初版：每类每轮只提议一个方向，同大小下降 18.49%，混合大小下降 20.31%。这说明过早把整类锁死在一个方向会损失机会；保留失败/较弱版本，避免只展示最后的较好数字。</p>
<p>32 份保存方案（两种聚合调度×两种匹配×八层）已独立检查：专家可执行、副本上限、每卡 134,144 个计算对、通信重新计数和预计收益一致。混合匹配的整数表还对 40 个小规模随机计数做了穷举对照。本次没有 GPU 实测，没有生成最终 dispatch 地址，也没有权重复制或训练计时。</p>
<p><a href="../../analysis/vendor/aggregate_exchange.py">新算法源码</a> · <a href="../../analysis/replay_aggregate_multipair.py">主实验入口</a> · <a href="../aggregate_exchange_multipair/results.json">逐层、逐轮结果</a> · <a href="../aggregate_exchange_multipair/independent_audit.json">独立核验与汇总</a> · <a href="../aggregate_exchange/results.json">较弱初版结果</a></p></section>'''.replace('__ROWS__',table)
p=root/'outputs/planner_explainer/template.html';s=p.read_text();s=s.replace('<section id="aggregate">', '<section id="aggregate">',1)
# Script intended once; preserve earlier chapters and insert before sources.
if '<section id="aggregate">' in s:
 start=s.index('<section id="aggregate">');end=s.index('<section id="sources">',start);s=s[:start]+s[end:]
s=s.replace('<section id="sources">',section+'\n<section id="sources">')
s=s.replace('<a href="#evidence">实测与边界</a>','<a href="#evidence">实测与边界</a><a href="#aggregate">新增：批量组交换</a>')
s=s.replace('09 / 对应代码与结果','11 / 对应代码与结果')
p.write_text(s)
