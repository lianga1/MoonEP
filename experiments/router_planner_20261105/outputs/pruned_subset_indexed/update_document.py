from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[2];out=root/'outputs/pruned_subset_indexed';q=json.loads((out/'summary.json').read_text())
section='''<section id="subset"><h2>10 / 新实验：用共享专家子集替代完整状态类</h2>
<p class="lead">这版不再把全体 token 按完整 8 专家状态排序合并，也不逐卡对遍历所有完整状态类。它先挑少量“专家子集＋移动方向”，为每个子集建立包含它的 token 名单，交换时只检查这些名单。它仍会读取名单中的 token，不能称为“只做一百多次计算”。</p>
<h3>先把两个限制分开</h3><p>h 是一组最多包含几个专家，本次测试 h=1 或 2；g 是每个“方向、子集大小、初始收益”类别保留几个最常见的候选。h=2 时同时考虑单专家和双专家子集。比如 {e1,e2} 就是一种共享子集，不要求 token 另外六个专家相同。g=0 在代码中表示不按候选数量截断，不表示零个候选。</p>
<h3>从全量扫描变成名单查询</h3><ol>
<li>固定同一份 256/g2 选出的驻留表和原生 rt0 起点，不增加专家副本。</li>
<li>只在开始时完整统计：对于每个 token，枚举一个或两个在同一旧卡上的专家。仅保留目标卡能执行、初始收益非负的方向，按方向/大小/收益统计频次，留下前 g 个模板。这里使用全量初始路由统计，没有再次随机抽样。</li>
<li>建立“专家子集 → 包含它的 token 行号”索引。这就是一份查名单的表。只为入选子集保存名单：两次线性扫描先计数再填写，无需把所有子集记录排序。原型仍检查全量单/双专家组合，但不保存未入选项。</li>
<li>每轮按卡对读入选模板的名单。重新检查这两个专家当前是否都在旧卡、token 是否已使用过、是否真的能少访问一个远端卡。负收益丢弃。一个 token 若同时匹配多个模板，在当前卡对只留一个提议。</li>
<li>按方向、大小、真实收益汇总数量，用上一节的整数配平法选择双方相等的计算对总量。允许正向两个任务，反向两个单任务凑齐。</li>
<li>提交时标记已使用的 token，使它本轮不能再次参与其他卡对。名单可复用四轮，因为 token 选择的专家不变；执行卡和通信收益每轮、每次提议仍要重新检查。</li></ol>
<p>例如所有含 {e1,e2} 的 token 都能进入同一份静态名单。但有的 token 已不在 A 执行这两个专家，有的仍留着其他任务在 A，有的已经需要访问 B。它们在这次 A→B 移动中的收益不同，因此还要按当前状态检查。<strong>仅凭名单长度就把全部 token 搬走，会算错收益。</strong></p>
<p>入选方向模板在四轮内固定。初始不合法或不够热门、但后来变得有价值的模板可能被漏掉；这里的“全部模板”也只是全量保留初始有效的单/双专家模板，不等于无约束搜索所有可能交换。</p>
<h3>复杂度具体减少在哪里？</h3><p>定义 F 为所有非空候选统计类别数；C 为入选的方向子集模板数；Z<sub>i</sub> 为第 i 轮读取的名单条目总数。一个 token 可以出现在多个子集或方向中，因此 Z 可以比 T 大。T、K、R、E 的含义沿用前文。</p>
<div class="formula">本次 h≤2：
初始统计：O(T×K²×R) 的保守上界
筛选模板：O(F log F)，保留 C ≤ 2×h×R×(R−1)×g
构建选中子集索引：O(T×K² + E²)，保存入选名单而非全部组合
第 i 轮：O(K×Z_i + 2×Σ卡对 W_ab + T×K)
四轮总计 = 一次准备 + 四轮上述成本。</div>
<p>统计表用整数键；字典查询按平均常数时间估计。h=1 时枚举项可以从 K² 降到 K。E² 来自当前实现给单/双专家子集编码的密集索引表，专家数很大时可改稀疏表，但这不是本次已测实现。W_ab 是两方向可配平的任务量上限，h=2 时只有大小 1、2，因此整数表类型数是常数。</p>
<p>它消除了上一版每轮 O(TK log T) 的完整状态排序，以及 O(GR²K) 的全状态类卡对扫描，替换为入选子集的名单查询。不过固定候选数并不能让执行成本与 T 无关：热门子集可能命中很多 token，初始统计也仍然是全量。</p>
<p>当前 h2/g8 每层只有 118～142 个方向模板；平均保存约 602,243 条子集—token 关系，四轮累计读取约 10,781,235 条名单记录。很多记录会因 token 本轮已使用而快速跳过，但仍有访问开销。这正是“一百多个模板”不等于“一百多次工作”的原因。</p>
<h3>同口径真实结果与 CPU 延迟</h3><p>8 层完整 trace、相同原生 rt0 起点、相同专家驻留、每卡最多 8 个额外专家、四轮。所有通信降幅相对 rt0。每层三次交错顺序执行，取每层中位数，再对八层求平均；JIT（首次把程序编译成机器指令）提前完成，不计磁盘读写。延迟包含本算法的候选统计、筛选、索引构建和四轮交换；各方法共同的原生起点生成、驻留选择、最终通信地址生成不计入。</p>
<table><tr><th>方案</th><th>通信下降</th><th>本机 CPU 优化组件延迟</th></tr>
<tr><td>原有逐 token 成组</td><td>24.98%</td><td>110.87 ms</td></tr>
<tr><td>上一版完整状态聚合、混合大小</td><td>26.63%</td><td>672.03 ms</td></tr>
<tr><td>共享单专家 h1/g8</td><td>25.53%</td><td>116.20 ms</td></tr>
<tr><td>共享单/双专家 h2/g2</td><td>25.85%</td><td>128.62 ms</td></tr>
<tr><td><strong>共享单/双专家 h2/g8</strong></td><td><strong>26.82%</strong></td><td><strong>134.82 ms</strong></td></tr>
<tr><td>共享单/双专家，不截断初始模板</td><td>26.82%</td><td>135.43 ms</td></tr></table>
<p>h2/g8 相比完整状态聚合约快 5 倍，同时通信降幅略有增加；但比原有逐 token 成组仍慢约 22%。这不是 GPU 延迟，不可与此前数百微秒的 H200 kernel 数据混用。只有三次本地 CPU 配对轮次，不作统计显著性或跨机器保证。</p>
<p class="note">这份 trace 上 g=8、g=32 与不截断的模板集合已经相同，输出也逐项相同。所以不能把约五倍的 CPU 改善归功于 g=8 剪枝。主要变化是不用完整状态排序聚合、只考虑 1/2 专家子集、直接查名单。真正收紧到 g=2，候选为 103～119 个，延迟约降 4.6%，但通信降幅损失约 0.97 个百分点。</p>
<p>40 份最终方案全部通过独立检查：固定驻留不变、每卡副本不超过 8、每卡计算对精确 134,144、专家执行合法、去重通信重算和预测收益一致。新过滤索引与保留全部子集索引的初版，在所有方案上输出逐项一致。还检查了相同 {e1,e2} 在不同 token 上具有不同收益的构造例子。</p>
<p><a href="../../analysis/vendor/pruned_subset_exchange.py">共享子集算法</a> · <a href="../../analysis/replay_pruned_subsets_indexed.py">仿真入口</a> · <a href="../../analysis/benchmark_pruned_cpu.py">配对计时入口</a> · <a href="../pruned_subset_indexed/summary.json">结果与核验</a> · <a href="../pruned_subset_indexed/cpu_benchmark.json">逐次 CPU 延迟及环境</a></p>
<p>这是共享子集方向的第一版可核验原型，不是最终最优实现。若要继续追求 GPU 延迟，应测“建立索引＋模板匹配＋计划输出”的完整 GPU 路径，并考虑按更细收益类别维护计数，减少长名单的重复检查。本次没有新增 GPU 测量。</p></section>'''
p=root/'outputs/planner_explainer/template.html';s=p.read_text();s=s.replace('<section id="sources">',section+'<section id="sources">');s=s.replace('10 / 对应代码与结果','11 / 对应代码与结果');s=s.replace('<a href="#aggregate">新增：批量组交换</a>','<a href="#aggregate">批量组交换</a><a href="#subset">新增：共享子集剪枝</a>');p.write_text(s)
files=['analysis/vendor/pruned_subset_exchange.py','analysis/replay_pruned_subsets_indexed.py','analysis/benchmark_pruned_cpu.py','analysis/audit_pruned_subsets.py']
(out/'provenance.json').write_text(json.dumps({'sources':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},'gpu_benchmark':False,'support':'same saved 256/g2 seed20261026','scope':'static initial subset shortlist; h<=2; four rounds; no model execution'},indent=2))
