# 八卡 planner 阶段诊断与精确缓存优化

> **验收：八层基准、17组精确参考核对、八卡接口与权重/梯度回归均通过。** planner 加速不等于完整训练步加速。

同一台 `omni-dev` 的八张 H200、相同 router trace 和配置下，八层完整 sampled planner 平均从 **1,214.140 μs 降到 815.758 μs，耗时减少 32.81%，加速 1.488×**。主要改动是复用专家组的检查结果，并只更新上一轮真正新增专家的目标卡。没有减少样本、交换轮数或候选方案。

最终代码为 `sampled_residency.cu` SHA256 `c778fcfd411cb2a82d7238c6075f1cf59ba7e0f017184a9270747158c9471c47`；基于 `81a5a31`，本提交包含对应代码与原始证据。

## 为什么之前是 170 μs，现在是 1.2 ms

| 口径 | 之前的两卡测试 | `81a5a31` 八卡历史结果 |
|---|---:|---:|
| 物理卡 / 逻辑卡 | 2 / 2；原八个来源合并为两组 | 8 / 8；保留原八个来源 |
| 交换轮数 | 1 | 4 |
| 原生 planner | 203.496 μs | 67.709 μs |
| 完整新 planner | 372.686 μs | 1,208.038 μs |
| 相对原生增加 | **169.190 μs** | 1,140.329 μs |

之前的“约 170 μs”是**新增开销**，不是完整 planner。卡数、候选卡对数量、每卡 token 数及交换轮数也不同。两个时期的软件环境与采样协议不同，不能拿两栏相减推算卡间通信开销。当前分项测试使用同一八卡环境重新测量。

本次输入为 `N=134144，R=8，S=16768，E=352，K=8，B=8`；全局抽样 `256` 个 token，每方向/大小/收益类别保留 `2` 组，交换 `4` 轮，种子 `20261026`。环境为 Torch `2.13.0+cu130`、CUDA `13.0`、NCCL `2.29.7`。拓扑与设备状态见 [topology_01.txt](../experiments/planner_stage_profile/evidence/topology_01.txt)、[gpu_environment_01.csv](../experiments/planner_stage_profile/evidence/gpu_environment_01.csv)。

完整计时包含原生起点、采样汇总、副本选择、四轮交换、物理地址生成、全局校验与发布同步。采用 warmed CUDA Graph，每卡 10 次样本、每次 5 次重放；先取每卡中位数，再取最慢卡，最后对所列层求平均。**不含** JIT、输入直方图、专家权重与梯度传输、token dispatch/combine、GEMM 或优化器。

| 本次相同条件的比较 | 修改前 | 最终候选 c2 | 说明 |
|---|---:|---:|---|
| 完整 planner，第 0、7 层平均 | 1,178.565 μs | **814.450 μs** | 无阶段打点，主要延迟口径 |
| 原生 planner，同次测量 | 73.024 μs | 73.237 μs | 起点基本不变 |
| **完整 planner，八层平均** | **1,214.140 μs** | **815.758 μs** | 最终全 trace 比较；减少 32.81% |
| 原生 planner，八层平均 | 67.621 μs | 68.312 μs | 同次基准 |

原始记录：[代表层基线](../experiments/planner_stage_profile/evidence/baseline_01.json)、[代表层 c2](../experiments/planner_stage_profile/evidence/optimized_02.json)、[八层基线](../experiments/planner_stage_profile/evidence/full_baseline_01.json)、[八层 c2](../experiments/planner_stage_profile/evidence/full_optimized_02.json)。

## 瓶颈在副本选择，不是全量 token 扫描

下表是第 0、7 层全部卡、全部采样的**阶段事件均值**。各行使用相同的记录集合；阶段事件会增加开销，因此不能用这张表替代上面的无打点完整耗时。表仅摘录主要部分，完整 36 阶段保留在 JSON。

| 阶段 | 修改前 μs | c2 μs | 代码位置 |
|---|---:|---:|---|
| 原生 MoonEP 起点 | 69.430 | 68.824 | `planning.py`；内部也有卡间同步 |
| 专家副本选择，含全部内部 kernel | **796.568** | **443.534** | `sampled_residency.cu` |
| 四轮全量扫描并提出交换 | 57.697 | 57.764 | `sampled_exchange.cu` |
| 四轮分配名额与应用交换 | 58.041 | 57.851 | `sampled_exchange.cu` |
| 最终任务排序与本卡计数 | 46.462 | 46.361 | `sampled_materialize.cu` |
| 最终物理执行地址生成 | 17.512 | 17.483 | `sampled_materialize.cu` |
| 七次 NCCL 调用所在阶段合计 | 151.051 | 140.754 | 包括通信、等待及阶段事件 |

额外的 Torch profiler 将副本选择进一步定位到贪心 kernel：`global_greedy` 的平均每次重放 kernel 时长为 **719.624 μs**，c2 的 `global_greedy_table` 为 **366.656 μs**，下降约 49.05%。精确频数统计 `record_frequencies` 约为 **35.376 / 35.375 μs**，基本不变。eager trace 只用于代码归属，耗时来自 warmed Graph trace；两版各覆盖 16 个 rank/layer capture，均无未归属 GPU 事件。

七个 NCCL kernel 的 profiler 时长之和为 **118.713 / 106.943 μs**。它与阶段事件表不是同一种口径；两者都包含等其他卡的时间，不能标成“纯数据传输时间”。没有修改通信实现；差异可能来自跨卡等待或采样波动，未证明纯传输更快。原生起点和元数据发布还包含直接 NVLink 操作。profiler 会影响启动节奏，不能把其首尾跨度当作正式完整延迟。

证据：[baseline_trace_analysis_01.json.gz](../experiments/planner_stage_profile/evidence/baseline_trace_analysis_01.json.gz)、[optimized_trace_analysis_02.json.gz](../experiments/planner_stage_profile/evidence/optimized_trace_analysis_02.json.gz)。仓库保留压缩分析及原始压缩 trace，可由分析脚本重建；kernel 总时长、区间并集与首尾跨度分开报告，没有合并不同卡的时钟。

| 卡间统计操作 | 次数 | 每卡输入 → 输出逻辑字节 |
|---|---:|---:|
| 样本 SUM | 1 | 18,436 → 18,436 |
| 交换计数 AllGather | 4 | 每次 4,608 → 36,864 |
| 最终任务计数 AllGather | 1 | 11,264 → 90,112 |
| 校验状态 MAX | 1 | 4 → 4 |

这些是张量大小，不是实测 NVLink 链路字节。本次没有修改通信次数与通信协议，也没有新增专家副本来换时间。

## 修改方式与未采用的尝试

原贪心会反复评估每个双向候选方案。同一个方向的专家组可出现在多个方案中，于是“还缺几个专家、补齐后有哪些组可执行”被重复计算。

c2 把这部分结果按方向组放入共享内存。第一次建立全表；每次接受方案后，只标记**确实得到新专家的目标卡**，下一轮重算这些卡对应的表，其他结果直接复用。双向方案通过查表评分。每轮仍重新检查全部方案的收益和两端副本预算，保留原有候选顺序、平分规则及完整决策历史，不使用过期的方案分数。

默认配置保留单个 block、256 线程；两个方向的候选索引预先读取，循环中复用。最终构建的 `cuobjdump` 资源读取结果为 **149 个寄存器/线程、33,840 字节静态共享内存、STACK=0、LOCAL=0**。资源证据见 [optimized_resources_02.txt](../experiments/planner_stage_profile/evidence/optimized_resources_02.txt)。 表结构超过 48 KiB 的配置仍走原通用实现，不能宣称所有 E/K/G 配置都有同样加速。

| 尝试 | 第 0、7 层完整 planner | 处理 |
|---|---:|---|
| c1：每轮重建整个方向组表 | 1,297.026 μs | 比基线慢 10.05%，未采用；回退的硬件原因尚未由专门实验确认 |
| **c2：只更新发生变化的目标卡** | **814.450 μs** | 最终选择 |
| c3：在 c2 上改为 512 线程 | 809.653 μs | 仅再降低 0.59%，不足以支持更换默认值，仍使用 256 线程 |

保留记录：[optimized_01.json](../experiments/planner_stage_profile/evidence/optimized_01.json)、[optimized_03.json](../experiments/planner_stage_profile/evidence/optimized_03.json)。没有只保留最快的尝试。

## 正确性、后续机会与结论边界

第 0、7 层的**全部保存输出摘要逐项一致**，包括最终执行地址、owner、专家支持表、有效候选包、完整有效 history、校验状态和发布元数据。八层完整基准的审计记录也逐项一致：owner SHA、负载、物理布局检查、通信矩阵、副本数量与回退状态均未改变，无回退层。

八层总的去重远程 `(token,目的卡)` 数，原生为 **5,054,165**，sampled 为 **3,791,665**，减少 **24.9794%**；额外专家副本计数跨层合计为 **71 / 507**。这些数量在优化代码前后完全相同：本次节省来自 planner 执行，不是改变通信方案。实际权重字节、梯度归并与传输时间没有在此基准中测量。

- [17组输入精确核对](../experiments/planner_stage_profile/evidence/selector_exact_02.json)全部通过：原8层、B44的165步选择、可变E/K/D/G、超过48KiB的通用路径、Graph更换输入与污染工作区、强制哈希碰撞、无效副本拒绝及恢复。逐项比较完整history原始行，含候选编号、成本及专家位图。
- [八卡集成回归](../experiments/planner_stage_profile/evidence/integration_02.log)每个rank均6项通过，包含公开dispatch/combine、保存计划的反向流程、权重预取及副本参数梯度归并；[配置回归](../experiments/planner_stage_profile/evidence/host_config_02.log)13项通过。它们不构成完整模型训练的吞吐验证。

仍可评估小 kernel 融合或小统计表的直接 NVLink 汇总，但需要新的实测与一致性检查。各轮名额依赖全部卡计数，下轮又依赖本轮更新，不能简单合并四轮通信。副本表确定后启动权重预取是另一个集成机会；本次没有测量实际权重传输或完整训练步，**不声称端到端训练已经提速，也不把 planner 节省的时间直接等同训练收益**。

复现入口为 `tests/profile_sampled_stages.py`（36 阶段及全 rank trace）、`tests/benchmark_sampled_group.py`（完整基准），离线使用 `experiments/planner_stage_profile/analyze_traces.py` 与 `build_report.py`。所有正式延迟均在 profiler 之外测量。
