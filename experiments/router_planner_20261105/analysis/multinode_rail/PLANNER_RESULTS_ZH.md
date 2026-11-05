# 多机 rail planner 首轮优化结果

后续更新：[计入权重后的训练 planner](WEIGHT_AWARE_RESULTS_ZH.md) 已完成权重刷新、梯度归并与新算法测试。下面的首轮建议未计副本成本，不能直接用于训练启用决策。

2026-11-02。已实现有限副本选择、固定候选执行交换及最终地址重建的 CPU 参考版本。4 台机器 × 8 卡、全组 131,072 token、top-8、352 专家，所有结果保持每卡 **32,768 个计算对**，每卡全部额外专家不超过 **8**。token 仍先从源 rail 跨机，再在目标机通过 NVLink 转发。

推荐将 **度数不超过 4 的交换图、每 pair 最多两个候选、单轮反向等量交换** 作为下一步 kernel 实现候选。它在 fanout 偏斜 2.0 的场景中将 rail 最大/平均降到 **1.766**，最忙 NIC 流量降低 **15.00%**，NVLink 总量降低 **8.74%**。这些是准确的 CPU 重放通信计数，完整 GPU planner 的两倍时延门槛尚未测量。

## 主方法如何运行

1. **保留合法均衡 seed。** home 和 seed 实际使用的专家副本全部保留，初始 owner 来自上一轮保存的 MoonEP 语义重放。源 token 和逻辑 top-k 不变。
2. **按热点和可实现收益选择副本。** 一次线性扫描得到 token 的 rank/node seen 与 multiple 位掩码。优先给能删除最后一个远端节点访问的专家投票，同时计入 NVLink 副本减少。每 pair 最多投给源 GPU 和源机器中一个已访问 GPU。投票按 rail、发送端口和接收端口压力加权。
3. **限制小表规划规模。** 在 rank 小表上选择互惠复制边，rank 度数最多 4，每条边每方向最多新增两个专家；先扣除 seed 必需副本，再使用 B8 剩余额度。新增同节点重复专家受到抑制。最多 64 条无向 rank 边，使用固定候选贪心，无 LP。
4. **每 pair 最多两个执行候选。** 第一个为源 GPU；若无副本，则选源机器内已访问且有副本的 GPU。第二个为另一个已访问且有副本的 GPU。目标还必须在上述交换图的邻边上。每 token 最多提交一个 pair 移动，避免相互破坏 token 去重收益。
5. **反向等量配对。** 提案按有向边和 8 档收益分桶，`a→b` 与 `b→a` 接受相同 pair 数。匹配按热点优先的桶前缀确定，保持每 GPU 计算配额。稳定计数分配桶内序号，不对全部 token 排序。
6. **仅重建一次最终地址。** 重新计算 `(rank,expert)` 数量、物理 expert slots、padding 和 `cu_seqlens`，为每个 pair 分配唯一的目标行。保存 owner、驻留、物理 slot、目标行和输入哈希，供独立审计。

这轮做了一个重要收缩：新目标只能是源 GPU 或 token 已经访问的 GPU。因此对每次提交，**每条有向 RDMA 边及每条有向 NVLink 边的 token payload 都只会减少或保持**。不用通过向冷 rail 增加无益字节改善比值，也不需要为这一主方法做逐 pair 的全局端口准入。metadata、权重传输和实际排队时间不包含在这个非增保证中。

位掩码得到某 rank/node 是否只含一个 pair，单个候选的精确去重收益是 O(1)。仅移走远端 node 中多个 pair 的一个时，RDMA 收益为零；这种移动仍可能减少 NVLink，并为下一轮形成机会。

## 单轮结果

主数据位于 [planner_sparse/summary.json](../../outputs/multinode_rail/planner_sparse/summary.json)。下面均比较同一输入 seed 与 `rail_b8_r1`。

| 场景 | rail 最大/平均 前→后 | 最忙 NIC payload 降低 | RDMA 总量降低 | NVLink 总量降低 | 新增活动副本 |
|---|---:|---:|---:|---:|---:|
| 均衡 fanout 1.5 | 1.000→1.000 | 0.00% | 0.00% | 9.27% | 240 |
| 均衡 fanout 2 | 1.000→1.000 | 0.00% | 0.00% | 9.83% | 240 |
| fanout 偏斜 1.5 | 1.500→1.435 | 5.24% | 1.00% | 9.24% | 240 |
| fanout 偏斜 2.0 | 2.000→1.766 | 15.00% | 3.76% | 8.74% | 248 |
| 本机保留率偏斜 2.0 | 2.000→2.000 | 0.00% | 0.00% | 1.49% | 256 |
| trace 派生 layer00 | 1.023→1.048 | 6.23% | 8.74% | 10.20% | 220 |
| trace 派生 layer07 | 1.024→1.035 | 2.01% | 10.11% | 7.75% | 201 |

![单轮 planner 的通信变化](../../outputs/multinode_rail/planner_comparison.png)

两个 trace 派生场景的 rail 比值略升，但绝对最大 rail 和最忙 NIC 都下降。原因是总量也减少，分母改变；不能只拿 max/mean 判断时延是否改善。它们仍是原始完整 top-k 加合成 32 卡映射，不是实测多机 trace。

本机保留率偏斜场景中，跨机 token 的全部 8 个 pair 同在一个远端节点。移动一个 pair 不能取消这次 RDMA，单轮和两轮都未削减其跨机流量。此类场景需要多 rank 的 token 组交换或更合适的副本集合；本轮没有把单 pair 算法的能力扩大到这种情况。

新增活动副本是全组统计。主方法实际使用的副本多于 seed：合成 seed 为零，两个 trace seed 每卡最多两个，主方法每卡上限八个。该比较是在同一预算上限内，不是等实际副本数量。新副本需在执行前正确就绪；权重/scales 传输、训练更新与梯度归并都不能视为免费。

## 两轮和已有驻留对照

两轮共享同一驻留和交换图，仅多做一轮提案、计数协调与提交。

| 场景 | 单轮→两轮 rail 最大/平均 | 单轮→两轮最忙 NIC 降低 | 单轮→两轮 NVLink 降低 |
|---|---:|---:|---:|
| fanout 偏斜 1.5 | 1.435→1.420 | 5.24%→7.73% | 9.24%→12.39% |
| fanout 偏斜 2.0 | 1.766→1.740 | 15.00%→17.56% | 8.74%→11.63% |

为了控制复杂度，优先继续单轮版本。第二轮的收益较小，却增加全量扫描和跨机同步。已有阶段屏障/chunk 调度参考模型中，1.5 偏斜的单轮/两轮时间分别是 14.317/15.208 ms；layer00 与 layer07 的两轮也比单轮慢。所有数据均保留，没有按场景挑最好轮数汇总成一个方法。

同专家交换 `same_expert` 不创建副本，严格保持 `(expert,rank)` 数量。五个 home-only 合成场景无候选；layer00 的 RDMA 降低 0.18%、NVLink 降低 0.30%；layer07 分别降低 2.19%、0.91%，最忙 NIC 降低 1.76%。另有 `bounded_existing`，在 seed 已有驻留下允许不同专家反向配对，作为同驻留对照。两者均包含在每个场景的结果中。

第一版不限执行图的 35 份结果仍保留在 [planner/summary.json](../../outputs/multinode_rail/planner/summary.json)。最终交换图约束保持了两个主要偏斜场景的单轮收益，同时将主方法计数表从 8,192 个计数压到最多 1,024 个。可以用 `--full-graph` 重放该消融；它不属于推荐主路径。

## 复杂度和 kernel 预算

记 `P=T×K`，本次 `P=1,048,576`、`R=32`、`E=352`、`B=8`、图度数 `D≤4`、收益档数 `Q=8`。

| 部分 | 工作量或规模 | 并行实现方向 |
|---|---|---|
| 副本投票 | O(P)，每 pair 最多两个目标 | 源 token 分片、局部直方图归约 |
| 压缩投票表 | `R×A`，`A≤E+RB`；上界 19,456 项 | 实际 seed 活跃 expert/owner 槽索引 |
| rank 小表选择 | `O(R²(E/R+B)log(E/R+B) + R²log R)` 的 CPU 参考 | 小表 top-2/固定预算贪心；不是 per-token 扫 R² |
| 一轮状态与候选 | O(P)，每 pair 至多两个候选；每 GPU 至多 65,536 次候选检查 | 位运算与查表，按源 GPU 并行 |
| 桶匹配 | O(T+RDQ)，无 token 比较排序 | 本地计数、全局小计数协调、稳定 prefix |
| 地址与 padding | O(P+RE) | 复用 MoonEP 式分组统计和源前缀，一次最终发布 |

最多 64 条无向边对应 `2×64×8=1,024` 个 int32 计数，即每源 GPU **4 KiB**。若直接 AllGather 32 张卡的这份计数，每 rank 收集的计数缓冲为 **128 KiB**；这不是网络线上字节或 collective 延迟的实测。副本投票、seed 统计、最终地址前缀的协调也必须纳入完整 planner，不能只计这一次匹配。

当前主实现由 NumPy 完成，匹配是集中式 CPU 参考。它的全部工作随 token 数近似线性，保留了按源 GPU 分片的结构，但并行 GPU kernel、分布式 prefix 和跨机地址发布尚未实现。CPU 结果记录了副本选择、交换和最终行号重建的分别耗时，不可拿它们推算 GPU μs 或与历史 8 卡 planner 比较。

代码还包含提交后的精确流量重算与断言，作为参考正确性检查；这部分计入 CPU 时间。每轮候选 `2P` 不是整个 planner 的全部工作量，原始状态统计、投票、提交、重建和通信都另外记录/说明。

## 时延与副本成本的边界

事件模型继续使用此前的假设参数，没有测量 GPU 或真实网络。本轮主 1.5/2.0 偏斜场景的模型 MoE 时间由 15.970/20.543 ms 变为 14.317/17.941 ms，但 planner、权重/scales、metadata、训练梯度、HBM/SM 争用等成本仍排除。

均衡场景虽然 NVLink 字节减少约 9%，参考模型时间仍分别回退约 0.80% 和 3.41%；本机保留率偏斜场景回退约 1.64%。逐边 payload 非增不能保证 padding、chunk 重组和排队后的关键路径非增。不能把此版本无条件用作所有层的默认 planner，也不能把模型数值写成物理多机端到端加速。

建议 GPU 原型先验证单轮的两个 fanout 偏斜场景，同时保留零收益和回退场景。使用相同硬件、相同输入、相同副本合同测完整最慢 rank 规划时间，门槛仍为原完整多机 planner 的两倍以内。副本冷启动与已就绪两种情况分别测；只有通信节省覆盖新增规划、未隐藏权重和计算开销，才能启用。

## 正确性检查与复现

可复现测试 [test_planner.py](test_planner.py) 已通过 7 项测试，包含 120 个随机通用交换输入、60 个同专家输入、10 组副本选择、空交换图、零预算，以及 physical slot/行号回归用例。随机性质测试使用的宽副本预算与主 B8 实验分开标注。

最终主计划保存在 [planner_sparse](../../outputs/multinode_rail/planner_sparse)，每个场景保存 5 个方法，共 35 份。[最终主版独立审计](../../outputs/multinode_rail/planner_sparse/audit.json) 35/35 通过，[旧全图版审计](../../outputs/multinode_rail/planner/audit.json) 也为 35/35。审计入口 [audit_planner.py](audit_planner.py) 检查输入哈希、每卡配额、B8 驻留、逻辑专家需求、物理 slot、padding 前缀与执行行唯一性，以及通信变化。

开发审查修正了一个地址布局问题：逻辑 expert ID 顺序与“home slots 在前、extra slots 在后”的物理顺序不同，不能直接以逻辑专家前缀生成执行行。最终版本从 physical slots 的 padding 前缀生成 `cu_seqlens` 和目标行，独立核验每行属于对应 slot 的实际占用区间，并保留明确的回归测试。CPU 行号正确性不等同于真实 RDMA buffer 发布和数值 MoE 正确性。

```bash
python3 analysis/multinode_rail/test_planner.py
python3 analysis/multinode_rail/run_planner.py
python3 analysis/multinode_rail/audit_planner.py --input outputs/multinode_rail/planner_sparse
python3 analysis/multinode_rail/plot_planner.py

# 不限执行图的消融，另存目录
python3 analysis/multinode_rail/run_planner.py --full-graph --output outputs/multinode_rail/planner_full_graph_replay
```

核心入口：[副本选择](rail_support.py)、[固定候选与地址重建](rail_planner.py)、[同专家对照](same_expert.py)、[实验驱动](run_planner.py)。原始 MoonEP、历史报告与场景数据保持原样。
