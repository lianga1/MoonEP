# 多机 EP 的 rail 仿真与联合规划方案

2026-11-02。目标拓扑为 4 台机器、每台 8 张 GPU、8 个 rail；全 EP 组 128K 按 131,072 token 解释，每源 GPU 4,096 token。沿用现有输入的 top-8、352 个专家，home 映射暂取 `expert // 11`，均衡目标为每 GPU 32,768 个计算对。用户报告的偏斜口径是 **最大 rail 流量 / 平均 rail 流量 = 1.5–2**，并非标准差 / 均值。

最初采用两层设计：慢路径选择有限专家副本与计算配额，快路径优先减少热 rail 的远端节点访问。token 的跨机路径仍为源 GPU RDMA 到目标机同 local rank，然后目标机内 NVLink。场景、离线模型和 CPU planner 已实现；GPU kernel 的完整 2 倍延迟门槛尚未测量。最新训练模型已计入权重刷新与梯度归并，发现大量新增副本不划算，转向节点优先配额，见 [计入权重后的结果](WEIGHT_AWARE_RESULTS_ZH.md)。[首轮 planner 结果](PLANNER_RESULTS_ZH.md) 和下面内容保留为前期架构与候选方案。

## 路径约束与可优化的量

令源位置为 `(s,l)`，最终执行 GPU 为 `(d,j)`。跨机时路径是 `(s,l) → RDMA rail l → (d,l) → NVLink → (d,j)`；`j=l` 时没有后一跳。本机执行可以直接用 NVLink，禁止先在源机把待跨机 token 转移给另一 rail。

这与 [DeepSeek-V3 报告 3.2.2 节](https://arxiv.org/html/2412.19437v2#S3.SS2.SSS2) 描述的跨机后机内转发方向一致；具体协议、chunk 和聚合实现仍需锁定实际部署版本，不能以当前 DeepEP 主分支替代用户的传输合同。

固定 token 来源后，其 RDMA rail 不可选择。调整远端机器内的 expert rank 只改变 NVLink 路径。真正能减少跨机字节的动作是：让 token 少访问一个远端节点，把需要的执行留在源节点，或将多个目的节点上的任务合并到更少的节点。把某节点上的一个 pair 移走，而该 token 仍需访问这个节点，不会节省这份 RDMA payload。

设 `owner[t,k]` 为每个计算对的执行 GPU，`D(t)` 为它访问的远端节点集合，则

```
F[s,l,d] = Σ_{t: source=(s,l)} 1[d ∈ D(t)]
B_dispatch[s,l,d] = F[s,l,d] × token_input_bytes
L[l] = Σ_{s,d≠s} B_dispatch[s,l,d]
```

计算对、token 到执行 GPU 的副本、token 到远端节点的副本是三种不同计数。原 `analysis/moonep_cpu.py` 的通信指标按执行 GPU 去重；在新拓扑中不能直接把它当作 RDMA 流量。

优化目标首先是绝对最忙端口的服务时间和整层关键路径，`max/mean`、CV 和总字节辅助解释。给冷 rail 增加无益流量也能改善比值，所以比值不能单独作为目标。

## 仿真架构

数据流分为六层，所有方案使用相同输入和硬件参数。

| 层 | 输入与输出 | 必须保留的约束 |
|---|---|---|
| 工作负载 | source、完整 top-k、可选权重与到达时刻 | 同一个 token 的共现结构；原始专家 ID 不改 |
| 驻留和 planner | `resident[rank,expert]`、`owner[token,k]`、执行槽 | 副本只增加执行资格；每个计算对只执行一次 |
| 精确流量展开 | `F[4,8,4]`、机内有向流、metadata、prefetch | 远端 node 去重后再按目标 GPU 去重转发 |
| 资源和事件 | NIC Tx/Rx、NVLink Tx/Rx、队列、chunk、计算依赖 | 源 rail 固定；RDMA 完成后才释放后续 NVLink |
| 端到端路径 | planner、权重就绪、dispatch、GEMM、combine | 等待与重叠显式建模，不能把各阶段独立最小值拼接 |
| 检查与报告 | 守恒、驻留、配额、尾延迟、热点和时间线 | 结果、假设、CPU 仿真、GPU 测量分开保存 |

每个 rail 的交换机默认为无阻塞交换机，**每台机器的 NIC Tx 和 Rx 是共享资源**。不能把整个 rail 串行成一根带宽为单 NIC 的链路，也不能让三个 node-pair 各自独享完整 NIC 带宽。若实际存在 oversubscription，再加入 rail 交换机共享容量。NVLink 同样记录 GPU 注入与接收瓶颈，后续校准 NVSwitch、HBM 与通信 SM 的共享限制。

Dispatch DAG 为 `RDMA chunk → 目的机 NVLink fanout → expert-ready → GEMM`。Combine 是反向的 `执行 GPU 局部加权归并 → NVLink 到同 rail relay → node 内归并就绪 → RDMA 返回 source`。返回必须保留 token 身份和 gate 权重；结构仿真没有原始 gate weights，不检验数值误差。

Combine 有两个独立合同。节点内先归并时，每 token 每远端 node 返回一份；只在执行 GPU 归并时，每 token 每执行 GPU 返回一份，RDMA 字节可能更大。输入和输出 dtype 也分别设置，例如 FP8 dispatch 与 BF16 combine 不能简单乘二。训练还需加入反向 activation 通信、专家权重更新与副本梯度归并。

已实现的 [simulate.py](simulate.py) 采用全局 dispatch、compute、combine 阶段屏障，阶段内部 RDMA 与 NVLink 按 chunk 依赖流水；NIC/NVLink Tx 和 Rx 分别共享。整 chunk 的固定次序预留可能引入队头阻塞，属于可审计的保守参考调度，并非生产协议的预测器。当前实现节点内归并的 combine 合同，另一个按执行 rank 返回的合同留作扩展。

默认假设为 hidden=7,168、输入输出均 BF16、RDMA 40 GB/s、NVLink 150 GB/s、chunk=128 token、启动成本分别 2/0.5 μs、expert padding=128、每 GPU 20M pairs/s。planner 默认 0 明确标为未测且排除；权重/梯度、metadata 字节、有限缓冲、HBM/SM 争用、归并算术均未计入。后续校准应使用本机消息大小曲线、同 rail 多源争用、目的机转发、分组 GEMM 大小曲线及完整 dispatch/combine traces。吞吐带宽值不是固定消息延迟模型。

报告至少包含：

- 每 rail 字节、max/mean、CV，以及 32 个 NIC 的 Tx/Rx；`4×8×4` 原始流量张量。
- RDMA 与 NVLink 总量、按 token 的远端节点 fanout、远端 pair 数和执行 GPU fanout。
- 每 GPU 实际及 padding 后计算量、每 expert 分组大小、额外驻留与每次新增权重字节。
- chunk 完成 p50/p95/p99、最慢端口及队列等待、dispatch/combine 尾部；整层用全组 makespan。
- burst 场景的时间窗内热点与队列峰值，不能仅报告全批累计量。少量合成重复的分位数不能叫生产 p99。
- planner 最慢 rank 的 p50/p95/p99、完整层耗时、冷副本与热副本两种成本。

## 已生成的工作负载

入口是 [generate.py](generate.py)，输出位于 [场景清单](../../outputs/multinode_rail/scenarios/manifest.json)。NPZ 保留 `topk[32,4096,8]`、`owner` 和 `home_rank`；派生 trace 另保留原 source rank、原 token 行号和原文件哈希，便于逐行追溯。

| 场景 | rail max/mean | RDMA token 副本数 | 用途 |
|---|---:|---:|---|
| balanced | 1.000 | 196,608 | 主均衡对照 |
| fanout_skew_1p5 | 1.500 | 196,608 | 同总量的中度偏斜 |
| fanout_skew_2p0 | 2.000 | 196,608 | 同总量的高度偏斜 |
| balanced_fanout2 | 1.000 | 262,144 | 总通信量增加的独立对照 |
| locality_skew_2p0 | 2.000 | 65,536 | 由本机保留率造成的偏斜 |
| trace_lift_layer00 | 1.02337 | 354,780 | 原始早层共现保留的派生回归 |
| trace_lift_layer07 | 1.02384 | 340,273 | 原始后层共现保留的派生回归 |

主三场景严格保持每 GPU 32,768 pairs、全局 524,288 个远端 pairs、196,608 个 RDMA token 副本、917,504 个 NVLink token 副本，以及逐专家全局频数。平均远端节点 fanout 都是 1.5；热 rail 的 fanout 分别是 1.5、2.25、3，其余 rail 调整为相应的平均值。这样可以分辨偏斜损失与总流量损失。

构造采用四个源节点的平移对称性和 8-token 局部 rank 轮转，因此每卡计算精确相等。主合成数据每 token 的执行 local rank 覆盖 0…7，是有意的控制变量；其专家热度和共现没有拟合生产分布。它满足已知拓扑、均衡与偏斜条件，不能被称为完整复现真实生产分布。home-only 场景没有现成跨机副本，快路径交换可能无候选；这正好检查算法是否错误地把副本当作免费可用。

两个 trace lift 从每个原 source 的 16,768 个 token 无放回抽 16,384 个完整 top-k，再拆为四个合成节点。352 个专家被重新映射到 32 卡，采用现有 CPU MoonEP `B=8, rt=0` 生成 owner；两层均满足驻留、无非法 physical slot、精确计算配额。原始 token 共现是真实的，节点标签与 32 卡排布是合成的。实际偏斜约 1.024，不能把它直接宣传为用户的 1.5–2 场景。

后续场景应加入：相同总字节但不同 burst、同 rail 目的端口 incast、2–3 个热 rail、热点跨 step 轮换、无可用副本、有限 B 预算、异构 NIC 带宽。实际多机数据至少采集 source node/local rank、完整 top-k、最终 owner、驻留、有效 token mask、dispatch/combine 时间和每 NIC 计数；有连续 step 才能评估副本稳定性。

三个主场景最忙 NIC 的单向 payload 分别为 6,144、9,216、12,288 份 token。在上述 hidden/BF16/40 GB/s 假设下，单向 RDMA 端口服务下界分别为 **2.202、3.303、4.404 ms**。这是从精确字节数计算出的下界，未含消息启动、转发和计算；不是实测完整层延迟，也不是新算法收益。

![相同总通信量下的 rail 偏斜与端口服务下界](../../outputs/multinode_rail/rail_comparison.png)

完整参考调度的 MoE 时间分别为 11.606、15.970、20.543 ms，保存在 [模型结果](../../outputs/multinode_rail/simulation/summary.json)。这些绝对值依赖上述阶段屏障、整 chunk 调度及假设速率，主要用于检查资源与依赖是否接通；比较优化器前还需校准并做调度/参数敏感性分析。

## 快路径算法建议

首版采用 **同专家执行位置交换**。保持 MoonEP 生成的每个 `(expert,执行 rank)` 计数不变，比只保持 rank 总 pairs 更强；这样 expert 分组大小、padding 和既有驻留都保持不变。

例如，源机 0 的热 rail 0 上 token A 的专家 e 在机 1 执行，冷 rail 1 上 token B 的同一个 e 在机 0 执行。交换这两个 e 的执行位置：A 留在源机，B 使用已有的远端 e 配额。若 A 没有别的专家在机 1，则 rail 0 少发送一份；若 B 已因其他专家访问机 1，rail 1 无新增发送，否则增加一份。两种情况分别是减少总字节或以相同总字节削峰，token 的 source 与路径模式都不改变。

实现步骤建议如下。

1. 在 seed 的临时 owner/raw dst 产生后统计 node mask、各 token 在各 node 的 pair 计数和每端口负载，并尽量与这次映射扫描融合。前置 logical top-k/tpe 尚不知道最终执行节点，不能替代这一步。每个节点为每个专家选择最多一个有非零配额的执行 rank 对 `(a,b)`，优先一个本机副本与一个远端副本。
2. 每源 GPU 并行扫描自己的 4,096 个 token。每 token 最多检查 top-8 的 8 个候选，只保留一个提案。记录“旧 node 是否被完全删除”和“新 node 是否首次加入”两位签名，计算真实 RDMA 增减和 NVLink 增减，不能用远端 pair 数代替。
3. 按 `(expert,方向,两位签名)` 建固定桶，在同源机器的 8 张 GPU 间交换计数和少量配对记录。只协调 metadata，token payload 不在源机预搬运。优先匹配热 rail 的减少与已有远端访问的冷 rail 补偿，再考虑有容量的冷 rail 新增。
4. 每个桶确定双向相同接受数，使用直方图和 prefix 分配唯一 partner；每 token 一轮只提交一次，避免掩码收益被另一提案改变。无候选或未获配额的 pair 保持原地址。
5. 对配对的同专家交换原始唯一执行地址。双方从稳定的旧 proposal record 读取，完成后仅发布一次最终 `dst/src_info/physical_topk` 和去重结果。不能先写最终 `src_info` 再在其后改 owner，也不能一端覆盖旧地址后让另一端读取。

一个专家本轮至多一个无向执行 rank 对，则每节点最多 `2E×4=2,816` 个有向签名桶；每源 GPU 的 int32 计数为 11 KiB，同机 8 卡合计 88 KiB，不含配对记录。每 GPU 的候选工作约 `O((T/R)K)`，节点内配额匹配只处理小表；同一专家的匹配独立并行。为限制小表常数，首版只选择固定数量的热/冷 rail 配对，不枚举所有 rank-pair 和所有 top-k 子集，不使用线性规划或通用排序。

此交换保持 `(expert,rank)` 数量与原始执行槽集合，所以可以复用 expert offsets 与 padding。若只保存 owner 而未保存唯一槽地址，仍需重新编码索引。现有 MoonEP 的跨卡 peer store 和 barrier 基于其 NVLink 后端，扩为跨机发布 metadata 必须由新 backend 实现并计时，不能仅把 R 改为 32 就认为完成多机支持。

## 端口价格与接受预算

提案代价采用本轮冻结的端口压力，例如 `bytes / effective_bandwidth` 的服务时间及超出目标值的部分。允许冷 rail 的单侧增流量，只要配对后更有利；旧的“每个单独提案都不增通信”规则会拒绝不少削峰机会。

只有本机统计不足以防止远端 RX incast。维护当前 seed 的 `F[4,8,4]` 全局快照，得到 32 TX 和 32 RX 端口预算。对 dispatch/combine 分别按其方向、dtype 和带宽展开。设该阶段旧最大端口服务时间为 `τ0`，则每资源 `c` 的非负字节余量为 `slack_c=max(0,τ0×BW_c−current_bytes_c)`。给每个 RX 的余量预先分给三个源节点，分配总额不得超过余量；节点内匹配独立使用自己的 TX 与获配 RX 额度。

为避免逐 token 争锁，先对每个配对类型提出数量 `n_j`，计算各资源的正增量 `U_c = Σ_j n_j max(Δ_jc,0)`，再令 `θ_c=min(1,slack_c/U_c)`；`U_c=0` 时取 1。接受数量为 `n'_j=floor(n_j×min_{c:Δ_jc>0}θ_c)`，没有正增资源时缩放系数取 1；两个方向一起缩小，配额仍守恒。这里不抵扣别的配对的负增量，较保守，但可并行且无需 LP。若没有当前快照，只能将未知 RX slack 设为零才能给出该端口模型的保守保证；历史压力只适合作启发式。

端口约束通过也不等于完整层时间必然下降。必须在仿真中计入 NVLink、planner、计算依赖和权重成本。首版量化收益门槛为 `通信关键路径节省 > 新增planner时间 + 未隐藏的副本/metadata成本`；把零收益和回退场景保留下来。

## 专家排布与更强的成组交换

如果当前驻留没有可交换的跨机副本，快路径没有自由度。慢路径需要显式增加这部分自由度，或者承认本轮无解，不能通过改 source rank 或更改逻辑 top-k 绕过约束。

建议先用 `source node × source rail × expert` 小计数表及有限 token 样本，为热点源节点选择高价值副本。优先统计它能消除的 **最后一个远端 node 访问**，而非只统计专家频率；多个专家共现时，单副本收益可能为零。每个 rank 只保留固定 top-c 候选，沿固定的小度数 rank 配对图尝试有限次互惠复制：a 的专家 e 复制到 b，b 的专家 f 复制到 a，交换相同数量的计算对，保证每 rank 的总配额不变。副本槽 B=8 是包括负载均衡副本在内的共同上限。没有可行互惠项就保持原计划。

权重传输、量化 scales、训练梯度同步和更新新鲜度都进入成本。慢路径可按 epoch 更新并设迟滞，但更新频率必须用连续 trace 验证；训练中驻留可复用不代表权重值更新和梯度同步免费。冷副本创建与热副本复用分别报告，实际新增数和等副本数量对照都要保留。

当同一个 token 在远端 node 的多个专家同时迁移才能省掉一次 RDMA 时，单专家交换可能停在局部最优。第二版再引入固定候选的 whole-rank group 交换，迁移该 token 在一个执行 rank 上的整个专家组，并匹配反向等大小组。只有它包含该 node 上该 token 的全部 pairs 才有删除 node 的收益；跨多个 donor rank 的 bundle 需要更复杂的配额协调，首版不默认支持。

这种 generic group 交换会改变 `(expert,rank)` 分组，必须重建 expert segments、padding 和最终地址，也可能改变 GEMM 效率。已有四轮 bucket exchange 可作为离线质量参考；不应直接扩到 32² 候选再作为在线默认。

## 两倍 planner 预算如何验证

旧实测是 8 卡 H200、全组 134,144 token：B8 原 planner 约 63.852 μs，四轮协调交换组件约 254.456 μs。后者已预先构造 seed/support，而且未包含最终通信地址重建。两者不能相加当作完整新实现实测，也不能把 254 μs 简单除以 4 外推到 32 卡。证据见 [原报告](../../REPORT_ZH.md) 与 [桶交换记录](../../experiments/bucket8/README.md)。

本次门槛定义为同一 4×8 GPU 环境、同一 131,072/top-8 输入、同 B/驻留成本合同下，`max_rank(T_new_full_planner) ≤ 2 × max_rank(T_baseline_full_planner)`。基线是功能等价的多机 MoonEP planner；统计、collective、驻留决策、同步、最终地址与去重都必须包含。若输入统计预计算，两边同样排除并另报完整在线路径。比较每 case 的 p50 和 p95，不能只用跨场景平均掩盖回退。

首先测试零交换/一轮交换、已有驻留/新增驻留、节点内协调/全局协调四组消融。完整 planner 超预算就降低候选、取消第二轮或回退原路径；候选数与 O(P) 复杂度相同只能说明有实现希望，不能证明时延达标。128K 数据下的模拟层时间改善也不能替代实机 kernel 验证。

## 检查与复现

```bash
python3 analysis/multinode_rail/generate.py
python3 analysis/multinode_rail/simulate.py
python3 analysis/multinode_rail/verify.py
python3 analysis/multinode_rail/plot.py
```

[独立核验](verify.py) 不导入生成器计数函数，使用 token rank/node bitmask 重新核对 7 个场景的流量、驻留、精确计算配额和 trace 行来源，并比对事件模型的字节、代码与输入哈希；报告见 [independent_audit.json](../../outputs/multinode_rail/independent_audit.json)。它另外检查 RDMA 后 NVLink 依赖、端口排他与反向 combine、node 去重、目的 local-rank 置换不改变 RDMA、单机 RDMA 为零，以及最小同专家交换例子。例子将 RDMA 副本数从 2 减到 1，保持 NVLink 副本数 2 和逐 `(expert,rank)` 计数不变；这是可行性用例，不是 128K 优化收益。

已完成同专家交换、有限副本选择和稀疏图单轮交换 CPU 参考，详见 [首轮结果](PLANNER_RESULTS_ZH.md)。后续需要校准事件模型、扩展 burst/漂移场景，并测量单轮 GPU 路径与多机 metadata 发布。所有阶段保存无收益/回退结果和相同输入的基线，不将组件速度或合成模型结果写成物理多机端到端加速。
