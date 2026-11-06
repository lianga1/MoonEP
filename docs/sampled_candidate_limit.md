# 八卡、128 候选、四轮 planner

本配置将候选数量限制和所需缓存更新直接接入 `sampled_group` 原生接口。八张 GPU 各自处理原本来自自己的 token，按原有全局顺序匹配等量交换；逻辑 rank 数、专家原属卡和计算负载均保持八卡语义。

## 启动配置

```python
from moonep import Buffer, SampledPlannerConfig

# 已初始化八 rank 的 NCCL 组，每个进程已设置自己的 CUDA 设备。
buffer = Buffer.from_model_config(
    model.config,
    S=local_tokens,
    num_ep_ranks=8,
    B=8,
    planning_backend="sampled_group",
    planner_config=SampledPlannerConfig(
        candidate_limit=128,
        rounds=4,
        # sample_size=256, groups_per_class=2, sample_seed=20261026
    ),
)
```

如果调用方已有 H/E/K，也可直接使用：

```python
buffer = Buffer(
    S=local_tokens, H=hidden_size, K=top_k, E=num_experts,
    num_ep_ranks=8, B=8,
    planning_backend="sampled_group",
    planner_config={"candidate_limit": 128, "rounds": 4},
)
```

H、E、K 仍直接复用模型或 Buffer 参数。`candidate_limit` 在整个 EP 组内必须相同；有效配置不一致会在初始化时拒绝。配置在 Buffer 生命周期内固定，变更后需要重新构造并预热 Buffer，然后再捕获 CUDA Graph。

`candidate_limit=0` 是兼容默认值，沿用全部候选路径。正数上限会按当前 K、保留组数和实际候选数缩小，因此模型变化后无需手工修改 kernel。`rounds=4` 表示全量计算任务的四轮交换，不表示副本选择只进行四次。B=8 是每张卡的额外专家槽上限，包含原生负载均衡已经需要的槽位。

## 128 限制的对象

一个候选方案包含两张卡上的反向专家组：两边移动相同数量的计算任务，至少一边减少 token 的远程目的地。组内专家数量可为 1…K；每个方向、任务数、收益类别保留两组，之后才形成这些双向方案。

GPU 先构造完整候选，再按起点时的“预计减少的通信次数 / 新增专家副本数”排序。并列时依次选择收益更高、副本更少、原始候选编号更小的方案。初始无正收益或不可行的候选排在后面，有剩余名额时也保留。前 128 个进入后续贪心选择。

每选中一个方案，已驻留专家和剩余槽位都可能变化，因此保留候选的收益和可行性继续重算。候选编号保持原始编号，以维持确定性的并列规则。限制大于当前候选数时，保留全部候选。

同时，kernel 建立被保留候选实际需要的缓存表项名单，只更新这些表项。每次计算一个表项，仍检查该类别中的全部原始保留组，避免丢失组合收益。排序、名单建立和缓存维护都在 GPU 计时路径内，没有用 CPU 预选方案。

全量 token 扫描、双向等量名额、地址重建、全组错误汇总和原生回退继续执行。每个计算对仍恰好执行一次，接受的交换保持各卡计算任务总量相等。128 候选是近似搜索参数，不保证在任意 trace 上保留相同收益。

## 已保留的选择依据：集成前实测

这组测量来自以 `b1518bf` 为基线的实验包装器；其算法与本次接入的候选剪枝及缓存维护方案一致。它用于保留选择 128 的依据，不能自动当作本次生产接口重新构建后的测量。

环境为同机八张 H200、Torch 2.13.0+cu130、CUDA 13.0、NCCL 2.29.7。每层 N=134,144、每卡 S=16,768、E=352、K=8、B=8、全组采样 256、每类两组、四轮交换，种子 20261026。每卡 10 次样本、每次 5 次预热后的 CUDA Graph 重放；先取每卡中位数，再取最慢卡，最后对同一八层取平均。

| 集成前八层实测 | 全部候选 | 上限 128 |
|---|---:|---:|
| 完整 planner 平均 | 815.773 µs | 683.292 µs |
| 八层远程 token 目的地次数 | 3,791,665 | 3,887,660 |
| 相对原生 MoonEP 减少通信次数 | 24.9794% | 23.0801% |
| 八层额外专家副本总数 | 507 | 434 |

128 候选降低 planner 时间约 16.24%，保留约 92.40% 的原通信节省。不限候选时各层实际有 922–1,026 个方案，平均接受 35.125 次；限制后平均接受 29.625 次。副本数量是八层合计，不是单卡容量。原生 MoonEP 的八层远程次数为 5,054,165。

完整计时包含原生起点、实时采样汇总、GPU 评分/排序/剪枝、副本选择、四轮交换、布局生成、校验和发布。输入直方图、JIT、分配及校验程序本身在计时外；专家权重、token payload、梯度传输、GEMM 和优化器不在此测量范围。因此这些数据不代表完整训练步提速。

原始记录、校验摘要及可重新计算的汇总见 [证据目录](../experiments/candidate_limit128/README.md)。该目录仅包含本配置相关的四轮历史记录，保留其中全部候选上限，避免只摘取最快一项。

## 本次集成验证

同机八卡回放入口使用正式配置接口，输入只读取每卡自身的 router 分片。CPU 数组摘要只在计时之外用于校验：

```bash
MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 \
  torchrun --standalone --nproc_per_node=8 \
  -m tests.benchmark_sampled_candidate_limit \
  --fixture-root /path/to/router-fixtures \
  --candidate-limit 128 --rounds 4 \
  --expected-hashes experiments/candidate_limit128/cpu_array_hashes.json \
  --output /path/to/new-cap128-result.json
```

用新的输出路径和 `--candidate-limit 0` 测同次完整候选基线；保留两个原始文件，不覆盖旧测量。真实 router 输入不随本提交公开打包，调用方应使用已有、获授权的同一份八层 fixture；文件哈希会记录在结果中。

<!-- INTEGRATION_VALIDATION_START -->
正式配置接口已在同机八张 H200 上完成本次复测。两组都使用四轮交换、同一八层 fixture、每卡 10 次计时样本且每次重放 5 次；先取每卡中位数，再取最慢卡，最后对八层取平均。源文件 SHA、设备身份和 fixture 哈希均已核对。

| 本次正式接口复测 | `candidate_limit=0` | `candidate_limit=128` |
|---|---:|---:|
| 完整 planner 平均 | 815.856 µs | **684.846 µs** |
| 同次原生 MoonEP planner 平均 | 68.581 µs | 68.428 µs |
| 八层远程 token 目的地次数 | 3,791,665 | 3,887,660 |
| 相对原生 MoonEP 减少通信次数 | 24.9794% | **23.0801%** |
| 八层额外专家副本总数 | 507 | 434 |

128 候选在本批次降低完整 planner 时间 **16.06%**，保留原方案约 **92.40%** 的通信节省。两组的通信量、副本数和最终执行卡哈希逐层与对应历史结果一致；各层实际 CUDA Graph 输出、所有 rank 的完整选择历史、独立 CPU 对照和物理布局审计均通过，正常路径没有回退。

| 检查 | 独立用例数 | 执行范围 | 结果 |
|---|---:|---|---|
| 配置解析与约束 | 18 | 单进程 | 全部通过 |
| 候选选择器 | 3 | 单 GPU | 全部通过 |
| cap128 正式八卡接口 | 5 | 每例八 rank 共同执行 | 每个 rank 全部通过 |
| 已有接口兼容性 | 5 | 每例八 rank 共同执行 | 每个 rank 全部通过 |

八个 rank 各报告“5 passed”，表示 **5 个分布式用例**，不是 40 个独立用例。选择器覆盖交付形状 E352/K8、上限缩小到较小模型的候选容量，以及共享内存不足时使用的通用 kernel；同时检查同一 CUDA Graph 更换输入、异常输入和恢复。正式接口检查跨卡配置不一致的拒绝、强制全组回退、计划保存与复用，以及带路由权重的前向/反向。

上述四类检查和两组真实 trace 基准共六个阶段，退出码全部为 0。原始计时、日志、退出标记、实际运行脚本、SHA 清单和可复算汇总保存在 [本次集成证据](../experiments/candidate_limit128/integration/README.md)。这是单次同环境复测，尚未做随机执行顺序的多批次稳定性研究；计时范围仍只覆盖 planner，不能据此宣称完整训练提速。
<!-- INTEGRATION_VALIDATION_END -->

## 训练接口

沿用 `Buffer.dispatch()`、最终 `MoonEPCommPlan`、权重预取和副本梯度归并接口。每次规划读取当前路由，采样种子只固定样本位置。候选上限不会提前发布尚未通过全局检查的计划。

训练时仍需按照前向计划保存和恢复专家副本槽、复用对应计划反向，并计入每次权重更新后的副本刷新和梯度归并成本。完整模型与流水线的端到端验证范围见 [sampled_group 生命周期说明](sampled_group.md#训练与生命周期)。
