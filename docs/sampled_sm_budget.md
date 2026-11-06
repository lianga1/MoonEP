# 完整 planner 的 SM 上限与分段融合

此路径在 `sampled_group` 内控制完整 planner 的执行规模，保留128候选、四轮交换和原有负载约束。当前严格32SM建议 `fuse=True` 的选择性融合，配合八卡频率分治与计数复用；新结果见 [后续优化报告](../experiments/sm32_divide/README.md)。`sm_limit=0, fuse=False` 默认继续原路径。本文后半的完整预算扫描表保留802版历史结果。

## 启动配置

```python
from moonep import Buffer, SampledPlannerConfig

# 八个进程已初始化通信组，并分别设置自己的 CUDA 设备。
buffer = Buffer.from_model_config(
    model.config,
    S=local_tokens,
    num_ep_ranks=8,
    B=8,
    planning_backend="sampled_group",
    planner_config=SampledPlannerConfig(
        candidate_limit=128,
        rounds=4,
        sm_limit=32,
        fuse=True,
        # sample_size=256, groups_per_class=2, sample_seed=20261026
    ),
)
```

也可以传 `planner_config={"candidate_limit": 128, "rounds": 4, "sm_limit": 32, "fuse": True}`。`fuse=True`启用下文的选择性融合；设为`False`保留同一SM预算及分治优化、拆开可融合阶段。E、K、H仍直接复用模型或 Buffer 参数，采样数和每类保留组数仍有原来的默认值。模型支持范围、字段识别规则和槽位约束见 [配置说明](sampled_group.md)。

`sm_limit` 是非负整数，正数不能大于本卡实际SM数量；`fuse` 必须是布尔值，且 `True` 要求正数预算。所有rank的配置必须一致，并在共享缓冲区分配前集体验证。改变预算、融合设置或模型形状时，需要重新构造和预热 Buffer，再重新捕获 CUDA Graph。

## 32SM保证的具体含义

线程块（CTA）是一组共同执行的GPU线程，一个线程块不会同时跨越多个SM。受限路径每次最多启动 `sm_limit` 个线程块，并将这些 kernel 依次放在同一个 CUDA stream 上执行。因此，对于一个在途的 planner，所有重叠执行的规划线程块合计不超过预算，同时使用的SM数量也不超过预算。

这不是SM固定分区：不同阶段可能被安排在不同的物理SM上；一个SM也可能同时容纳多个线程块。预算不保证恰好使用32个SM，也不直接表示SM利用率。它不为其他工作独占保留剩余SM，也不消除显存带宽竞争。

范围包括：

- 输入路由复制、原生 MoonEP 起点及原生回退计划。
- 样本汇总、候选构建和排序、专家副本选择。
- 四轮全量提议、计数汇总、交换名额和写回。
- 最终稳定分组、物理地址生成、全局状态检查及来源信息发布。

范围不包括 Buffer 初始化、内存分配、JIT编译、外部专家直方图构建，以及 planner 之外的 token dispatch/combine、权重刷新、梯度归并、GEMM或优化器。Buffer仍只允许一次在途调用；多个Buffer或其他stream上独立运行的任务不会自动共享这份预算。

`Buffer(num_sms=32)` 本身不足以提供这个保证。它原来只影响部分原生kernel。新配置会把原生起点限制为 `min(num_sms, sm_limit)`，并另外控制新planner中的全部阶段。放宽 `sm_limit` 不会自动增大仍为32的原生 `num_sms`。

## 如何完整限制执行规模

| 阶段 | 正数 `sm_limit` 下的实现 |
|---|---|
| 扫描与计数 | 固定数量的物理线程块循环处理虚拟数据批次；数据量不减少 |
| 专家候选排序 | 在受限线程块内排序固定大小的片段，再进行有界归并；保留完整比较规则和原始候选编号，不调用CUB的设备级排序 |
| 贪心副本选择 | 保持单线程块及其共享缓存；保留128个候选上的实时重算 |
| token交换 | 保留原始虚拟块编号、token顺序和来源卡顺序，依次计算提议、前缀、名额与写回 |
| 最终任务布局 | 复用原生按专家稳定排序的顺序，按8个目标执行卡做稳定分组；代替CUB的3位排序 |
| 跨卡小表 | 各rank写入自己的共享区域，设备端同步后按rank顺序读取或归并，读完后再次同步 |
| 初始化与发布操作 | 使用受限复制、清零、状态合并和计划选择kernel，避免遗漏框架自动启动的kernel |

例如每卡16,768个token、K=8时，有些扫描包含524个虚拟数据批次。32个线程块轮流处理这些批次，而不是只处理前32批。用于稳定排序的计数表仍按原始批次编号保存；完成速度不会影响哪个token先拿到交换名额。

新的小表通信只用于当前同机NVLink八卡环境，不是通用NCCL替代品。初始化仍使用正常的进程组协商和内存句柄交换。运行时，每次小表通信都有“可读”和“全部读完”两个同步点，使用可重复的四状态信号，支持在同一个CUDA Graph中反复调用。错误输入也必须走完相同的通信序列，最后统一回退，不能让某个rank提前退出。

当前采用选择性融合：输入不超过256个整数时，`fuse=True`使用一个线程块完成通信；更大的表恢复“写入、同步、读取、同步”四次受限启动。原来的大表协作网格融合仍保留为诊断入口，但完整通信实验显示它在32SM下更慢，因此不再作为生产默认。两种路径都保留读完同步，防止较快的rank覆盖较慢rank尚未读完的数据。

受限路径也调整了小表内部的工作划分：一个warp（一组32线程）负责一个候选类别的前几名选择，或一个交换计数桶的前缀扫描。同一线程块的8个warp可以处理8个独立类别/桶，减少原来逐类别使用整个线程块同步的需求。前缀仍沿原始虚拟批次编号累加，候选比较规则也保持不变；这是执行布局调整，不是进一步剪枝。

## `fuse=True` 当前的选择性融合

当前采用分段融合，仍保留真正的数据依赖和跨卡同步边界，没有把整个planner改成一个巨型kernel。

| 分段 | 融合内容 |
|---|---|
| 小表跨卡汇总 | 输入≤256个整数时将写入、两次同步、读取/归并放入单块kernel；大表保持四次受限启动 |
| 候选初始排序 | 将初始收益评分和各片段内部排序放入同一kernel；后续归并仍单独执行 |
| 每轮提议 | 保持提议和稳定编号分开，256-token直方图不变；原提议融合只留作诊断 |
| 每轮配额 | 将清空名额表、汇总各来源计数和计算反向匹配名额放入一个kernel |
| 最终布局准备 | 将token重复目的地标记、输入验证/专家计数及8桶直方图合并 |
| 最终计划选择 | 将 `cu_seqlens`、`dst`、副本表、补零范围和通信统计共五组结果的选择合并到一个kernel |

跨虚拟块的前缀、归并、最终地址编码等阶段仍保留必要的边界。四轮交换依然顺序执行四次，不改变搜索轮数，也不因为融合而接受更多或更少的提议。融合能减少启动和中间读写，但可能改变每个线程块的工作量、寄存器需求和等待时间，因此不预设它一定比未融合版本快。

## 验证和测量

以下矩阵与初版结论来自历史提交 `802af3f`。当前实现的新测量、逐阶段解释与八卡分治见 [32SM后续优化报告](../experiments/sm32_divide/README.md)。

合成正确性测试不依赖私有router文件：

```bash
python -m pytest -q tests/test_planner_config.py

MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 \
  python -m pytest -q -s tests/test_sampled_sm_budget.py \
  -k 'exchange or selector'

MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 \
  torchrun --standalone --nproc_per_node=8 -m pytest -q -s \
  tests/test_sampled_sm_budget.py -k 'not exchange and not selector'
```

真实八层trace基准使用正式配置入口，保持候选128、四轮和其他模型参数相同。输入只读取每卡原始来源分片。以下路径需要替换为已有、获授权的router fixture和新的输出目录：

```bash
MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 \
  torchrun --standalone --nproc_per_node=8 \
  -m tests.benchmark_sampled_sm_budget \
  --fixture-root /path/to/router-fixtures \
  --candidate-limit 128 --rounds 4 --sm-limit 32 \
  --expected-hashes experiments/candidate_limit128/cpu_array_hashes.json \
  --profile-dir /path/to/new-run/traces \
  --output /path/to/new-run/results.json
```

加上 `--fuse` 可测当前32SM选择性融合路径。要复现下方48/64/96/128的历史预算矩阵，必须在历史提交 `802af3f` 的独立checkout运行相同命令；当前源码的融合策略已变化，用它测试其他预算属于新的实验，应另存结果。预算数值必须不超过当前设备实际SM数。`--sm-limit 0` 且不传 `--fuse` 是兼容路径。与旧提交 `407c223` 对比时，应另保留其独立源码、源文件摘要、同一fixture和原始记录，不能把旧的684.85µs直接当成本批次基线。

默认每卡保留10次样本，每次执行5次预热后的CUDA Graph重放；每层取各卡中位数中的最大值，再对同一八层取平均。测量包含完整planner及小表跨卡同步，不包含token、权重、梯度的实际payload、GEMM和优化器。Profiler在正式计时之外运行。

正确性核对包括全部执行位置、物理地址、专家副本、选择历史、来源信息和CPU对照；另外检查更换Graph输入、错误回退与恢复，以及前向/反向复用。通信收益按去重后的 `(token, 远程目的卡)` 次数统计。只改变执行预算或融合方式时，目标是与同配置旧planner的输出逐项一致，而不是允许通信收益发生近似漂移。

`--profile-dir` 会为每个rank保存Graph trace，核对每个kernel的线程块数、时间上是否叠加，以及受限路径是否仍出现NCCL kernel。这个检查给出planner并发线程块的保守上界，不测量实际活跃SM编号或每个SM的利用率。

<!-- SM_BUDGET_VALIDATION_START -->
2026-11-06，在 `omni-dev` 同机8张H200上完成正式矩阵。每卡132个物理SM，Torch 2.13.0+cu130、CUDA 13.0、NCCL 2.29.7。固定每层 N=134,144、每卡 S=16,768、E=352、K=8、B=8、全局采样256、每类保留2组、候选128、四轮交换、种子20261026。旧版本使用独立的 `407c223` 源码；同一旧基线在矩阵前后各测一次。

| 配置 | 八层完整planner平均 | 相对本批次前基线 | 第0层每次planner的kernel数 |
|---|---:|---:|---:|
| 旧版 `407c223`，矩阵前 | 686.216 µs | 基线 | 89，其中7个NCCL |
| `sm_limit=32, fuse=False` | **868.298 µs** | +26.53% | 104 |
| `sm_limit=32, fuse=True` | 895.517 µs | +30.50% | 64 |
| `sm_limit=48, fuse=True` | 808.995 µs | +17.89% | 64 |
| `sm_limit=64, fuse=True` | 777.983 µs | +13.37% | 64 |
| `sm_limit=96, fuse=True` | 705.497 µs | +2.81% | 64 |
| `sm_limit=128, fuse=True` | 699.035 µs | +1.87% | 64 |
| 旧版 `407c223`，矩阵后 | 685.629 µs | −0.09% | 未重复采集trace |

32SM融合版本比32SM未融合版本慢 **3.13%**。虽然kernel数量从104降到64，整体延迟仍增加，不能把“启动更少”直接当成“更快”。每个线程块的工作量、资源需求和同步方式都可能影响结果；本次尚未通过单独实验确定这些因素各占多少。用于预算检查的trace存在跨rank profiler启动偏斜，不能将其中的kernel时长直接当作正式分阶段计时。

802历史版本当时的选择是：**严格要求32SM预算时使用 `fuse=False`；只追求这批负载的最低planner延迟时，保留旧执行路径。** 放宽融合路径的预算会接近旧版延迟，但本次没有任何受限配置快过旧版。尚未测量48/64/96/128的未融合组合，不能据此断言这些预算下融合最优。默认配置仍为 `sm_limit=0, fuse=False`，没有自动切换。

所有正式配置的八层累计远程 `(token, 目的卡)` 次数均为 **3,887,660**，原生MoonEP为 **5,054,165**，减少 **23.0801%**；额外专家副本累计均为 **434**。副本数是八层合计，不是单卡槽位数。结果与旧版128候选、四轮计划一致：执行卡、专家支持表、选择历史、候选诊断和所有最终物理plan输出逐项/摘要核对通过，正常trace路径没有回退。旧源码摘要与 `407c223` 的提交内容、新源码摘要与最终被测源码均已独立核对。

合成测试共 **53个独立用例**：24个配置测试、9个单GPU组件测试、15个八卡新接口测试及5个八卡兼容测试。分布式用例由8个rank共同执行，不能把“各rank通过15项”累加成120个独立测试。覆盖更换Graph输入、重复标量/大表通信、不满线程块的尾部、动态E/K/采样设置、跨卡配置不一致的拒绝、错误回退和恢复，以及带路由权重的前向/反向。

独立复算56份原始GPU trace通过：旧版及6种受限配置，各8个rank、各3次完整Graph重放。所有受限配置的最大单次启动块数与时间重叠后的保守块数上界均不超过所设预算（32/48/64/96/128），受限路径没有NCCL kernel；采样、专家选择、四轮提议/前缀、布局和发布锚点均完整。trace覆盖第0层，完整输出及CPU对照覆盖全部8层。

正式计时和输出、原始日志、所有阶段退出码、预算trace、首次未达到性能目标的优化尝试及复算记录已保留在 [实验目录](../experiments/sm_budget/README.md)，并附有[逐层结果](../experiments/sm_budget/summary.md)和离线复算脚本。此处只报告planner时间与计划通信量；没有实际传输token/专家权重/梯度payload，也没有测完整训练步或与GEMM重叠的吞吐收益，不能据此宣称训练净收益。
<!-- SM_BUDGET_VALIDATION_END -->
