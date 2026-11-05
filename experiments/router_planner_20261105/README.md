# MoonEP 路由规划实验：八个逻辑 rank、两张物理 GPU

本目录保存 2026-11-05 的路由规划 kernel、验证程序和原始测量结果。实验保留 router dump 中的 **8 个逻辑 rank**，在 **2 张物理 H200** 上分别测量 GPU 计算与通信操作带宽，再对原始八卡流量建立训练通信成本模型。

本次确认的结果是：稳定桶前缀交换将本来源四轮交换计算从 **179.163 µs 降至 86.314 µs，减少 51.824%**；完整的 GPU 驻留选择计算为 **887.841 µs/层**。后者不包含八卡通信，也不是完整 planner 的执行时间。在每次更新刷新专家权重、归并 FP32 专家梯度的训练模型下，本策略目前没有相对 MoonEP `rt=0` 的全局正收益，因此默认不建议启用额外复制。

## 实验协议

| 符号 | 本次取值 | 含义 |
|---|---:|---|
| `N` | 134144 | 每层全部 token 数量 |
| `R` | 8 | 原始逻辑 rank 数量 |
| `S=N/R` | 16768 | 每个原始来源的 token 数量 |
| `K` | 8 | 每个 token 选择的专家数量 |
| `E` | 352 | 专家总数；每卡拥有 44 个原生专家 |
| `B` | 8 | **每卡**可用的额外专家槽位，不是全局 8 个 |
| `d` | 256 | 从全部 token 中无放回抽取的样本数 |
| `g` | 2 | 每个方向、任务数量、收益类别保留的专家组数 |
| `L` | 4 | 成组交换轮数 |
| 随机种子 | 20261026 | 固定全局样本 ID |
| 层数 | 8 | 原始 trace 的 8 层 |

`rt=0` 关闭按来源热门程度提前保留专家的步骤，仍保留负载均衡必需的副本。`rt=2` 在负载均衡前，每卡按原始来源 token 的专家热门程度选择两个保留目标，再与均衡所需副本共享 `B=8` 限制。实验优化方案从 `rt=0` 的执行计划开始，选择样本支持的双向专家复制组合，再执行四轮成组交换。

本实验对每一层全量检查专家可执行性、每个计算对唯一执行，以及每卡恰好承担 `S×K` 个计算对。`B=8` 是该 trace 的实验限制。原生 MoonEP 文档对通用训练集成要求 `B=E/R`；本次 trace 验证不能替代该通用要求。

## 算法与实现

`experiments/eight_rank_estimate/residency8.py/.cu` 实现八逻辑 rank 的 GPU 驻留选择：读取运行时样本，重建均衡必需的专家表，统计专家组、保留各类别前两组，组成双向候选，再执行共享槽位预算的贪心选择。这里的“驻留”表示某卡拥有可执行该专家的本地副本。每个候选专家组可以包含多个专家，`g=2` 不是“每组两个专家”。全部逻辑来源的数据已位于同一张 GPU；实际跨 rank 收集样本和发布专家方案的成本尚未计入。

`experiments/eight_rank_estimate/exchange_prefix.py/.cu` 在每个来源的 16768 个 token 上执行交换。它缓存当前支持表对应的可执行掩码，每轮为每个 token 只保留一个最佳提议；桶表示“旧执行卡、新执行卡、通信收益、任务数量”。线程块内的稳定计数与前缀替代全 token 的 CUB 排序，保留原来的 token 顺序与配额分配规则。每轮仍需要所有来源的计数，才能确定双向等量名额并提交交换。

对照实现是 `experiments/h100_planner/bundle_lane8.py/.cu` 中已有的八逻辑 rank CUB 排序交换。它是交换算法的对照，**不是原生 MoonEP planner kernel**。`baseline_virtual.py/.cu` 也是语义移植，不能用它的耗时替代原生八卡 planner 耗时。

## GPU 计算结果

环境为 NVIDIA H200，PyTorch `2.11.0+cu130`，CUDA `13.0`。每张物理 GPU 独立测量原始八个来源；每个配置保留 10 个样本，每样本执行 20 次 CUDA Graph replay。分配、编译、CPU 参考计算和正确性检查均在计时之外。

汇总方式为：先取每配置原始样本的中位数，再对同层的 8 个来源和 2 张物理 GPU 取最大值，最后对 8 层取算术平均。驻留选择没有来源维度，对两设备中位数取最大值后再跨层平均。该汇总不是八个真实 rank 同步运行的关键路径测量。

| 测量部分 | 八层均值 | 范围 |
|---|---:|---|
| 全局 GPU 驻留选择 | 887.841 µs | 含运行时样本索引读取及全部候选选择；不含跨卡收集/发布 |
| 原 CUB 排序交换，单来源四轮 | 179.163 µs | 每来源实际扫描 16768 个 token |
| 稳定桶前缀交换，单来源四轮 | 86.314 µs | 相同输入、候选、配额和最终执行计划 |
| 交换计算下降幅度 | 51.824% | 约 2.076 倍加速 |

四轮计时包含本来源缓存准备、种子执行计划恢复、提议和计数、本 GPU 上计数折叠与配额计算、计划提交。计时使用已验证的逐轮全局计数输入，因此不含实际计数集合通信与等待。不能把分项相加称为实测完整 planner，也不能把全局计算时间除以 8 当作单卡时间。

仍未测量：原生 MoonEP 八物理卡 planner、真实八卡样本/计数同步、专家方案发布、最终物理地址构建与发布，以及八卡完整训练时延。

## 通信与训练成本

通信统计按去重后的 `(token, 目的卡)` 计数：同一 token 的多个计算对若在同一张远端卡执行，只计一次 token 发送。以下数字是原始八层之和；专家复制数量也跨层累加。

| 方案 | 远端 token 目的地数 | 每次完整刷新需要的专家副本数 |
|---|---:|---:|
| MoonEP `rt=0` | 5,054,165 | 71 |
| MoonEP `rt=2` | 4,771,576 | 203 |
| 样本驻留 + 成组交换 | 3,791,665 | 507 |

优化方案相对 `rt=0` 减少 **24.979%** 的远端 token 目的地，同时增加 **436 个**跨层累计副本。减少 token 发送不等于训练总成本下降。

训练模型采用 `H=7168`、专家中间维度 `I=2048`：每个专家的三个矩阵共 `3HI` 个参数，BF16 权重为 **84 MiB**，FP32 参数梯度为 **168 MiB**。这些尺寸是本实验假设，router dump 本身不提供权重内容。每次参数更新后重新传权重；每个更新只处理一次该 trace，即 `M=1`，包含前向和反向的四次 token 通信，以及副本梯度回传累加。

两卡实测操作有效带宽分别约为 dispatch **156.323 GB/s**、combine **182.010 GB/s**、专家权重 **330.315 GB/s**、FP32 梯度 **278.171 GB/s**。数值按每个发送/接收端点的单方向有效载荷计算，没有把 TX+RX 相加翻倍。原八卡模型逐层计算最忙发送或接收端点，再把层耗时相加。它假设单机无阻塞 NVSwitch；两卡校准没有测出八卡争用，校准中的 token 目的地结构也不同于原始八卡 trace。

`estimate_attempt03.json` 的基准场景如下，单位均为八层通信模型毫秒，**不含额外 planner、GEMM 或 optimizer**：

| 方案 | token 四阶段 | 专家权重 | 专家梯度 | 串行通信合计 |
|---|---:|---:|---:|---:|
| MoonEP `rt=0` | 230.153 | 8.800 | 20.898 | 259.851 |
| MoonEP `rt=2` | 216.760 | 21.332 | 50.663 | 288.755 |
| 样本驻留 + 成组交换 | 178.512 | 34.665 | 82.327 | 295.504 |

相对 `rt=0`，token 通信节约 **51.642 ms**，额外权重与梯度增加 **87.294 ms**，净增加 **35.652 ms**。相对 `rt=2` 净增加 **6.749 ms**。即使乐观地将所有方案的权重传输全部隐藏，优化方案相对 `rt=0` 仍增加 **9.787 ms**；该乐观模型相对 `rt=2` 有 **6.584 ms** 的余量，但还未扣除完整 planner，不能据此判定可启用。

“每层只刷新一次”要求对应权重版本和副本从前向保留到反向；原生跨层共享额外槽位池不会自动满足这一条件。多微批摊销需要固定映射、有效权重版本和正确梯度累积，当前仅提供条件模型，没有测得通用训练正收益。保留负结果，默认选择原生方案。

## 已通过的验证

- 八层驻留选择的所有中间表、候选顺序、接受历史和最终支持矩阵与 CPU 参考逐项一致；重复运行和 CUDA Graph replay 检查通过。
- 两物理 GPU × 八层 × 八来源的四轮交换均与独立 CPU 参考逐轮一致；原排序版和前缀版的提议、计数、提交结果一致，并通过计时后的复核。
- 独立读回覆盖 128 条来源/层记录、512 个来源/轮次、24,480 个原始计时样本；重新计算 4,896 个 CUDA/主机中位数，并核验输入与执行代码指纹。
- 两卡带宽校准执行真实 MoonEP dispatch/combine、专家权重预取和 FP32 梯度归并。检查真实 payload、专家槽位内容、home 梯度与远端副本梯度之和，以及未参与槽位不变。梯度重置在每次测量之外执行，避免重复测到已清空的梯度槽位。

原始记录还保留传输附带的 macOS `._` 元数据文件指纹，以及测量后修改的非执行辅助脚本差异；独立审计对这些情况单独说明，没有把它们当成执行代码。GPU 结果张量没有全部写入 JSON；离线读回核验的是保存的检查状态、执行代码和输入指纹，不声称重新执行了 GPU 检查。

## 目录与复现

下列命令从本实验包根目录执行，即包含 `experiments/`、`analysis/`、`outputs/` 的目录。保留这些相对目录关系，不要把 Python 文件单独移动到同一级。

```text
experiments/eight_rank_estimate/   R8 驻留、前缀交换、微基准、带宽校准、模型与审计
experiments/h100_planner/          原 R8 排序交换与语义对照
experiments/two_gpu_overlap/      原生 MoonEP 预取/梯度辅助与早期两卡适配实验
analysis/                        CPU 参考及 router dump 处理
outputs/eight_rank_estimate/      原始结果、独立审计、模型与报告
```

GPU 微基准需要兼容的 CUDA PyTorch、CUDA 编译器、C++ 编译工具和 Ninja；CPU 参考准备另外需要 NumPy、Numba。两卡带宽校准需要编译本包中测试过的 `inputs/MoonEP` 快照，并使用可进行对称内存访问的两张互联 GPU。实测 CuTe DSL 是 `nvidia-cutlass-dsl==4.5.2`（快照 setup.py 声明的是 4.4.2），因此在隔离的 CUDA Python 环境预装实测依赖，再执行 `python -m pip install --no-deps --no-build-isolation -e ./inputs/MoonEP`，避免安装时悄悄改换 DSL。原 setup.py 保留不改。本包脚本优先导入该快照，不能用父仓库其他版本的 `_C` 二进制替代。CPU 依赖固定为 NumPy 2.0.2 与 Numba 0.60.0，GPU 主环境为上表所列的 PyTorch/CUDA。已测环境不代表其他 CUDA/PyTorch 版本均已验证。

本包附带用于复核的真实路由索引 NPZ，不包含 token 文本或模型权重；使用其他数据时，需要自行提供有权使用的原始 router dump 或符合下表的输入 NPZ。直接运行 kernel 微基准使用 `outputs/eight_rank_estimate/kernel_inputs/layer_00.npz` 至 `layer_07.npz`，每个文件至少包含：

| 字段 | 形状 | 含义 |
|---|---|---|
| `topk` | `[8,16768,8]` | 原始专家 ID，int32 |
| `seed` | `[8,16768,8]` | `rt=0` 执行卡，int32 |
| `support` | `[8,352]` | 优化后可执行专家，bool |
| `owner_prefixes` | `[5,8,16768,8]` | 初始和四轮后的 CPU 参考执行卡 |
| `stats` | `[4,4]` | 每轮交换数、搬动计算对数、接受 token 数、预测通信收益 |
| `native_experts_to_copy` | `[8,8]` | 原生计划中的额外专家 ID，空槽为 -1 |
| `sample_ids` | `[256]` | 固定全局采样 ID，int64 |

如果从原 router dump 重建，先用 `analysis/trace_audit.py`、`analysis/compare_rt2_b8.py` 和相应 replay 程序形成 `outputs/trace_audit/`、`outputs/rt2_b8_comparison/`、`outputs/replay/` 中的原始参考文件，再运行 `prepare_kernel_inputs.py`。该准备步骤会重新执行 CPU 驻留及四轮交换，并与归档 replay 比对；它不是一个只读取 router dump 就能跳过参考验证的转换器。

```bash
# 可选：由原始参考 NPZ 重新构建全轮次 CPU 对照，保留包内输入。
NUMBA_NUM_THREADS=1 python experiments/eight_rank_estimate/prepare_kernel_inputs.py \
  --output-dir outputs/eight_rank_estimate/kernel_inputs_rebuilt

# 两物理进程各自测原八来源；这里不建立两 rank 的路由问题。
torchrun --standalone --nproc_per_node=2 \
  experiments/eight_rank_estimate/benchmark_kernels.py \
  --input-dir outputs/eight_rank_estimate/kernel_inputs \
  --layers 0,1,2,3,4,5,6,7 --sources 0,1,2,3,4,5,6,7 \
  --mode graph --samples 10 --repeats 20 --require-residency \
  --output outputs/eight_rank_estimate/kernel_graph_new.json

# 真实两卡操作带宽；使用合成均衡路由校准，不改变原八卡 trace。
torchrun --standalone --nproc_per_node=2 \
  experiments/eight_rank_estimate/calibrate_bandwidth.py \
  --tokens 16768,65536 --samples 10 --hidden 7168 --intermediate 2048 \
  --output outputs/eight_rank_estimate/bandwidth_new.json

# 由原八卡流量与两卡带宽重建训练通信模型，无 GPU 调用。
python experiments/eight_rank_estimate/build_estimate.py \
  --traffic outputs/eight_rank_estimate/traffic.json \
  --bandwidth outputs/eight_rank_estimate/bandwidth_new.json \
  --weight-refreshes-per-layer 1 \
  --output outputs/eight_rank_estimate/estimate_new.json
```

输出文件使用新名字；保留先前的成功、失败和部分结果。`benchmark_kernels.py` 自动给两物理进程的输出追加 `_physical0`、`_physical1`。`residency8_validate.py` 提供更细的中间表及接受历史检查，需要原始 trace、原生种子、归档 replay 和 `traffic.json`。

本次交付的读回入口是 `audit_kernel_results.py`，对应固定的 `kernel_graph_attempt01_physical{0,1}.json` 与 `kernel_inputs/manifest.json`。它是该次证据的审计程序，不是自动选择最新结果的汇总器；发布版已经把清单中的历史路径重定位到本包对应输入，并继续验证原 SHA256。用 `python experiments/eight_rank_estimate/audit_kernel_results.py --output-dir local_runs/readback` 将新审计写入独立目录；不要用新输入替换旧证据。`analysis/trace_audit.py --synthetic` 是历史对照入口，仍依赖原工作区额外的合成生成器；本包的真实路由与主复现流程不使用该选项。

## 结果文件与集成边界

优先阅读 `outputs/eight_rank_estimate/report.html`、`kernel_summary.json`、`traffic.json` 和 `estimate_attempt03.json`。原始 Graph 样本在 `kernel_graph_attempt01_physical0.json`、`kernel_graph_attempt01_physical1.json`；正确性与独立读回分别在 `residency_gpu_validation_attempt01.json`、`kernel_independent_readback.json` 和 `estimate_independent_readback03.json`。带宽采用 `bandwidth_attempt02.json`，保留早期 attempt 作为历史证据。

`experiments/two_gpu_overlap/` 的 split-phase 适配器研究专家方案可用后提前预取、token 方案可用后启动 dispatch；训练梯度辅助复用原生归并 kernel，当前原生归并仅支持 FP32。不同流通过事件建立依赖，映射和副本槽位必须存活到消费结束。该目录还包含早期将八来源折叠成两 rank 的探索，不能把那组结果当成原始 R8 效果。

这些代码是独立研究适配器和微基准，没有修改生产默认行为，也没有完成八物理 GPU 的训练框架集成。下一步的启用条件仍是：在原八卡路由和实际训练生命周期下，同时计入完整 planner、专家刷新、梯度归并及同步，测得可重复的全链路正收益。
