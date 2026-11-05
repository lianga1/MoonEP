# sampled_group 原生集成验证（2026-11-05）

本次在 **CI-情境智能 / omni-dev 的 8 张 H200** 上完成原生集成测试。启用方式为 `Buffer(..., planning_backend="sampled_group")`；默认仍是 `moon`。原生 `Buffer.dispatch()` 实际调用新后端，使用当前输入产生计划。

## 正确性结果

| 验证 | 实际范围 | 结果 |
|---|---|---|
| 八卡公开接口 | 每卡257 tokens，E352/K8，B8/B44，动态输入，同步/异步，旧计划跨批次复用 | 8个rank各5项通过 |
| token 前后向 | 实际dispatch/combine，标量线性专家参考，输入梯度和router权重梯度 | 与参考逐元素一致 |
| 回退与图 | 单rank拒绝→全组回退；非默认进程组；同一CUDA Graph更换路由重放 | 通过 |
| 专家权重和梯度 | 八卡，B44，两个不同热点专家集合和权重版本，三个独立BF16投影，FP32副本梯度 | 8个rank各1项通过 |
| 双卡组件回归 | 两张H200各负责四个原始来源；真实样本SUM和每轮计数all-gather；同一图跨八层换输入 | 全部support、副本、逐轮owner和统计逐项一致 |
| 真实trace八卡计划 | N134144，S16768，E352，K8，B8，d256，g2，四轮，全部八层 | 所有地址唯一、专家段合法、负载守恒；零回退 |

八卡环境：PyTorch 2.13.0+cu130、CUDA 13.0、CuTe DSL 4.6.2、NCCL 2.29.7。双卡组件环境为 PyTorch 2.11.0+cu130；这两类证据分别记录。

## 完整规划函数时间与流量

使用CUDA Graph。每层、每种方法采10个样本，每样本5次重放；先取每rank的样本中位数，再取八rank最大值，最后对八层平均。原始各rank、各样本值均保留。

| 指标 | 原生 MoonEP | sampled_group |
|---|---:|---:|
| 平均每层规划函数 | 68.024 μs | 1,286.090 μs |
| 八层规划函数之和 | 0.544 ms | 10.289 ms |
| 八层远程(token,目标卡)份数 | 5,054,165 | 3,791,665 |
| 八层计划专家副本数 | 71 | 507 |

远程token目的地减少 **24.9794%**；每层规划额外增加约 **1.218 ms**。新后端的全部八层状态均为0，没有以回退结果冒充优化收益。

计时包括原生起点、真实样本汇总、GPU副本选择、四轮实时计数汇总及交换、物理地址生成、全组状态检查、来源信息发布，以及两边共同的规划前同步。**不包括**外部专家计数直方图、dispatch中的去重结构构建、token/权重实际传输、梯度负载、GEMM或optimizer。因此本表不代表端到端训练加速。

此前训练成本模型已显示额外权重和梯度成本可能抵消流量下降，本次完整规划开销也不能忽略；保持显式启用是有意的。完整多层模型、optimizer和长期运行的正确性及速度，仍需针对部署训练框架验证。

## 保留的失败与修复

`native8_weights_attempt02`：公开接口5项仍通过，新增专家梯度测试在编译阶段失败。原仓库 `grad_reduce.py` 使用的 `cute.make_fragment` 在当前CuTe版本中已移除。改为仓库其他kernel已使用的 `cute.make_rmem_tensor` 后，`weights_attempt03` 八个rank全部通过，包含实际权重拷贝和梯度归并，没有跳过失败功能。

完整planner计时发生于该一行兼容性修复之前，其执行路径不调用梯度归并。计时记录保留当时 `grad_reduce.py` 的旧哈希；最终源码哈希及修复后的测试另行保留。其余参与规划的新模块与计时记录的哈希一致。

## 证据和复现

证据目录：[experiments/native_sampled_validation_20261105](../experiments/native_sampled_validation_20261105/)。保留首轮成功、失败尝试、修复后结果、完整计时JSON、组件回归JSON、最终源码哈希和独立重算记录。

在已准备好依赖的环境构建当前源码，可使用 `python setup.py build_ext --inplace`，避免改变现有CuTe/PyTorch版本。仓库原依赖声明仍保留CuTe 4.4.2；本次实测环境是上面的4.6.2，不把两者混称。

```bash
# 原生八卡及专家训练通信检查
MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 \
  torchrun --standalone --nproc_per_node=8 -m pytest -q -s \
  tests/test_sampled_group.py tests/test_sampled_weights.py

# 完整八卡 planner 计时；fixture可复用上次提交的原始实验包
MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 \
  torchrun --standalone --nproc_per_node=8 -m tests.benchmark_sampled_group \
  --fixture-root experiments/router_planner_20261105/outputs/eight_rank_estimate/kernel_inputs \
  --output planner_new_attempt.json

# 两物理卡、八原始来源的独立组件回归
PYTHONPATH=. TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 \
  torchrun --standalone --nproc_per_node=2 tests/sampled_components.py \
  --fixture-root experiments/router_planner_20261105/outputs/eight_rank_estimate/kernel_inputs \
  --output components_new_attempt.json
```
