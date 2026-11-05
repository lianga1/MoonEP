# 可配置 sampled_group 验证（2026-11-05）

本次基于 `45c2432a8da6f465028e833464f0ec25379cce0e`，将 E/K/H 接入模型配置，将全局采样数、每类保留组数、交换轮数和随机种子整理为带默认值的配置。驻留、交换、执行地址生成和原生起始规划的相关布局均已参数化。

验证实例为用户指定的 CI-情境智能 / omni-dev，八个不同的 NVIDIA H200。环境为 PyTorch 2.13.0+cu130、CUDA 13.0、CuTe DSL 4.6.2。保留实际日志、版本及源文件摘要，见 `experiments/configurable_planner_validation_20261105/`。

## 验证范围

每个运行 case 的所有八张卡分别持有其来源 token，实时采样汇总、实时交换计数汇总、真实 NVLink dispatch/combine。检查每个计算对恰好执行一次、各卡精确等负载、执行专家有物理槽、地址/去重/填零/来源信息正确，且更换路由后能复用旧 plan。测试使用可独立核验的合成 local/hot 路由；原始 router dump 另做默认设置回归，不将合成输入当作真实模型路由。

| E | K | 请求采样数 | 每类保留组数 | 轮数 | 每卡 token | B |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 37 | 1 | 1 | 17 | 8 |
| 128 | 3 | 64 | 2 | 2 | 33 | 16 |
| 256 | 4 | 128 | 3 | 2 | 65 | 32 |
| 352 | 8 | 256 | 2 | 4 | 257 | 8 |
| 520 | 7 | 37 | 1 | 2 | 65 | 65 |
| 512 | 16 | 512 | 2 | 4 | 65 | 64 |
| 1024 | 32 | 128 | 2 | 2 | 33 | 128 |
| 4096 | 1 | 256，实际 8 | 1 | 1 | 1 | 512 |
| 64 | 1 | 4096 | 8 | 1 | 513 | 8 |
| 8 | 8 | 37 | 2 | 2 | 17 | 1 |

E512/K16 另有更换输入的 CUDA Graph 重放；E512/K16 和 E1024/K32 另有带路由权重前向、旧计划反向、精确 dX 与每计算对路由权重梯度检查。原生权重预取和副本梯度归并的默认 E352/K8 测试继续执行。

独立单卡采样组件对照使用 E520/K3/D37/G3、E1024/K7/D128/G8、E520/K16/D512/G3、E1024/K32/D128/G8。每组使用两个结构化混合路由输入，包含正收益副本选择；逐项对照 NumPy 的完整中间表、support 和选择历史。还强制所有哈希相同，验证完整专家集合校验仍能得到相同频数。这个单卡测试只验证算法组件，不代替上述八卡运行。

## 执行结果

所有完成的验证命令退出码均为 0，独立回读也通过：

| 验证 | 实际结果 |
|---|---|
| 配置、模型字段复用和默认兼容 | 13 项通过 |
| 十配置八卡矩阵＋四项初始化一致性检查 | 每个 rank 14 项通过，八个不同 H200 |
| 默认公开 API、Graph、回退、真实专家权重预取与副本梯度归并 | 每个 rank 6 项通过 |
| 四种形状的独立采样 oracle、Graph 换输入、强制哈希碰撞 | 单 GPU 4 项通过 |
| K16/K32 带权重前向、保存旧计划后的反向 | 每个 rank 2 项通过；含同步/异步路径 |
| 原始八层 trace 组件回放 | 驻留表、每轮 owner、统计值和动态 Graph 回放逐项一致 |
| 原始八层 trace 的完整八卡 planner | 所有物理布局审计通过，8 层均无回退 |

完整 planner 计时使用每层 10 组、每组 5 次重放，先对每个 rank 取中位数，再取最慢 rank，最后跨八层求均值。独立脚本从保留的原始轮次重新计算：

| 指标 | 原生 MoonEP | 此次可配置 sampled_group |
|---|---:|---:|
| planner 平均每层 | 67.709 μs | 1208.038 μs |
| 八层 remote token-destination 总数 | 5,054,165 | 3,791,665 |
| 八层计划专家副本总数 | 71 | 507 |

默认设置的每层最终 owner 摘要与上一提交逐项一致；远端 token 目的地减少 **24.9794%**。上一提交同协议记录的 sampled_group 为 1286.090 μs/层，此次观察为 1208.038 μs/层；这是两次运行的记录，不作为稳定性能提升或训练加速承诺。此次 sampled_group 仍明显慢于原生 planner，且上表不含专家/token 载荷传输、GEMM 或 optimizer 时间。

保留 `matrix_01`：前七组配置通过后，在 E4096/B512 的原生 JIT 阶段主动停止（退出码 1，日志含 SIGTERM）。原因是副本槽循环完全展开导致编译体积过大，并非数值测试失败。修复为 B≤64 保留展开、B>64 使用循环后，`matrix_02` 的十组配置与四项初始化检查全部通过，八个 rank 报告 97 秒左右完成。首轮原始测试文件也保留，避免把后补的第十组算作首轮覆盖。

## 复现与边界

先构建 MoonEP 基础扩展，使用相同环境设置：

```bash
export MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 OMP_NUM_THREADS=1
python -m pytest -q tests/test_planner_config.py
python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -x -q -s tests/test_sampled_configurable.py
python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -x -q -s tests/test_sampled_group.py tests/test_sampled_weights.py
python -m pytest -x -q -s tests/test_sampled_residency_configurable.py
python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -x -q -s tests/test_sampled_configurable_weighted.py
```

默认 router 回放与完整 planner 计时继续使用 `tests/sampled_components.py` 和 `tests/benchmark_sampled_group.py`，输入是原始八层、每层 N=134144/E352/K8 的 fixture。前者两个 GPU 各持四个原始来源分片，仅检查组件；后者是真实八卡、每进程只读本卡来源路由，计入样本/计数/元数据集体通信及最终计划生成。

首次 JIT 编译时间不属于 steady-state planner 时间。公开接口测试与组件对照都不等同于完整模型训练：尚未以本次参数化版本验证真实 GEMM、optimizer、多层流水线或长期训练吞吐；不承诺所有 E/K/D/G 组合性能相同，也不将减少 token 通信等同于训练总收益。
