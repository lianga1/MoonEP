# MoonEP 原生 planner 八卡优化实测

日期：2026-10-26。机器：启智 `omni-dev`，8 张物理 NVIDIA H200。

结论：当前原生 planner 已经使用八卡协作。保持全部路由与驻留决策不变，C2 shared LUT 在默认 B=44 上得到小幅收益：八层中位延迟的等权平均由 **69.640 → 66.988 µs，降低 3.81%**。B=8 上平均近中性。A5 active-count 虽减少无效 argmax 扫描，当前整核实测平均收益不足 1%，有多层回退，暂不建议默认启用。两项优化都不会降低 token 或专家通信量。

## 协议与边界

- 输入：dump 的全部 8 层，保留源 rank、token、top-k 原始顺序；每层 `[8,16768,8]`，共 134,144 tokens / 1,073,152 计算对，E=352。
- 设置：8 个进程各使用一张物理 H200，`num_sms=32`、`token_padding=128`、`replica_target=0`；分别比较 B=44、B=8。
- PyTorch 2.13.0+cu130、CUDA 13.0；本机安装的 CuTe DSL 为 4.6.2，而包内 setup.py 声明的是 4.4.2。本次使用现有环境，只原地构建 NVLink 扩展，没有将版本差异混同为源码优化收益；结论限于此实际环境。
- top-k 已在 GPU，tokens-per-expert 已预计算。测量包括一个原生 cooperative planning kernel 的 A/B/C/D 阶段及其内部跨卡同步；不包括编译、CPU/GPU 输入拷贝、tpe 计算、dispatch/combine、权重预取、梯度同步、专家 GEMM 或训练端到端。
- 先编译、warmup、capture，再进行 9 组配对轮次；每组每实现 20 次 CUDA Graph replay。每张卡用 CUDA Event 测量每次 replay 均值，每组取八卡最大值，最后取 9 组中位数。基线/优化顺序逐组交替。
- 每个候选与自己同一轮实验的原版基线比较；LUT 与 A5 两次实验的基线略有漂移，不交叉混算。这里的平均是八层中位数的等权算术平均，降低比例为 `1 - 优化均值/基线均值`，并非训练吞吐提升。

## 汇总

| 版本 | B | 基线平均 µs | 优化平均 µs | 平均延迟降低 | 层间降低范围 |
|---|---:|---:|---:|---:|---:|
| LUT | 44 | 69.640 | 66.988 | +3.81% | -0.23% ～ +8.28% |
| LUT | 8 | 63.852 | 63.910 | -0.09% | -1.51% ～ +3.00% |
| A5 active-count | 44 | 68.705 | 68.130 | +0.84% | -2.50% ～ +6.05% |
| A5 active-count | 8 | 63.812 | 63.356 | +0.71% | -0.38% ～ +2.65% |

正数表示更快，负数表示回退。未做跨日期或多进程重启的统计置信区间，因此亚百分比变化视为近中性，不据此宣布稳定加速。

## B=44 逐层配对结果

| 层 | LUT 基线→优化 µs | 降低 | A5 基线→优化 µs | 降低 |
|---|---:|---:|---:|---:|
| 00 | 80.925 → 78.813 | +2.61% | 80.117 → 81.254 | -1.42% |
| 01 | 68.781 → 68.942 | -0.23% | 70.118 → 69.898 | +0.31% |
| 02 | 67.144 → 65.930 | +1.81% | 66.072 → 66.126 | -0.08% |
| 03 | 69.672 → 65.346 | +6.21% | 68.330 → 67.373 | +1.40% |
| 04 | 66.830 → 64.597 | +3.34% | 64.437 → 66.046 | -2.50% |
| 05 | 64.858 → 64.610 | +0.38% | 63.082 → 63.922 | -1.33% |
| 06 | 69.582 → 63.822 | +8.28% | 69.565 → 65.357 | +6.05% |
| 07 | 69.330 → 63.845 | +7.91% | 67.923 → 65.062 | +4.21% |

## 正确性与 CPU replay 可信度

两版 × 8 层 × 2 个 B 共 32 个候选配置全部通过以下核验。每个配置都是八张物理卡共同执行，而非单卡模拟八 rank：

- 原版与优化版逐项 bitwise 一致：dst（含去重负数编码）、physical_topk、experts_to_copy、cu_seqlens、zero_fill_ranges、remote_stats、src_info；rank0 另外比较 alloc、group_tokens 和 z。
- CUDA Graph 重放后的输出与首轮快照完全一致。
- 独立重建 physical-slot 查找表核验物理专家映射；所有计算对地址合法，全局输出槽无冲突，每目的 rank 精确 134,144 计算对。
- 对每个原始 `(source rank, token, top-k index)`，原生 owner、physical_topk 与 `analysis/moonep_cpu.py` 逐对一致；复制清单和 rank0 alloc 也一致。
- 本次 rt=0 的 B44/B8 全部 `physical_topk >= 0`。这只证明这些 dump 输入可执行，不能证明 B=8 对任意路由或额外 retention 都可行。

因此，新的 NumPy replay 已通过实际八卡 native kernel 校验，可以用于本次 dump 的算法质量比较；它依然是 CPU 语义模型，其运行时间不是 GPU planner 延迟。

## 两项实现

**C2 shared LUT**：原版对离开专家 home 的计算对遍历 B 个复制槽；新版本每 CTA 建立 `[destination,expert]` 物理槽表，后续每对 O(1) 查询。实际只有 9.07%–21.41% 计算对离开专家 home，不能用“源 token 远端通信比例”替代这个比例。

设每 rank 计算对数为 M，离 home 的计算对数为 M_remote，CTA 数为 C：旧查找工作为 O(M_remote×B)，新版本为 O(C×R×(E+B)+M)，每 CTA 新增 11,264 bytes shared memory。建表重用 Phase D 的两次 CTA barrier，无新增 kernel、grid barrier、跨卡 barrier 或 metadata 布局变化。shared memory 增长和建表成本会抵消部分收益。

**A5 active-count**：先并行将 B 个空槽写为 -1，再只执行 `min(B,A_d)` 次原 argmax，A_d 为该目的 rank 的正分数远端专家数，保留原有排序和 tie-break。总选择工作由 O(R×B×E) 降为 O(E×Σ_d min(B,A_d)+R×B)，未新增 shared memory 或 barrier。虽然工作量减少，本批整核延迟收益很小，说明局部扫描减少不能直接当作总执行时间等比例减少。

这两个补丁都不改变专家驻留、配额或 token 分配算法，因此没有通信量优化收益；不应与另一个桶交换算法的通信降幅合并。

## 修复、失败与保留证据

初次 smoke 在导入阶段失败：脚本目录中的优化 `moonep/` 副本抢先于 PYTHONPATH，被当作主包加载，找不到原版已编译的 `_C`。已在 benchmark 显式将 `inputs/MoonEP` 放到 sys.path 首位并断言 baseline 源码绝对路径，优化仅通过独立模块名加载。失败日志 `lut_smoke.log` 保留，失败轮没有计入性能统计。

原始无 `.git` 导出的 build 日志包含 `git rev-parse` 提示；setup.py 捕获该情况后扩展成功完成编译，该提示不代表构建失败。

- 运行入口与补丁：`experiments/moonep_lut/benchmark_pair.py`、`run_pair.sh`、`shared_lut.patch`、`sparse_select.patch`。
- LUT 原始数据：`outputs/gpu_lut_full/summary.json` 与逐层 B44/B8 JSON。
- A5 原始数据：`outputs/gpu_sparse_select/summary.json` 与逐层 B44/B8 JSON。
- 日志：`outputs/gpu/native_build.log`、`lut_smoke.log`、`lut_smoke_v2.log`、`lut_full.log`、`sparse_select.log`。
- 运行前八卡空闲状态：`outputs/gpu/pre_full_gpu.csv`。
- 源码哈希记录于每份 summary 的 metadata；输入 NPZ 哈希记录于各配置。

## 选择建议

默认 B44 可保留 shared LUT 作为小幅、语义等价的可选优化；目前没有依据把它称为大幅加速。B8 保留原实现即可，当前 LUT 近中性。A5 active-count 留作实验候选，不默认启用。要继续降低在线开销，应结合完整 planner 各阶段的时间与新增算法的驻留/交换成本判断，不能只看单个复杂度项。
