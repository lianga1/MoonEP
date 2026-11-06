# 完整 planner 的 SM 上限与分段融合

本目录交付候选上限 128、四轮交换的完整八卡 planner 实测证据。正式结果见 [summary.md](summary.md)，逐卡中位数、逐层延迟、完整复核状态见 [summary.json](summary.json)。机器是同一台 omni-dev 的八张 H200，每卡 132 个 SM。

## 使用方式

```python
from moonep import Buffer, SampledPlannerConfig

config = SampledPlannerConfig(
    candidate_limit=128,
    rounds=4,
    sm_limit=32,
    fuse=False,
)
buffer = Buffer(
    S, H, K, E, R,
    planning_backend="sampled_group",
    planner_config=config,
)
```

`sm_limit=0, fuse=False` 保留原有实现。正数上限覆盖采样收集、专家选择、四轮交换、布局生成、验证与结果发布。`fuse=False` 先应用上限；`fuse=True` 再启用分段融合。上限约束每个 launch 的线程块数量，并让同一 CUDA stream 串行执行各阶段；线程块循环完成原来所有逻辑 tile，没有截断任务、减少候选或改变交换轮数。

这里“32 SM 上限”指 planner 同时发出的线程块最多 32 个，因此同时占用的 SM 数不会超过 32。它不固定 SM 编号、不保证恰好占用 32 个 SM，也不意味着为其他计算预留了固定编号的 SM。八卡通信使用同一 stream 上、受相同上限控制的 peer 表交换；本次不包括跨节点通信。

严格32SM时，本批次未融合868.30µs，分段融合895.52µs，因此示例选用 `fuse=False`。融合已实现并验证，设置 `fuse=True` 可启用；当前没有受限配置快于旧版约686µs。

## 正式矩阵及统计口径

仅以下 `*_03` 文件进入正式性能表：

| 文件前缀 | 上限 | 分段融合 | 原始 GPU trace |
|---|---:|---|---|
| baseline_before_03 | 无限制 | 原版 | 第 0 层，八卡各 3 次 Graph replay |
| bounded32_03 | 32 | 否 | 同上 |
| fused32_03 | 32 | 是 | 同上 |
| fused48_03 | 48 | 是 | 同上 |
| fused64_03 | 64 | 是 | 同上 |
| fused96_03 | 96 | 是 | 同上 |
| fused128_03 | 128 | 是 | 同上 |
| baseline_after_03 | 无限制 | 原版 | 无；用于时间漂移与完整输出复核 |

原版指提交 `407c223c04d7e7cb335a3db09650401c9c6d62cb` 的 sampled planner，不是只有原始 MoonEP 负载均衡。每份记录还单独测量 native MoonEP，供结果与开销区分。

每配置覆盖 8 层，各层 10 个样本，每个样本平均 5 次 CUDA Graph replay。先对每张卡的 10 个样本取中位数，再取八卡中的最大值，最后平均八层。不得把“每轮先取八卡最大值后求中位数”混作同一指标。样本本身已除以 5，不再重复除法。主表中的对照值是 before/after 两份原版结果的算术平均；另保留前后漂移、单独相对 before/after 的结果，不能把这视为随机实验或置信区间。

计时包括 native seed、采样和计数通信、候选筛选、专家选择、全量 token 交换规划、物理布局、合法性验证、回退选择和 source metadata 发布。计时不包含直方图生成、分配、JIT、审计、专家权重、token 数据、梯度数据、GEMM 或 optimizer；不能据此声称完整训练获得同样比例的加速。

通信收益是从完整计划精确统计的 remote token-destination 数量：一个 token 访问一张远程卡记一次，同张卡执行多个专家不重复计数。它不是计算对数，也不是实测网络耗时。原版 sampled 与各 SM/融合版本必须逐位一致，因此本次不能通过牺牲通信效果获得更低延迟。专家副本总数按八层求和；每层每卡最多八个远程副本，SM/融合调整不能增加副本。

## 独立复核

交付的证据目录是 `evidence/`，JSON 支持原始 `.json` 或 `.json.gz`，同名两种格式同时存在会失败。验证不需要 GPU、PyTorch、私有 router NPZ 或联网：

```bash
python3 experiments/sm_budget/summarize.py \
  --check-sources \
  --output experiments/sm_budget/summary.json \
  --markdown experiments/sm_budget/summary.md
```

首次整理证据时，维护者明确加 `--write-manifest` 生成 `evidence_manifest.json`；正常验收不加这个参数，以免覆盖已封存的文件哈希。下载目录不同可用 `--evidence-dir /path/to/evidence`。脚本遇到缺失文件、不完整层数/样本数、错误配置、错误 CPU oracle、任意不一致输出、回退、测试失败、超出上限或缺失 trace 锚点都会报错。

复核覆盖：

- 逐轮逐卡的原始 CUDA/host 时间与所有派生中位数；8 层完整矩阵。
- 每个配置、每层、每张卡的九类输出数组哈希与 baseline_before 完全一致；专家副本、token 通信矩阵、计数和完整候选诊断表一致。
- 独立 CPU oracle 中的 support、四轮 owner、greedy history、候选 ID 哈希一致。Graph 与 eager 一致；正常正式运行没有 native fallback。
- 原始 `.json.gz` trace 的压缩前后哈希；独立计算每个 kernel 的 grid、所有 kernel 时间区间重叠时的线程块数量上界、NCCL kernel 数、采样/选专家/四轮交换/布局/发布锚点。不只相信 benchmark 内的 `passed`。
- 03 轮配置、组件、八卡 public 与兼容性测试的原始日志和退出码。分布式日志有八份 pytest 汇总，不重复计算为八倍测试用例。
- 原版前后源码哈希一致，各新版本源码哈希一致；`--check-sources` 另外校验交付源码与实际测量的新版本源码一致。

`evidence_manifest.json` 包含证据目录内全部文件、三轮 `implementation_attempts/payload_*.tar.gz` 源码快照的 SHA-256。`summary.json` 记录原始 router 文件的 SHA-256 与 CPU oracle 哈希；私有 NPZ 不随仓库公开，不能在未获得它们时重新运行真实 trace benchmark。已有哈希可核对数据身份，但无法取代重新生成输出数组的 GPU 审计。

## 保留的探索记录

`probe*_01`、`probe*_02`、`probe*_03`、对应 trace、启动日志，以及早期测试记录全部保留。它们包含第一轮实现、修正与短测过程；样本、层数或源码可能不同，均不进入正式性能表，也不替换 03 轮完整矩阵。原始 profiler 中的时间用于检查 launch 与阶段边界，正式延迟来自没有 profiler 的 CUDA-event 测量。
