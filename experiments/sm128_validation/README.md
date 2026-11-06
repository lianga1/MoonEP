# 128SM 配置验证交付

本次验证原样使用算法提交 `55564bcbe1febd82d99b35ae7db0e16c5ccba712`，没有修改运行时代码，也没有改变默认配置。算法实现、CUDA kernel 和测试已经包含在仓库中；本目录补齐128SM配置的实测证据和独立复算入口。

在 omni-dev 的同机八张 H200 上，保持候选上限128、四轮交换、256个全局样本、每类两组，八层完整 planner 平均延迟为：

| 配置 | 延迟 | 相对本批32SM对照 |
|---|---:|---:|
| 32SM选择性融合，前后对照中点 | 763.182 µs | 基准 |
| 128SM未融合 | 629.366 µs | 降低17.53% |
| 128SM选择性融合 | 615.976 µs | 降低19.29% |

完整计划保持一致，远程 `(token, 目的卡)` 次数比原生 MoonEP 减少23.0801%，专家副本数量不变。这里测的是 planner，包含其跨卡元数据通信，不包含实际 token、权重、梯度传输和完整训练步。详细阶段、测量口径和正确性检查见 [报告](report.md)。

## 启动配置

八个进程先初始化通信组、设置各自 CUDA 设备，随后使用同一配置：

```python
from moonep import Buffer, SampledPlannerConfig

buffer = Buffer.from_model_config(
    model.config,
    S=local_tokens,
    num_ep_ranks=8,
    B=8,
    planning_backend="sampled_group",
    planner_config=SampledPlannerConfig(
        candidate_limit=128,
        rounds=4,
        sm_limit=128,
        fuse=True,
    ),
)
```

本次形状为 `E=352, K=8, S=16768/rank`，八卡合计134144个token；八层使用同一批路由数据。模型维度仍从模型配置读取，支持范围见 [配置说明](../../docs/sampled_group.md)。设备须至少有128个SM。更改配置后须重新构造、预热并捕获 CUDA Graph。

128是单个在途 planner 的同时执行线程块上限，由此限制同时使用的SM数；不固定物理SM编号，也不承诺实际占用128个SM。原生 MoonEP 起点仍受 `Buffer.num_sms=32` 限制。当前32SM配置启用八卡频率分治，128SM配置恢复每卡完整频率计算、减少一次频率表汇总，因此这不是仅改变SM数量的对照。具体约束见 [SM预算说明](../../docs/sampled_sm_budget.md)。

## 独立复算

在仓库根目录运行；只需要 Python 3 标准库，无需 GPU、PyTorch 或私有 router NPZ：

```bash
python3 experiments/sm128_validation/verify_delivery.py --check-sources
```

脚本先检查交付文件和运行时/测试源码的哈希，在临时目录解包原始结果，重新解析预算和阶段 trace，再与保存的摘要、报告逐字节核对。`--check-sources`针对当前工作区；若已有其他算法修改，可省略此选项，仅复核已冻结的实测证据。脚本不会修改仓库文件。

也可在独立目录复制本目录内容，解包 `evidence.tar.gz` 后执行 `python3 analysis.py`；报告末尾描述的是这一解包后的分析入口。`summary.json.gz`保留原始摘要字节，解压后可直接阅读。

文件说明：

- `evidence.tar.gz`：五组八层结果、原始计时、预算/阶段trace、退出状态、源码检查和机器状态；不包含私有路由NPZ或模型权重。
- `refs/`：冻结的CPU参考、原802计划、上一批32SM阶段摘要及解析器；独立复算不依赖其他实验目录。
- `source_manifest.json`：测量源码和测试的62项哈希；其中 `base=802af3f` 是早期候选包的构建起点，**本次实际运行版本为 `run_context.json` 中的55564bc**，文件哈希与该提交一致。
- `run_validation.sh`：实际远端执行命令的记录，保留当时环境、私有数据和路径，重新做GPU测试前需要按环境调整；它不是任意机器可直接运行的安装脚本。
- `validation_receipt.json`：本次原始核验凭据；其中 `summary.json` 的哈希对应压缩摘要解压后的字节。
- `delivery_manifest.json`：本次交付包的完整文件哈希，另列测量源码哈希。

原始归档 SHA-256：`ffedba0057dcb7dc099e1d08eb9d43dd2af4bbdb9e65635a7b731042ef11afa2`。
