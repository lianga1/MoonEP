# 八卡可配置 sampled_group planner

`sampled_group` 将采样选择专家副本、全量 token 等量成组交换接入 `Buffer.dispatch()`。E（全局路由专家数）、K（每个 token 选择的专家数）、H（隐藏维度）直接复用模型配置；planner 自己的近似搜索参数另有默认值。原生 `moon` 仍是默认后端。

## 复用模型配置

```python
from moonep import Buffer, SampledPlannerConfig

# 先初始化 NCCL，并为每个进程设置对应 CUDA 设备。
buffer = Buffer.from_model_config(
    model.config,                  # 也可以传配置字典
    S=local_tokens,
    num_ep_ranks=8,
    planning_backend="sampled_group",
    planner_config=SampledPlannerConfig(
        sample_size=512,           # 全部八张卡合计请求的采样数
        # groups_per_class=2, rounds=4, sample_seed=20261026
    ),
)
received, weights, cu_seqlens, plan, *rest = buffer.dispatch(
    hidden,
    route_weights_sk=router_weights,
    topk_experts_sk=topk,
    tokens_per_expert=tokens_per_expert,
)
```

也支持 `planner_config={"sample_size": 512}`，便于将 YAML/JSON 或启动参数直接传入。省略 `planner_config` 使用下表默认值。旧式 `Buffer(S,H,K,E,8,...)` 保留，E/K/H 仍只从 Buffer 参数取一次；旧 `planning_rounds`、`planning_sample_seed` 参数保留，显式传入时不能与配置对象冲突。

| 参数 | 默认值 | 含义和范围 |
|---|---:|---|
| `sample_size` | 256 | 全组采样数，1–4096；小批次实际取 `min(sample_size,8*S)`，无放回采样 |
| `groups_per_class` | 2 | 每个方向、计算任务数、收益类别保留的专家组数量，1–8；不是每组的专家数 |
| `rounds` | 4 | 全量等量交换轮数，正整数 |
| `sample_seed` | 20261026 | 初始化采样位置的非负整数种子 |
| `candidate_limit` | 0 | 双向候选方案上限，非负整数；0 保留全部候选。正数按初始收益排序后截取，并随当前模型候选容量和实际候选数缩小 |

要启用本次交付的八卡、128 候选、四轮配置，使用 `B=8` 和 `SampledPlannerConfig(candidate_limit=128, rounds=4)`。详细启动示例、剪枝规则和证据见 [候选上限版本](sampled_candidate_limit.md)。样本数仍默认 256，每类仍保留两组；128 指最终双向方案的数量，不是采样数，也不是专家数量。

模型字段自动读取如下；支持对象或字典，也支持其 `text_config`。同一维度有多个值且冲突时会报错，避免猜错模型结构。

| 维度 | 自动识别字段 |
|---|---|
| H | `hidden_size`, `d_model`, `n_embd` |
| E | `num_experts`, `n_routed_experts`, `moe_num_experts` |
| K | `num_experts_per_tok`, `num_experts_per_token`, `moe_top_k` |

特殊模型可指定 `expert_field="text_config.num_local_experts"`、`topk_field="router.k"`、`hidden_field="model.width"`。`num_local_experts` 可能表示分片数，普通 `top_k` 可能是文本生成参数，因此不自动推断这两个字段。也可显式提供 `MoEModelConfig(hidden_size=..., num_experts=..., top_k=...)`。不同层 E/K 不同时，为每层提供具体配置和 Buffer；不把列表直接作为一个 E/K 使用。

## 支持范围与初始化

当前仍是同机 NVLink 八卡：R=8、8≤E≤4096、E 能被 8 整除、1≤K≤min(32,E)、S≥1。副本槽数 B 默认 E/8，范围 1≤B≤E−E/8；H 需能被 128 整除，padding 等继续遵守 MoonEP 原有约束。B 较小时必须保证原生起点本身可执行，不能靠回退修复一个原生也无法容纳的计划。

所有 EP 进程必须使用相同后端及模型/算法配置。构造时对 sampled_group 做一次配置一致性检查，避免有效但不相同的配置进入尺寸不同的通信。各进程必须共同调用相同构造入口（全部使用 `Buffer()` 或全部使用 `from_model_config()`）；模型解析入口另有一次错误汇总。不要让一部分进程选择另一后端。

配置在 Buffer 生命周期内固定。改变 E/K/S、样本数或保留组数等时构造新 Buffer 并预热。CUDA kernel 按需要的形状和源文件摘要编译、缓存，既不会在训练迭代中重新编译，也不支持直接修改一个已捕获 CUDA Graph 的配置。需要 CUDA toolkit 和 ninja；源码包和 wheel 包含 CUDA 源码。

## 算法顺序与可变布局

1. 原生 MoonEP planner 根据实时路由得到负载平衡的起点和必需副本。
2. 每卡读取其来源 token 中命中的采样位置，全组 SUM 汇总当前路由和执行卡。实际采样数为 D 时，采样数据包包含 `D*(2*K+2)+1` 个 int32；D=256/K=8 时仍是 4,609 个。种子固定的是位置，每次调用读取的是新批次的数据。
3. 从这些样本识别专家组。组是某个 token 在某执行卡上的专家集合，大小可为 1…K。每个方向、大小、收益类别保留 G=`groups_per_class` 组。双向等量组形成候选；`candidate_limit>0` 时，在 GPU 上按起点的收益/新增副本数排序并保留前若干个。随后在每卡 B 个副本槽预算下贪心选择，每次追加副本后重算当前收益。缓存只维护保留候选依赖的表项，但每个表项仍统计对应类别中的全部原始保留组。保留原属专家和起点必需副本。专家集合使用完整排序列表校验，哈希仅作快速过滤，不以哈希相等代替集合相等。
4. 每卡全量处理原本来自自己的 S 个 token，每轮每 token 最多一个交换建议。移动规模为小于 K 的 1、2、4…以及当前完整组；K=8 保留原 1/2/4/全组规则。汇总每来源 `2*8*8*(K+1)` 个 int32 计数，并按来源顺序分配双向等量名额。执行 `rounds` 轮，每轮使用真实当前计数。
5. 复用原生 C1 的专家排序，只稳定排序 3 位目标卡号。汇总每来源 `[8,E]` 专家计数，生成唯一执行行及真实副本槽，输出原生 `dst/cu_seqlens/experts_to_copy/zero_fill_ranges/remote_stats`。
6. 全组汇总错误状态，检查通过才发布候选；任一 rank 拒绝则全组选择原生起点，再发布来源信息。正常 dispatch 根据最终地址建立该 plan 自己的去重信息。

全量路由和 `dst` 不做 all-gather。采样只近似选择副本；全量交换与地址生成覆盖所有计算任务。K≤8 保留并行共享计数路径，K>8 用有界计数表，避免共享内存随着 warp 数和 K 同时膨胀。驻留集合长度随 E 变化，采样频数按固定大小分块，物理专家布局按块扫描。原生起始规划也按设备共享内存上限选择 512/256/128/64 线程，保留每块 2048 个计算任务的顺序；默认 E352 仍用 512 线程。B≤64 保留副本槽循环展开，大 B 使用循环控制编译体积。K≤32 的 token 掩码用无符号 32 位处理，包括第 32 个专家位。

这些可变布局避免为每个模型手工改源码，但不代表所有配置的性能相同。采样频数比较仍近似随 D² 增长，双向候选数量随 G² 增长；调大采样和保留组数需要衡量质量与延迟。配置支持范围与本次实测矩阵见验证记录，不把边界检查等同于全部组合已测试。

## 训练与生命周期

诊断阶段可读取 `buffer._require_ctx()["sampled_group_planner"].status.item()`：0 使用新计划，非 0 全组回退。不要在计时或 CUDA Graph 捕获中调用 `.item()`。局部原因见 `residency.status`、`materializer.status`。

`plan.experts_to_copy` 沿用权重预取和副本梯度归并接口；当前仍在全局检查通过后才确定最终计划。此次配置改造没有提前发权重的新事件。

Buffer 的临时区只支持一次在途调用。前向后保存最终 `plan.clone()`、`cu_seqlens.clone()` 及不借用通信缓冲的激活和路由权重，反向复用该计划。专家副本槽在反向前不能被别的层覆盖；若复用槽，需恢复对应前向的权重，并在覆盖梯度前完成归并。

通信目的地数量下降不等于训练加速。额外权重刷新、副本梯度归并、完整 planner 时间都必须计入。此后端保持显式启用；完整模型 GEMM、optimizer、多层流水线与长期稳定性仍需单独验证。

## 验证

```bash
python -m pytest -q tests/test_planner_config.py
MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 \
  torchrun --standalone --nproc_per_node=8 -m pytest -q -s tests/test_sampled_configurable.py
```

测试在所有 rank 预编译原生内核后才进入通信。旧版八卡默认配置测试和真实 router 回放继续保留。历史结果见 [上一版验证记录](sampled_group_validation.md)，本次变更与执行结果见 [可配置版本验证记录](sampled_configurable_validation.md)。
