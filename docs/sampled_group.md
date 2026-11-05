# 八卡 sampled_group planner

`sampled_group` 把「256 个样本选择专家副本、每类保留两个专家组、全量 token 做四轮等量成组交换」接到 MoonEP 的正常 `Buffer.dispatch()` 路径。无需加载 router dump，也无需外部传入统计表。原生 `moon` 仍是默认值；较早的 `affinity_gpu` 是另一个算法。

```python
from moonep import Buffer

# 先按 MoonEP 原有方式初始化 NCCL，并给每个进程设置对应 CUDA 设备。
buffer = Buffer(
    S=local_tokens,
    H=hidden_size,
    K=8,
    E=352,
    num_ep_ranks=8,
    B=8,                       # 每张卡的额外专家槽数；默认 B=None 为 44
    planning_backend="sampled_group",
    planning_rounds=4,
    planning_sample_seed=20261026,
)
# hidden、topk、tokens_per_expert 都来自当前批次。
received, weights, cu_seqlens, plan, *rest = buffer.dispatch(
    hidden,
    route_weights_sk=router_weights,
    topk_experts_sk=topk,
    tokens_per_expert=tokens_per_expert,
)
```

启动脚本可把自己的命令行 `--planning-backend sampled_group` 传给上述构造参数。所有 EP 进程必须使用相同的后端、轮数、种子、S/E/K/B；一个 EP 组内不能混用。此版本限定同机 NVLink 八卡、E=352、K=8、每卡 S≥32、1≤B≤44，保留 MoonEP 对 hidden size、padding 等原有约束。一次性 CUDA 编译需要 CUDA toolkit 和 ninja；源码发行包与 wheel 都包含所需 CUDA 文件。部署前构造并预热，CUDA Graph 捕获期间不编译。

## 实际运行顺序

1. 执行本仓库的原生 MoonEP planner，得到平衡起点和必需副本。这个起点仍由实时路由产生。
2. 每卡只取自己的来源 token 中命中的样本。全组总共 256 个样本，卡间一次 SUM 汇总专家、执行卡和有效性信息，共 4,609 个 int32（18,436 字节）。索引在初始化时按种子确定；每次调用重读这些位置上的新路由和新执行计划。
3. 每卡根据同一批样本执行相同的全局副本选择。对每个方向、组大小和收益类别保留两个专家组，在全体卡对之间共享每卡 B 个额外槽的预算。原属专家与起点必需副本始终保留。B44 的选择历史和最多追加次数也扩展到 8B，不受原实验 64 次上限限制。
4. 每卡全量处理原本来自自己的 S 个 token。每个 token 每轮只提出一个交换建议。每轮真实汇总 1,152 个 int32 计数（每来源 4,608 字节），以全局来源顺序分配双向等量名额，再更新执行卡。四轮使用四次**当前**计数汇总，没有预录计数。
5. 复用刚才原生 C1 阶段按专家排好的计算任务顺序，只稳定排序 3 位目标卡编号，生成新的执行地址。汇总每来源 `[8,352]` 专家计数，分配不重叠的行和真实副本槽，生成 `dst/cu_seqlens/experts_to_copy/zero_fill_ranges/remote_stats`。
6. 所有卡汇总错误状态。副本缺失、容量不足等任一检查失败时，全组选择原生起点；否则使用新结果。随后清理并重新发布来源信息。正常 fresh dispatch 使用这些最终结果构建属于该 plan 的去重信息。

全量路由和全量 `dst` 不做 all-gather。副本选择是采样近似，交换与地址生成覆盖全部实际计算任务。kernel 复用驻留位图、每 token 的执行掩码、前缀计数和原生专家顺序；GPU 临时空间在 Buffer 构造时分配。

## 回退、训练和生命周期

可以在**非计时、非图捕获**的检查阶段读取：

```python
adapter = buffer._require_ctx()["sampled_group_planner"]
status = int(adapter.status.item())  # 0：使用新计划；非 0：全组退回原生计划
```

这属于诊断接口，不能把 `.item()` 放进训练热路径。局部详细原因分别在 `adapter.residency.status` 和 `adapter.materializer.status`。最终状态经 MAX 汇总，只表达“是否退回”，不保证保留全部原因位。

B=8 是受限物理副本预算。原生起点如果需要 B 个槽以外的专家，候选会被拒绝；回退只保留原生行为，不能使原生自身不支持的训练配置变得合法。一般训练仍应遵循原生的槽位容量要求，并检查自己的真实路由。

`plan.experts_to_copy` 直接沿用原生权重预取和副本梯度归并接口。当前版本在全局检查通过后才确定最终方案，没有新增“样本选完就提前发权重”的事件。若后续增加该功能，必须处理地址检查失败后的方案变更。

同一个 Buffer 的临时区只支持一次在途调用。前向后保存最终 `plan.clone()`、`cu_seqlens.clone()` 及不借用通信缓冲区的激活和路由权重，反向复用该计划。专家副本槽在反向前不能被别的层覆盖；若复用槽位，需恢复该前向对应的专家权重，并在覆盖副本梯度前完成归并。这些训练生命周期要求不会由新 planner 自动消除。

通信目的地数量下降并不等于训练加速。额外权重传输、每次更新后的刷新、副本梯度归并和完整 planner 时间都必须计入；此前八卡流量估算在部分训练设定下仍为负收益，因此此后端为显式启用。

## 验证与复现

配置检查不分配通信缓冲：

```bash
python -m pytest -q tests/test_sampled_group.py -k configuration
```

真实八卡公开接口验证：

```bash
MOONEP_MEM_HANDLE_TYPE=fd TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 \
  torchrun --standalone --nproc_per_node=8 -m pytest -q -s tests/test_sampled_group.py
```

测试要求八个不同的物理 GPU，包含动态输入、B8/B44、同步及异步 dispatch/combine、旧 plan 在新计划之后的前后向复用、dX/路由权重梯度、非默认进程组、单卡拒绝后的全局回退和更新输入后的 CUDA Graph 重放。测试先在所有 rank 完成原生 kernel 编译，再同步进入通信，避免首次编译导致设备内等待超时。

完整训练模型的 GEMM、optimizer、多层流水线及长期稳定性需要另行验证。实际执行结果、失败尝试和软件版本随本次提交的验证记录保存，不能把组件回放替代为八卡训练吞吐测试。

本次已执行的八卡测试、完整planner实测、保留失败及兼容性修复见 [验证记录](sampled_group_validation.md)。
