# 真实 top-8 路由的八卡桶交换组件实测

2026-10-26，启智 omni-dev，NVIDIA H200。八卡全局计数协调版本已实际运行并通过正确性检查。在 reciprocal 驻留下，四轮桶交换从单卡 **326.884 μs** 降至八卡 **254.456 μs**，组件延迟减少 **22.16%**，加速 **1.285×**；同时完整保留相对 native MoonEP seed 的 **24.6166%** 远端去重 token 目的地降幅。

这里的“保留”有逐 `(token,expert)` 证据：八卡 coordinated 的最终 owner 与单卡 global CPU 参考完全相同。它不是仅靠相近通信总量推断等价。

## 测试范围与统计口径

- 输入：本次 router dump 的8层；每层8个source rank、每源16768个路由行、top-8、352专家，共134144行和1073152个计算对。每卡额外专家上限8。
- 驻留：`seed` 只保留原MoonEP均衡分配必需的专家；`visited` 根据seed已访问目的地的专家票数填充剩余槽；`reciprocal` 用4096个样本和每方向最多4组候选选择可双向交换的专家组。三种驻留均已在CPU准备，不计入本组件时间。
- `single`：1张物理GPU保存8个逻辑source的全量路由，匹配全局桶。
- `local`：8张物理GPU各处理一个source，桶只在分片内部匹配，无跨卡桶计数通信。
- `coordinated`：8张物理GPU各处理一个source，每轮汇总全局桶计数，匹配集合与single一致。
- 环境：PyTorch 2.13.0+cu130、CUDA 13.0、Python 3.12.3；三种模式均捕获一次完整 `run(seed)` 到CUDA Graph，包括seed到owner复制和4轮交换。每例预热5次、测25次。
- 时间：每次分布式重复取最慢物理rank，先求每层25次重复的中位数，再对8层算术平均；tail不进入下表。CUDA Event记录完整图重放设备时间线，coordinated包含NCCL等待；不是把各kernel执行时间简单相加。同步wall时间另列。
- 通信降幅：`1 - sum(各层最终远端目的地数) / sum(各层seed远端目的地数)`，使用8层合计量，**不是逐层百分比的算术均值**。该口径的reciprocal为24.6166%；逐层百分比均值为24.5853%，两者不能混写。

## 三种驻留 × 三种执行模式

|驻留|执行模式|物理GPU|图重放时间 μs|同步wall时间 μs|相对seed的远端目的地降幅|
|---|---|---:|---:|---:|---:|
|seed|single|1|295.236|309.886|4.6566%|
|seed|local|8|187.700|203.385|1.7196%|
|seed|coordinated|8|249.832|265.112|4.6566%|
|visited|single|1|320.052|334.684|17.5829%|
|visited|local|8|192.648|207.932|12.3016%|
|visited|coordinated|8|253.656|269.033|17.5829%|
|reciprocal|single|1|326.884|341.535|24.6166%|
|reciprocal|local|8|193.856|209.042|15.5456%|
|reciprocal|coordinated|8|254.456|269.856|24.6166%|

reciprocal的8层seed合计有5,054,165个远端 `(token,destination)`，single和coordinated均降到3,810,002，local为4,268,465。local更快，但放弃了一部分跨source可配对的候选，少省458,463个远端目的地。因此当前需要完整保留global收益时，应使用coordinated；不能把local的193.856 μs与global的24.6166%拼成一个结果。

reciprocal的逐层中位设备时间范围：single为318.240–332.576 μs，coordinated为253.056–258.624 μs，local为190.912–197.216 μs。同步wall口径下，single与coordinated分别为341.535和269.856 μs；其比例不能替代上文CUDA Event口径的1.285×。

## 为什么四次计数协调足够

每个token每轮只保留一条提案，提案记录方向、组大小、收益以及要移动的top-k位图。这样不同token的提交不会互相修改同一个token的目的地集合；正反方向接受相同数量、相同大小的组，精确保留各执行rank负载。

本实现保留16线程处理一个token，仅lane 0–7加载真实专家，lane 8–15仍参与ballot、shuffle与64个方向的候选评分。CUB按桶号稳定排序，本地桶序保持原token顺序。

桶数为 `B = 2 R² (K+1) = 1152`。每张GPU每轮贡献1152个int32，即4608 B（4.5 KiB）；AllGather后每卡得到36864 B（36 KiB）的计数。全局桶序号为：

`前面source rank在该桶中的计数之和 + 本地桶序号`。

source分片是连续且顺序固定的，所以这个序号与全量稳定排序完全一致。只汇总小计数，不需要传输提案列表或全量owner。每轮一次AllGather，4轮共4次；最终owner汇总只用于正确性核验，位于计时外。

旧项目的逐阶段协调原型沿rank-pair对角线顺序运行，两轮需要 `2 × (2R−3) = 26` 次AllGather；其文档为 [gpu_sharding.md](/Users/liangzuning/Documents/dev_env/EP/moe_solver_free/docs/gpu_sharding.md:19)。本次把所有方向的桶计数合并到每轮一次协调，减少的是同步阶段数。旧原型的100K/E256/top8/B32、两轮输入和算法与本次不同，**26→4是协议结构对照，不能把旧延迟直接作为本次加速基线**。

## 工作量、空间与并行化边界

令 `T` 为全局token数、`K` 为top-k、`R` 为逻辑rank数、`P` 为执行planner的物理GPU数、`I` 为交换轮数；`B=2R²(K+1)`；`d` 为桶号radix排序所需的固定趟数。本实现的掩码限制在一个机器字内，候选组仅为1/2/4和完整组，不枚举所有专家子集。

每卡每轮的保守工作量可写为：

`O((T/P)·(R K + R² + d + K) + P B)`。

其中，`R K`覆盖rank掩码构造，`R²`覆盖有限组大小的方向评分，`d`覆盖稳定radix排序，`K`覆盖提交；`P B`是每卡对已汇总计数计算总量和source前缀。I轮乘以I。当前固定 `R=8、K=8、P=8、I=4` 时，大规模token部分随T线性增长，没有token两两搜索的 `T²` 项；八卡将该部分分摊为每卡 `T/8`，同时保留4次collective的固定等待及小计数处理。

GPU工作存储约为 `O(TK/P + T/P + RE + PB)`：路由/owner、提案/排序临时区、驻留信息、汇总计数。正确性核验额外汇总全量owner的内存不属于运行时协议需求。每轮AllGather每卡接收的其他rank计数payload为 `(P−1)·B·4` 字节，8卡时32256 B；4轮129024 B，不含协议开销。不能由这些字节数推算NCCL固有延迟或声称八卡应获得8倍加速。

这个复杂度只针对固定驻留桶交换。CPU抽样、专家组聚合、reciprocal候选评分和驻留生成仍是另一部分工作，不能用本式宣称完整算法已全部GPU化或只需一次 `O(TK)` 扫描。

## 正确性、证据和剩余范围

三种Graph模式各完成25个case：8层×3种驻留，加一个每source 19行的tail，共 **75 cases，其中72个真实全量case、3个tail**。所有case均通过owner逐pair与对应CPU参考完全一致、驻留合法、负载保留、额外槽≤8、预测收益与独立重算一致。全量输入每个执行rank都是134144个计算对；tail仅要求保留其原输入负载，不宣称tail原本均衡。Graph重复执行后再次核验owner一致。

tail并非空操作：global改变101个计算对，local改变38个，覆盖非整块以及仅半warp有效的尾部。另有CPU独立协议审计40轮通过；单卡eager tail smoke保存在 `single.json`，未与Graph主表混算。没有据此宣称整个CUDA/NCCL路径通过sanitizer。

当前计时包含seed复制、提案、排序、计数、配额、提交以及coordinated的4次AllGather；**不含seed生成、驻留生成、输入路由搬运、support打包、物理row重编码、专家权重复制、dispatch/combine或梯度归并**。减少的token目的地由owner独立重算，未运行真实激活通信测出同幅度网络时间下降。

因此，不能把其他报告中约60 μs的native planner与这里254 μs的桶交换直接相加，就称为完整在线planner或训练端到端时间。还需要接入原生地址/row重编码和完整通信路径，并在同一连续调用中测量。当前已证明的是：**真实八卡可以加速这个固定驻留组件，同时保留global方案的执行分配和通信收益。** 本轮GPU测试已结束，GPU已释放；没有新增后台实验。

原始证据：

- [single Graph JSON](/Users/liangzuning/Documents/ChatGPT/EPMoE/experiments/bucket8/results/single_graph.json)、[local Graph JSON](/Users/liangzuning/Documents/ChatGPT/EPMoE/experiments/bucket8/results/local_graph.json)、[coordinated Graph JSON](/Users/liangzuning/Documents/ChatGPT/EPMoE/experiments/bucket8/results/coordinated_graph.json)。
- [机器可读汇总](/Users/liangzuning/Documents/ChatGPT/EPMoE/outputs/gpu/bucket8_summary.json)；原始日志位于同目录的 `bucket8_*log`，保留每rank/每次重复时间及输入、CUDA实现hash。
- [冻结源码清单](/Users/liangzuning/Documents/ChatGPT/EPMoE/experiments/bucket8/SOURCE_MANIFEST.json)。本次计时脚本benchmark.py的SHA256为 `bf1d1b14f3ce708c0f74133b52e7574788acba43596067f8dd00d5db61e3f9b2`；JSON中的CUDA/Python组件hash与当前清单一致。
