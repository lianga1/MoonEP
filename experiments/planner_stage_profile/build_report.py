#!/usr/bin/env python3
"""Build a Chinese report from measured planner JSON; standard library only.

python3 experiments/planner_stage_profile/build_report.py \
  --baseline /path/baseline.json --optimized /path/optimized.json \
  --baseline-trace-summary /path/baseline_trace_analysis.json \
  --optimized-trace-summary /path/optimized_trace_analysis.json \
  --output-dir outputs/planner_stage_profile

The source evidence is never modified. Missing or failed cohorts are visible.
Stage means use the same rank/sample records for every stage, so they add up.
They are instrumentation diagnostics, not the uninstrumented latency result.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
from urllib.parse import quote


NAMES = {
    "pre_sync": "起始卡间同步", "input_copy": "复制本卡路由输入",
    "native_seed": "原生 MoonEP 起点", "decode_owner": "解码执行卡",
    "residency_gather_local": "提取本卡样本", "sample_sum": "汇总八卡样本",
    "residency_select": "选择额外专家副本", "materialize_prepare_support": "建立专家物理槽表",
    "exchange_prepare": "准备交换支持表", "exchange_reset": "恢复交换起点",
    "materialize_prepare": "任务排序与本卡计数", "materialize_counts_allgather": "汇总最终任务计数",
    "materialize_layout_encode": "生成物理执行地址", "status_merge": "合并本卡校验状态",
    "status_max": "汇总八卡校验状态", "fallback_select": "选择新计划或原生回退",
    "publication_clear": "清空来源元数据", "publication_barrier_before": "发布前卡间同步",
    "publication_write": "发布来源元数据", "publication_barrier_after": "发布后卡间同步",
}
for _round in range(1, 5):
    for _suffix, _label in (("propose", "扫描 token 并提出交换"), ("counts_allgather", "汇总交换计数"),
                            ("quota", "分配双向交换名额"), ("apply", "应用获批交换")):
        NAMES[f"round{_round}_{_suffix}"] = f"第 {_round} 轮：{_label}"
KIND = {"kernel": "kernel", "collective": "NCCL 通信", "sync": "卡间同步", "glue": "辅助操作"}


def read(path):
    if not path:
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def esc(value):
    return html.escape(str(value), quote=True)


def number(value, digits=2):
    return "—" if value is None else f"{value:,.{digits}f}"


def mean(values):
    return statistics.mean(values) if values else None


def diff(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [name for key in sorted(set(left) | set(right))
                for name in ([path + "/" + key + "/missing"] if key not in left or key not in right
                             else diff(left[key], right[key], path + "/" + key))]
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path + "/length"]
        return [name for i, (a, b) in enumerate(zip(left, right)) for name in diff(a, b, path + f"/{i}")]
    return [] if left == right else [path]


def layer_map(doc):
    return {row["layer"]: row for row in doc.get("layers", [])}


def full_latency(doc, layers):
    rows, by_layer = [], layer_map(doc)
    for layer in layers:
        row = by_layer.get(layer, {})
        methods = row.get("uninstrumented", row.get("variants", row.get("timing", {})))
        # The dedicated full benchmark and stage harness share this method schema.
        if not methods:
            methods = row.get("latency", {})
        vals = {name: entry["max_rank_median_cuda_ms"] * 1000
                for name, entry in methods.items()
                if isinstance(entry, dict) and "max_rank_median_cuda_ms" in entry}
        rows.append(dict(layer=layer, values_us=vals, completed=row.get("completed"),
                         fixture=row.get("fixture")))
    keys = set.intersection(*(set(row["values_us"]) for row in rows)) if rows else set()
    return dict(layers=rows, mean_layer_us={name: mean([row["values_us"][name] for row in rows])
                                          for name in sorted(keys)})


def completed(doc):
    return (doc.get("status") == "complete" and not doc.get("failures") and bool(doc.get("layers"))
            and all(row.get("completed", row.get("post_timing_audit_passed", False)) for row in doc["layers"]))


def check_full_vs_stages(full, stage, label):
    for key in ("physical_ranks", "S", "N", "E", "K", "B", "rounds", "sample_seed"):
        if full["metadata"].get(key) != stage["metadata"].get(key):
            raise ValueError(f"{label}: full benchmark and stage benchmark disagree on {key}")
    for full_key, stage_key in (("samples", "uninstrumented_samples"), ("replays_per_sample", "uninstrumented_replays_per_sample")):
        if full["metadata"].get(full_key) != stage["metadata"].get(stage_key):
            raise ValueError(f"{label}: benchmark timing protocol differs on {full_key}")
    # Older full benchmark records omit these defaults. Their production
    # source hashes are checked below; newly recorded explicit values must
    # agree rather than silently overriding the diagnostic configuration.
    for key in ("global_samples", "groups_per_class"):
        if key in full["metadata"] and full["metadata"][key] != stage["metadata"].get(key):
            raise ValueError(f"{label}: full benchmark and stage benchmark disagree on {key}")
    fs, ss = full["metadata"].get("source_files", {}), stage["metadata"].get("source_files", {})
    if not any(name.startswith("moonep/") for name in fs) or not any(name.startswith("moonep/") for name in ss):
        raise ValueError(f"{label}: production source SHA manifest missing")
    changes = [name for name in sorted(set(fs) | set(ss))
               if name.startswith("moonep/") and fs.get(name) != ss.get(name)]
    if changes:
        raise ValueError(f"{label}: production code differs between full and stage benchmark: {changes}")
    for layer in set(layer_map(full)) & set(layer_map(stage)):
        if layer_map(full)[layer].get("fixture") != layer_map(stage)[layer].get("fixture"):
            raise ValueError(f"{label}: full/stage fixture differs for layer {layer}")


def compare_full_audits(base, opt, layers):
    results = []
    for layer in layers:
        b, o = layer_map(base)[layer], layer_map(opt)[layer]
        ba, oa = b.get("audit"), o.get("audit")
        mismatches = diff(ba, oa) if ba is not None and oa is not None else ["audit missing"]
        results.append(dict(layer=layer, audits_equal=not mismatches, mismatches=mismatches,
                            baseline=ba, optimized=oa))
    return results


def stage_means(doc, layers):
    rows, manifests, by_layer = [], {}, layer_map(doc)
    for layer in layers:
        entry = by_layer.get(layer, {})
        for stage in entry.get("stages", []):
            manifests[stage["name"]] = stage
        for sample in entry.get("instrumented", {}).get("raw", []):
            for rank in sample.get("per_rank", []):
                rows.append(dict(layer=layer, sample=sample["sample"], rank=rank["rank"],
                                 total_us=rank["total_cuda_ms"] * 1000,
                                 stages_us={k: v * 1000 for k, v in rank["stage_ms"].items()}))
    if not rows:
        return dict(records=0, stages=[], total_mean_us=None, closure_error_us=None)
    keys = set(rows[0]["stages_us"])
    if any(set(row["stages_us"]) != keys for row in rows):
        raise ValueError("Stage diagnostics have differing keys; refusing non-additive averages")
    if manifests and keys != set(manifests):
        raise ValueError("Stage diagnostic values differ from manifest")
    stages = [dict(name=name, label=NAMES.get(name, name),
                   mean_us=mean([row["stages_us"][name] for row in rows]),
                   kind=manifests.get(name, {}).get("kind", "unknown"),
                   index=manifests.get(name, {}).get("index", 999),
                   note=manifests.get(name, {}).get("note", "")) for name in keys]
    stages.sort(key=lambda x: (x["index"], x["name"]))
    by_kind = defaultdict(float)
    for stage in stages:
        by_kind[stage["kind"]] += stage["mean_us"]
    total = mean([row["total_us"] for row in rows])
    closure = max(abs(row["total_us"] - sum(row["stages_us"].values())) for row in rows)
    # These records also make the averaging population and additive check auditable.
    return dict(records=len(rows), record_keys=[{k: row[k] for k in ("layer", "sample", "rank")} for row in rows],
                stages=stages, total_mean_us=total, sum_stage_mean_us=sum(x["mean_us"] for x in stages),
                closure_error_us=closure, by_kind_mean_us=dict(by_kind))


def compare_outputs(base, opt, layers):
    result, bmap, omap = [], layer_map(base), layer_map(opt)
    for layer in layers:
        b, o = bmap[layer], omap[layer]
        bc, oc = b.get("correctness", {}), o.get("correctness", {})
        bh, oh = bc.get("output_hashes_by_rank"), oc.get("output_hashes_by_rank")
        mismatches = diff(bh, oh) if bh is not None and oh is not None else ["output hashes missing"]
        fixtures = b.get("fixture", {}).get("sha256") == o.get("fixture", {}).get("sha256")
        result.append(dict(layer=layer, fixture_equal=fixtures,
                           all_output_hashes_equal=not mismatches, mismatches=mismatches,
                           baseline_package_counts=bc.get("package_count_by_rank"),
                           optimized_package_counts=oc.get("package_count_by_rank"),
                           baseline_history_counts=bc.get("history_count_by_rank"),
                           optimized_history_counts=oc.get("history_count_by_rank")))
    return result


def trace_summary(doc, layers):
    if doc is None:
        return None
    pairs = [pair for pair in doc.get("pairs", []) if pair["layer"] in layers]
    table, metrics, unmapped = defaultdict(list), [], 0
    for pair in pairs:
        graph = pair["graph"]
        unmapped += graph.get("mapping_coverage", {}).get("unmapped", 0)
        m = graph["kernel_metrics"]
        metrics.append(dict(layer=pair["layer"], rank=pair["rank"],
                            sum_us=m.get("sum_us_per_replay"), union_us=m.get("union_us_per_replay"),
                            mean_replay_span_us=graph.get("mean_replay_kernel_span_us"),
                            nccl_us=graph.get("nccl_name_matched", {}).get("sum_us_per_replay"),
                            barrier_us=graph.get("explicit_barrier_stage_gpu", {}).get("sum_us_per_replay")))
        for row in graph.get("kernel_and_activity_table", []):
            if row.get("sum_us_per_replay") is not None:
                key = (row["stage"], row["name"], row["category"])
                table[key].append(dict(us=row["sum_us_per_replay"], calls=row["calls_per_replay"],
                                       method=row["attribution_method"]))
    # Denominator is every included rank/layer capture, so missing operations
    # are zeros and the table remains additive across captured GPU operations.
    kernels = [dict(stage=k[0], name=k[1], category=k[2],
                    mean_us=sum(x["us"] for x in v)/len(pairs),
                    calls_per_replay=sum(x["calls"] for x in v)/len(pairs),
                    attribution_methods=sorted({str(x["method"]) for x in v})) for k, v in table.items()]
    kernels.sort(key=lambda row: row["mean_us"], reverse=True)
    return dict(status=doc.get("status"), failures=doc.get("failures", []), pairs=len(pairs),
                unmapped_events=unmapped, per_rank_layer=metrics, kernels=kernels,
                means={key: mean([m[key] for m in metrics if m[key] is not None])
                       for key in ("sum_us", "union_us", "mean_replay_span_us", "nccl_us", "barrier_us")})


def source_for(stage):
    if stage == "native_seed":
        return "moonep/planning.py · launch_moon_planning"
    if stage.startswith("residency_"):
        return "moonep/csrc/sampled_residency.cu"
    if (stage.startswith("round") and not stage.endswith("allgather")) or stage.startswith("exchange_"):
        return "moonep/csrc/sampled_exchange.cu"
    if (stage.startswith("materialize_") and not stage.endswith("allgather")) or stage in ("decode_owner", "publication_write"):
        return "moonep/csrc/sampled_materialize.cu"
    if stage == "pre_sync" or stage.startswith("publication_barrier"):
        return "moonep/inter_rank_sync.py"
    if stage in ("sample_sum", "status_max") or stage.endswith("allgather"):
        return "PyTorch distributed / NCCL（包括等待其他卡）"
    return "moonep/sampled_group.py / sampled_materialize.py"


def table(headers, rows, css=""):
    return '<div class="scroll"><table class="' + css + '"><thead><tr>' + ''.join(
        '<th>' + esc(x) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + str(x) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def link(path, output_dir, label=None):
    relative = Path(os.path.relpath(path.resolve(), output_dir.resolve())).as_posix()
    return f'<a href="{esc(quote(relative, safe="/"))}">{esc(label or path.name)}</a>'


def kernel_name(name):
    visible = name if len(name) <= 170 else name[:167] + "…"
    return '<code title="' + esc(name) + '">' + esc(visible) + '</code>'


def make_report(summary, args):
    b, o = summary["baseline"], summary["optimized"]
    primary = summary.get("primary_full")
    bf, of = (primary["baseline"], primary["optimized"]) if primary else (b["full"], o["full"])
    primary_layers = primary["layers"] if primary else summary["comparison_layers"]
    bm, om = bf["mean_layer_us"], of["mean_layer_us"]
    old, new = bm.get("sampled_group"), om.get("sampled_group")
    speedup = old / new if old is not None and new else None
    reduction = (1 - new / old) * 100 if old and new is not None else None
    status = "完整" if summary["inputs_complete"] else "存在未完成或失败记录；本页不是完整验收"
    content = [f'<header><p class="eyebrow">MoonEP · 八卡实测 · planner 阶段诊断</p><h1>1.2 ms 花在哪里？</h1><p>本次输入状态：{esc(status)}。完整计时层：{esc(primary_layers)}；分阶段诊断层：{esc(summary["comparison_layers"])}。</p></header>']
    content += ['<div class="cards">' + ''.join(f'<div><span>{esc(label)}</span><strong>{esc(value)}</strong></div>' for label, value in [
        ("优化前完整 planner", number(old) + " μs"), ("优化后完整 planner", number(new) + " μs"),
        ("加速比", number(speedup) + "×"), ("耗时减少", number(reduction) + "%")]) + '</div>']
    content += ['<p class="notice">上述为<strong>没有插入阶段计时事件</strong>的完整 planner：每卡先取样本中位数，再取最慢卡，最后对所列层求平均。JIT、输入直方图、专家权重传输、token dispatch/combine、梯度传输、GEMM 和优化器均不在此计时内。它不是完整训练步耗时。</p>']
    content += ['<h2>170 μs 与 1.2 ms 为什么不能直接比较</h2>', table(
        ["项目", "此前两卡测试", "81a5a31 的八卡结果（历史实测）"], [
            ["物理卡 / 逻辑卡", "2 / 2；原八个来源合并为两组", "8 / 8；保留原八个来源"],
            ["交换轮数", "1", "4"], ["原生 planner", "203.496 μs", "67.709 μs"],
            ["完整新 planner", "372.686 μs", "1,208.038 μs"],
            ["相对原生增加", "169.190 μs（此前所说约 170 μs）", "1,140.329 μs"],
        ]), '<p>170 μs 是当时的<strong>新增开销</strong>，不是完整 planner。卡数、每卡 token 数、候选卡对和交换轮数也不同。两栏环境和采样协议不同，不用两者相减推算某个阶段耗时；下面用本次同一八卡环境拆分。</p>']
    if args.optimization_note:
        content += ['<h2>本次修改</h2><p>' + esc(args.optimization_note) + '</p>']
    content += ['<h2>本次完整耗时：每层独立比较</h2>']
    bmap, omap = {x["layer"]: x for x in bf["layers"]}, {x["layer"]: x for x in of["layers"]}
    layer_rows = []
    for layer in primary_layers:
        bv, ov = bmap[layer]["values_us"], omap[layer]["values_us"]
        bo, oo = bv.get("sampled_group"), ov.get("sampled_group")
        layer_rows.append([str(layer), number(bv.get("moon")), number(ov.get("moon")), number(bo), number(oo),
                           number(bo/oo if bo and oo else None) + "×"])
    content += [table(["层", "原生 / 优化前测量 μs", "原生 / 优化后测量 μs", "新 planner 修改前 μs", "修改后 μs", "加速"], layer_rows)]
    if summary["failed_attempts"]:
        content += ['<h2>保留未采用的尝试</h2><p>以下记录保留失败或回退的实际测量，未混入最终版本的平均值。每次尝试只与基线的相同层比较；回退原因尚未由专门实验确认。</p>', table(
            ["记录", "测量状态", "共有层", "同层基线 μs", "候选 μs", "候选相对变化", "说明"], [
                [link(Path(row["path"]), args.output_dir), esc(row["status"]), esc(row["layers"]), number(row["baseline_us"]), number(row["candidate_us"]),
                 number(row["change_percent"]) + "%", "未采用；原因未证实"] for row in summary["failed_attempts"]])]
    content += ['<h2>36 阶段拆分：同一记录集合的可加平均</h2><p>每个阶段都对相同的「层、采样次数、卡」记录求平均，因此表内可以相加；没有把各阶段分别取最慢卡再拼成虚假的总时间。这些数值含阶段事件开销，只用于定位，不代替上面的完整耗时。</p>']
    bs, os_ = b["stages"], o["stages"]
    bst, ost = {x["name"]: x for x in bs["stages"]}, {x["name"]: x for x in os_["stages"]}
    stage_rows = []
    for name in list(bst) + [x for x in ost if x not in bst]:
        before, after = bst.get(name, {}), ost.get(name, {})
        bu, ou = before.get("mean_us"), after.get("mean_us")
        stage_rows.append([esc(NAMES.get(name, name)), esc(KIND.get(before.get("kind"), before.get("kind", ""))),
                           number(bu), number(ou), number(bu-ou if bu is not None and ou is not None else None),
                           '<code>' + esc(name) + '</code>'])
    content += [table(["阶段", "类型", "修改前 μs", "修改后 μs", "减少 μs", "代码阶段名"], stage_rows),
                f'<p>诊断记录数：{bs["records"]} / {os_["records"]}；阶段均值之和：{number(bs.get("sum_stage_mean_us"))} / {number(os_.get("sum_stage_mean_us"))} μs；同记录整段均值：{number(bs["total_mean_us"])} / {number(os_["total_mean_us"])} μs。逐条记录加和误差最大 {number(bs["closure_error_us"], 4)} / {number(os_["closure_error_us"], 4)} μs。</p>']
    content += ['<h2>卡间通信：传的是哪些表</h2><p>NCCL kernel 时间含等待其他卡的时间；小消息通常还受启动与同步影响。下面列的是张量逻辑大小，不是实测 NVLink 链路字节。原生起点 kernel 内也含跨卡操作，无法仅靠这个分段计时拆成纯计算与纯传输。</p>', table(
        ["操作", "次数", "每卡输入字节", "每卡输出字节", "用途"], [[esc(row[k]) for k in ("name", "calls", "input_bytes", "output_bytes", "purpose")]
        for row in summary["collectives"]])]
    content += ['<h2>Profiler：kernel 与代码归属</h2><p>这是额外采集的 warmed Graph 时间线。kernel 时长相加会重复计算重叠；区间并集表示这张卡至少有一个 kernel 在运行的时间；首尾跨度还包含间隙。不同卡的时间轴没有合并。eager trace 只用来对应代码位置，不拿它的耗时替代 Graph。</p>']
    for label, entry in (("修改前", b["trace"]), ("修改后", o["trace"])):
        if entry is None:
            content += [f'<p>{label}：未提供 trace 分析，不推断 kernel 内部耗时。</p>']
            continue
        m = entry["means"]
        content += [f'<h3>{label}</h3><p>分析状态 {esc(entry["status"])}；覆盖 {entry["pairs"]} 个 rank/layer capture；失败 {len(entry["failures"])} 个；未归属事件 {entry["unmapped_events"]} 个。每次重放平均 kernel 总和 {number(m["sum_us"])} μs，区间并集 {number(m["union_us"])} μs，单次首尾跨度 {number(m["mean_replay_span_us"])} μs。NCCL 名称匹配时长 {number(m["nccl_us"])} μs；独立 barrier 时长 {number(m["barrier_us"])} μs。</p>']
        content += [table(["kernel / 操作", "阶段", "调用/次", "总时长/次 μs", "代码归属", "映射方式"], [
            [kernel_name(row["name"]), esc(NAMES.get(row["stage"], row["stage"])),
             number(row["calls_per_replay"], 1), number(row["mean_us"]), esc(source_for(row["stage"])),
             esc(", ".join(row["attribution_methods"]))] for row in entry["kernels"][:30]])]
        if len(entry["kernels"]) > 30:
            content += ['<p>展示耗时最高的 30 行；全部 kernel 及每卡分项在 summary.json 与原始 trace 分析中。</p>']
    content += ['<h2>重叠机会（分析，不等同已实现）</h2>', table(["位置", "数据依赖", "结论"], [
        ["副本选择 → 全量交换", "交换要读取最终副本表", "本卡这两步存在依赖，不能直接同时执行。"],
        ["各轮计数通信 → 名额与应用", "本轮名额依赖全部卡的计数；下轮依赖本轮新执行位置", "四轮通信不能简单并成一轮，跳过依赖会改变算法。"],
        ["专家权重预取", "副本表确定后可以独立准备权重；执行专家前必须等待完成", "这是训练集成的重叠机会；本次 planner 计时没有传权重，不能据此声称已测得训练收益。"],
    ]), '<h2>融合候选（待额外验证）</h2>', table(["候选", "可能节省", "必要条件"], [
        ["组装候选 / 前缀 / 填包", "减少小 kernel 启动", "共享内存和单 block 容量允许，候选顺序及 tie-break 必须一致。"],
        ["状态合并 / 回退选择", "减少辅助 kernel 启动", "保留全局拒绝传播、输出生命周期和完整计划的一致性。"],
        ["小统计表的直接 NVLink 汇总", "可能减少小消息通信启动开销", "需独立 mailbox、正确的 release/acquire 和复用屏障；须实测优于 NCCL。"],
    ])]
    content += ['<h2>正确性：跨版本逐项比对</h2><p>比对的是保存的所有输出字段及中间状态摘要，包括最终任务地址、owner、support、有效候选包、完整有效 history、校验状态与发布元数据。数值相同才显示一致；缺失证据不会当作通过。</p>', table(
        ["层", "输入 SHA 一致", "所有输出 SHA / 标量一致", "差异路径"], [
            [str(row["layer"]), "是" if row["fixture_equal"] else "否", "是" if row["all_output_hashes_equal"] else "否 / 缺失", esc(", ".join(row["mismatches"]) or "无")]
            for row in summary["correctness"]])]
    if primary:
        content += ['<h3>完整计时队列的逐层审计</h3><p>这里比较完整基准保存的 owner SHA、负载、通信计数、副本计数及物理布局检查结果。它不替代上述代表层对全部中间状态的比较。</p>', table(
            ["层", "完整审计跨版本一致", "差异路径"], [[str(row["layer"]), "是" if row["audits_equal"] else "否 / 缺失", esc(", ".join(row["mismatches"]) or "无")]
            for row in primary["correctness"]])]
    content += ['<h2>环境与证据</h2><pre>' + esc(json.dumps(summary["environment"], ensure_ascii=False, indent=2)) + '</pre><ul>']
    for label, path in (("优化前分阶段记录", args.baseline), ("优化后分阶段记录", args.optimized),
                        ("优化前 trace 分析", args.baseline_trace_summary), ("优化后 trace 分析", args.optimized_trace_summary),
                        ("优化前完整基准", args.full_baseline), ("优化后完整基准", args.full_optimized)):
        if path:
            content += ['<li>' + link(path, args.output_dir, label) + '</li>']
    content += ['<li><a href="summary.json">本报告汇总 JSON</a></li></ul>']
    if args.commit:
        content += ['<p>代码：<a href="https://github.com/lianga1/MoonEP/commit/' + esc(args.commit) + '">' + esc(args.commit) + '</a></p>']
    for label, doc in (("优化前", b), ("优化后", o)):
        if doc["failures"]:
            content += ['<h3>' + label + ' 保留的失败记录</h3><pre>' + esc(json.dumps(doc["failures"], ensure_ascii=False, indent=2)) + '</pre>']
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MoonEP planner 分阶段诊断</title><style>' + CSS + '</style><body><main>' + ''.join(content) + '</main></body></html>'


CSS = """
:root{color-scheme:light;font-family:system-ui,-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#19312d;background:#f4f6f3}*{box-sizing:border-box}body{margin:0}main{max-width:1260px;margin:auto;padding:40px 26px 80px}header{border-bottom:2px solid #247b69;padding-bottom:22px;margin-bottom:24px}h1{font-size:38px;margin:8px 0 14px;letter-spacing:-1px}h2{margin-top:38px;font-size:24px}h3{font-size:18px}.eyebrow{color:#247b69;font-size:14px;letter-spacing:.06em}p,li{line-height:1.75}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.cards>div{background:white;border:1px solid #d7e1db;border-radius:10px;padding:22px}.cards span{display:block;font-size:14px;color:#5b6e66}.cards strong{display:block;margin-top:10px;font-size:27px;font-variant-numeric:tabular-nums}.notice{background:#e5eee9;border-left:4px solid #247b69;padding:15px 20px}.scroll{overflow:auto;background:white;border:1px solid #d7e1db;border-radius:7px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;border-bottom:1px solid #e3e8e2;padding:10px 12px;vertical-align:top;line-height:1.55}th{background:#e5eee9;white-space:nowrap}tr:last-child td{border-bottom:0}td{font-variant-numeric:tabular-nums}code{font-size:11px;overflow-wrap:anywhere}a{color:#186b58}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf0ec;padding:16px;font-size:12px}@media(max-width:760px){main{padding:24px 14px 48px}.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:30px}.cards strong{font-size:22px}}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--baseline-trace-summary", type=Path)
    parser.add_argument("--optimized-trace-summary", type=Path)
    parser.add_argument("--full-baseline", type=Path, help="Optional all-layer benchmark_sampled_group JSON")
    parser.add_argument("--full-optimized", type=Path, help="Must accompany --full-baseline")
    parser.add_argument("--failed-attempt", type=Path, action="append", default=[],
                        help="Retain an unadopted stage/full attempt; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/planner_stage_profile"))
    parser.add_argument("--optimization-note", default="")
    parser.add_argument("--commit")
    args = parser.parse_args()
    if bool(args.full_baseline) != bool(args.full_optimized):
        parser.error("--full-baseline and --full-optimized must be supplied together")
    base, opt = read(args.baseline), read(args.optimized)
    common = sorted(set(layer_map(base)) & set(layer_map(opt)))
    if not common:
        raise ValueError("No shared layers; refusing an unmatched comparison")
    contract = ("physical_ranks", "S", "N", "E", "K", "B", "rounds", "global_samples", "groups_per_class",
                "sample_seed", "uninstrumented_samples", "uninstrumented_replays_per_sample")
    for label, doc in (("baseline", base), ("optimized", opt)):
        missing = [key for key in contract if doc.get("metadata", {}).get(key) is None]
        if missing:
            raise ValueError(f"{label}: required protocol metadata missing: {missing}")
    mismatches = {k: [base["metadata"].get(k), opt["metadata"].get(k)] for k in contract
                  if base["metadata"].get(k) != opt["metadata"].get(k)}
    if mismatches:
        raise ValueError(f"Comparison protocol differs: {mismatches}")
    for layer in common:
        bsha = layer_map(base)[layer].get("fixture", {}).get("sha256")
        osha = layer_map(opt)[layer].get("fixture", {}).get("sha256")
        if not bsha or bsha != osha:
            raise ValueError(f"Layer {layer} fixture identity missing or different")
    meta = base["metadata"]
    r, e, k, d, rounds = (int(meta[x]) for x in ("physical_ranks", "E", "K", "global_samples", "rounds"))
    sample_bytes, counts_bytes, layout_bytes = (d*(2*k+2)+1)*4, r*r*2*(k+1)*4, r*e*4
    summary = dict(comparison_layers=common,
        inputs_complete=all(completed(doc) for doc in (base, opt)),
        unmatched_layers={"baseline_only": sorted(set(layer_map(base))-set(common)),
                          "optimized_only": sorted(set(layer_map(opt))-set(common))},
        metric_contract=dict(primary="Per-layer maximum over rank medians, then mean of compared layers; uninstrumented Graph.",
            stages="Arithmetic mean of the identical rank/sample record population for each stage; instrumentation diagnostics.",
            trace="Profiled GPU sums, interval unions and per-replay spans are separately reported; never sum across rank clocks."),
        environment={key: meta.get(key) for key in (*contract, "gpu", "torch", "cuda", "nccl", "devices")},
        correctness=compare_outputs(base, opt, common), failed_attempts=[],
        collectives=[dict(name="样本 SUM", calls=1, input_bytes=sample_bytes, output_bytes=sample_bytes, purpose="所有卡共享同一采样集"),
            dict(name="每轮交换计数 AllGather", calls=rounds, input_bytes=counts_bytes, output_bytes=r*counts_bytes, purpose="决定互相等量的交换名额"),
            dict(name="最终任务计数 AllGather", calls=1, input_bytes=layout_bytes, output_bytes=r*layout_bytes, purpose="生成物理布局与来源前缀"),
            dict(name="状态 MAX", calls=1, input_bytes=4, output_bytes=4, purpose="任一卡拒绝则全部卡回退")])
    for name, doc, path, trace_path in (("baseline", base, args.baseline, args.baseline_trace_summary),
                                      ("optimized", opt, args.optimized, args.optimized_trace_summary)):
        summary[name] = dict(source=dict(path=str(path.resolve()), sha256=sha(path)), status=doc.get("status"),
                             full=full_latency(doc, common), stages=stage_means(doc, common),
                             trace=trace_summary(read(trace_path), common), failures=doc.get("failures", []),
                             metadata=doc.get("metadata", {}))
    if args.full_baseline:
        fb, fo = read(args.full_baseline), read(args.full_optimized)
        check_full_vs_stages(fb, base, "baseline")
        check_full_vs_stages(fo, opt, "optimized")
        fl = sorted(set(layer_map(fb)) & set(layer_map(fo)))
        if not fl or set(layer_map(fb)) != set(layer_map(fo)):
            raise ValueError("Full benchmark layer cohorts differ or are empty")
        for layer in fl:
            if layer_map(fb)[layer].get("fixture") != layer_map(fo)[layer].get("fixture"):
                raise ValueError(f"Full benchmark fixture differs for layer {layer}")
        summary["inputs_complete"] &= completed(fb) and completed(fo)
        summary["primary_full"] = dict(layers=fl, baseline=full_latency(fb, fl), optimized=full_latency(fo, fl),
            correctness=compare_full_audits(fb, fo, fl),
            baseline_source=dict(path=str(args.full_baseline.resolve()), sha256=sha(args.full_baseline)),
            optimized_source=dict(path=str(args.full_optimized.resolve()), sha256=sha(args.full_optimized)),
            baseline_failures=fb.get("failures", []), optimized_failures=fo.get("failures", []))
    for path in args.failed_attempt:
        attempt = read(path)
        layers = sorted(set(layer_map(attempt)) & set(layer_map(base)))
        before = full_latency(base, layers)["mean_layer_us"].get("sampled_group")
        candidate = full_latency(attempt, layers)["mean_layer_us"].get("sampled_group")
        summary["failed_attempts"].append(dict(path=str(path.resolve()), sha256=sha(path), status=attempt.get("status"),
            layers=layers, baseline_us=before, candidate_us=candidate,
            change_percent=100*(candidate/before-1) if before and candidate is not None else None,
            failures=attempt.get("failures", []), cause="Not established by a dedicated causal experiment."))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "report.html").write_text(make_report(summary, args))
    print(json.dumps(dict(output=str(args.output_dir.resolve()), layers=common,
                          inputs_complete=summary["inputs_complete"],
                          outputs_equal=all(x["all_output_hashes_equal"] for x in summary["correctness"]))))


if __name__ == "__main__":
    main()
