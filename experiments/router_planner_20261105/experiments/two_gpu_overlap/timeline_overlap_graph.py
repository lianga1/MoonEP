"""Eager/Graph cold-pipeline diagnostics, separate from accepted timing samples.

The existing timeline_overlap.py remains unchanged. Use this module's same-name
instrument_pipeline(..., execution='graph') to profile one warmed graph replay.
Timing events use external=True so capture creates real event-record nodes.
Internal tail events join BOTH diagnostic streams back to the capture origin;
external timestamp nodes alone do not establish CUDA capture fork/join edges.

PyTorch 2.11 event contract:
https://docs.pytorch.org/docs/2.11/generated/torch.cuda.Event.html
No diagnostics here measure NVLink byte occupancy or end-to-end model latency.
"""
from contextlib import nullcontext
import gc
import hashlib
from pathlib import Path

import torch
import torch.distributed as dist

from timeline_overlap import instrument_pipeline as eager_instrument_pipeline
from timeline_overlap import _intersection, analyze_kernel_overlap


PHASES = ('begin', 'native_plan_ready', 'expert_plan_ready',
          'weight_stream_start', 'weight_stream_end', 'token_plan_ready',
          'token_stream_start', 'token_stream_end', 'both_inputs_ready', 'combine_end')


def instrument_pipeline(*, native, decode, residency, materializer, exchange,
                        routes, hidden, pool, buffer, split, optimized=True, early=True,
                        trace_path=None, execution='eager', warmup=2):
    """Capture/profile only warmed execution, retaining exact identity checks.

    Every rank must invoke this function with identical mode/options. All JITs
    must already be compiled on both ranks. The graph owns NCCL work references
    and is reset before returning; the caller retains Buffer/weight ownership.
    Graph output is validated after the profiled replay, outside the trace.
    """
    if execution not in ('eager', 'graph'):
        raise ValueError("execution must be 'eager' or 'graph'")
    kwargs = dict(native=native, decode=decode, residency=residency,
                  materializer=materializer, exchange=exchange, routes=routes,
                  hidden=hidden, pool=pool, buffer=buffer, split=split,
                  optimized=optimized, early=early, trace_path=trace_path)
    if execution == 'eager':
        report = eager_instrument_pipeline(**kwargs)
        report['execution_mode'] = 'eager'
        return report
    if warmup < 1:
        raise ValueError('at least one eager warmup initializes event resources before capture')
    if int(buffer._require_ctx()['K']) != 8:
        raise ValueError('this audit expects exactly eight identity expert outputs')
    path = Path(trace_path) if trace_path is not None else None
    if path is not None:
        if path.exists():
            raise FileExistsError('choose a new diagnostic trace path; previous evidence is retained')
        path.parent.mkdir(parents=True, exist_ok=True)

    def drain_and_align():
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()

    # Keep timestamp event objects alive until all replays and elapsed-time reads
    # finish. Eager warmup records each event at least once, resolving lazy CUDA
    # event allocation before capture. These are markers, not dependency edges.
    stamps = {name: torch.cuda.Event(enable_timing=True, external=True) for name in PHASES}

    def mark(name, stream=None):
        stamps[name].record(torch.cuda.current_stream() if stream is None else stream)

    def prefetch(ids, ready):
        producer = torch.cuda.current_stream().record_event()
        split.expert_stream.wait_event(ready)
        split.expert_stream.wait_event(producer)
        mark('weight_stream_start', split.expert_stream)
        done = split.prefetch(ids, ready=ready, **pool.kwargs())
        mark('weight_stream_end', split.expert_stream)
        # The ordinary done event precedes the timestamp. A new INTERNAL event
        # after the timestamp is needed to join that final graph node as well.
        tail = split.expert_stream.record_event()
        return done, tail

    def body():
        mark('begin')
        plan, cu, ready = native()
        mark('native_plan_ready')
        if optimized:
            owner = decode()
            support = residency.run(plan.experts_to_copy, owner, local_routes=routes)
            ids, expert_ready = materializer.prepare_support(support)
            mark('expert_plan_ready')
            weight = prefetch(ids, expert_ready) if early else None
            final = exchange.run(owner, support=support)
            plan, cu, ready = materializer.run(final)
        else:
            mark('expert_plan_ready')
            weight = prefetch(plan.experts_to_copy, ready) if early else None
        mark('token_plan_ready')
        producer = torch.cuda.current_stream().record_event()
        split.token_stream.wait_event(ready)
        split.token_stream.wait_event(producer)
        mark('token_stream_start', split.token_stream)
        out = split.dispatch_fresh(hidden, None, plan=plan, cu_seqlens=cu, ready=ready)
        mark('token_stream_end', split.token_stream)
        token_tail = split.token_stream.record_event()
        if not early:
            # Same serialized weight-after-dispatch schedule as benchmark.py.
            weight = prefetch(plan.experts_to_copy, out.done)
        main = torch.cuda.current_stream()
        main.wait_event(out.done)
        main.wait_event(weight[0])
        main.wait_event(token_tail)
        main.wait_event(weight[1])
        mark('both_inputs_ready')
        result = buffer.combine(plan=plan, hidden_nvsh=out.hidden, zero_copy=True)[0]
        mark('combine_end')
        return result, plan

    graph = None
    captured_output = captured_plan = None
    complete = False
    try:
        drain_and_align()
        for _ in range(warmup):
            body()
        drain_and_align()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output, captured_plan = body()
        # This first replay resolves graph upload/initial replay effects. It is
        # outside the profiler, as are all eager initialization invocations.
        graph.replay()
        drain_and_align()
        profiler = (torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False) if path is not None else nullcontext())
        with profiler as prof:
            with torch.profiler.record_function('diagnostic_graph_replay_only'):
                graph.replay()
                # Synchronize the replay stream itself: the preceding warmup
                # also recorded these events, so do not rely on polling a
                # marker that might still describe that previous invocation.
                torch.cuda.current_stream().synchronize()
        base = stamps['begin']
        times = {name: base.elapsed_time(event) for name, event in stamps.items()}
        wr = [times['weight_stream_start'], times['weight_stream_end']]
        tr = [times['token_stream_start'], times['token_stream_end']]
        report = dict(rank=dist.get_rank(), optimized=optimized, early=early,
            execution_mode='graph', eager_warmup_invocations=warmup,
            unprofiled_graph_warmup_replays=1, profiled_or_observed_graph_replays=1,
            timing_events_external=True,
            scope='instrumented cold identity-expert Graph replay; excluded from benchmark samples',
            gpu_event_ms=times, weight_token_stream_interval_overlap_ms=_intersection(wr, tr),
            expert_copy_lead_over_token_plan_ms=times['token_plan_ready']-times['weight_stream_start'],
            expert_copy_exposed_after_dispatch_ms=max(0.0, wr[1]-tr[1]),
            caution='Timestamp nodes and profiler may perturb scheduling. Stream ranges include barriers; '
                    'kernel intersection establishes execution overlap, not simultaneous NVLink byte occupancy.',
            sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (Path(__file__), Path(__file__).with_name('timeline_overlap.py'))})
        if path is not None:
            prof.export_chrome_trace(str(path))
            report['profiler'] = analyze_kernel_overlap(path)
            report['trace_path'] = str(path)
        # Neither assertion kernels nor full-slot reads belong in the trace.
        torch.testing.assert_close(captured_output, hidden*8, rtol=0, atol=0)
        from moonep_common import check_weight_slots
        report['weight_slots'] = check_weight_slots(pool, captured_plan.experts_to_copy)
        if optimized:
            materializer.check_errors()
        report['identity_output_exact'] = True
        complete = True
    finally:
        # Release captured collective user objects while the process group and
        # all external mapped buffers are still alive. Never destroy the group
        # here; parent cleanup owns it. Avoid a new collective after a failure.
        if graph is not None:
            torch.cuda.synchronize()
            graph.reset()
            graph = None
        captured_output = captured_plan = None
        gc.collect()
        if complete:
            drain_and_align()
    report['graph_reset_before_return'] = True
    return report
