"""Cold-per-update weight-aware training-segment replay, not GPU timing.

All methods pay for their own active seed replicas. The training extension is
explicitly assumed: backward token paths equal forward paths and backward GEMM
cost is twice forward GEMM. Optimizer/router/other layers are outside scope.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from simulate import Config, evaluate, distribution
from weight_cost import WeightConfig, evaluate_weights
from rail_support import select_support
from rail_planner import optimize, finalize
from run_planner import initial_support
from cost_aware import optimize_weight_neutral

ROOT = Path(__file__).resolve().parents[2]


def training_cost(token, weights, microbatches=1, planner_delta_us=0.0):
    """Barrier model plus a clearly labelled resource-service lower bound."""
    l = token['latency_us']
    w, g = (weights[p]['schedule'] for p in ('forward', 'gradient'))
    token_train = 2*l['dispatch'] + 3*l['compute'] + 2*l['combine']
    port_work = {}
    for phase, multiplier in ((w, 1), (g, 1),
                              (token['dispatch'], 2*microbatches),
                              (token['combine'], 2*microbatches)):
        for port, data in phase['ports'].items():
            port_work[port] = port_work.get(port, 0.0) + multiplier*data['busy_us']
    # This bound relaxes dependencies and compute/communication contention. It
    # is not an implementable overlap schedule and must not be called a gain.
    resource_lb = max(max(port_work.values(), default=0.0),
                      3*microbatches*l['compute'])
    return dict(microbatches_per_update=microbatches,
                token_training_segment_us_per_microbatch=token_train,
                weight_refresh_us=w['makespan_us'],
                replica_gradient_reduce_us=g['makespan_us'],
                incremental_planner_us_per_microbatch=planner_delta_us,
                common_seed_planner_us='unmeasured_common_term_omitted',
                serialized_update_us=w['makespan_us'] + g['makespan_us'] +
                    microbatches*(token_train + planner_delta_us),
                relaxed_resource_service_lower_bound_us=resource_lb + microbatches*planner_delta_us,
                lower_bound_is_achievable=False)


def aggregate_payload(token, weights, microbatches=1):
    t = token['traffic']
    # BF16 activation/input-gradient/output-gradient all share the assumed H.
    f = np.array(t['rdma_copies_src_node_rail_dst_node'], dtype=np.int64)
    v = np.array(t['nvlink_copies_node_src_local_dst_local'], dtype=np.int64)
    b = token['config']['hidden']*token['config']['input_bytes_per_element']
    rdma = 2*microbatches*b*(f + f.transpose(2, 1, 0))
    nv = 2*microbatches*b*(v + v.transpose(0, 2, 1))
    for p in ('forward', 'gradient'):
        rdma += np.array(weights[p]['traffic']['rdma_bytes_src_node_rail_dst_node'])
        nv += np.array(weights[p]['traffic']['nvlink_bytes_node_src_local_dst_local'])
    return dict(rdma_bytes_src_node_rail_dst_node=rdma.tolist(),
                nvlink_bytes_node_src_local_dst_local=nv.tolist(),
                rdma_payload_bytes=int(rdma.sum()), nvlink_payload_bytes=int(nv.sum()),
                rail_rdma_bytes=rdma.sum((0, 2)).tolist(),
                rail_statistics=distribution(rdma.sum((0, 2))),
                max_nic_port_bytes=int(max(rdma.sum(2).max(), rdma.sum(0).max())),
                max_nvlink_port_bytes=int(max(nv.sum(2).max(), nv.sum(1).max())))


def save_case(name, method, x, seed, owner, support, home, meta, digest,
              optimizer, output, config, wconfig, planner_delta, microbatches,
              schedule_seed=None, gate_us=2.0):
    final = finalize(x, owner, support, home)
    npz_path = output / f'{name}_{method}.npz'
    np.savez_compressed(npz_path, owner=owner, support=support,
                        input_npz_sha256=np.array(digest), method=np.array(method),
                        scenario=np.array(name), budget=np.array(8), padding=np.array(128), **final)
    copies = np.full((len(support), 8), -1, np.int32)
    for rank in range(len(support)):
        ids = np.flatnonzero(support[rank] & (home != rank))
        copies[rank, :len(ids)] = ids
    token = evaluate(dict(topk=x, owner=owner, home_rank=home, experts_to_copy=copies), meta, config)
    if schedule_seed is not None:
        from schedule_reuse import evaluate_reused_schedule
        token = evaluate_reused_schedule(x, schedule_seed, owner, home, support, meta, config)
    weights = evaluate_weights(x, owner, home, 4, 8, config, wconfig)
    delta = 0.0 if method == 'seed' else planner_delta
    if optimizer.get('precheck_no_home_imbalance', False):
        delta = gate_us
    costs = [training_cost(token, weights, m, delta) for m in microbatches]
    row = dict(scenario=name, method=method, input_npz_sha256=digest,
               output_npz=npz_path.name, optimizer=optimizer,
               token=token, weights=weights, training=costs,
               aggregate_payload=aggregate_payload(token, weights),
               planner_assumption='incremental GPU planner allowance, NOT a measurement; common seed cost cancels in absolute comparison',
               training_contract='per update: cold current home weights -> weight fill -> M sequential microbatches (forward dispatch/GEMM/combine, backward dispatch/2xGEMM/combine) -> replica gradient reduction; local gradient accumulation across M; no overlap',
               unchanged_plan=np.array_equal(seed, owner),
               changed_pairs=int(np.count_nonzero(seed != owner)))
    # Keep model phases/port service but omit bulky time-window bins.
    for phase in ('dispatch', 'combine'):
        row['token'][phase].pop('rail_payload_bytes_by_window', None)
    (output / f'{name}_{method}.json').write_text(json.dumps(row, indent=2)+'\n')
    compact = dict(scenario=name, method=method, output_json=f'{name}_{method}.json',
                   active_extra_copies=weights['active_extra_copies'],
                   weight_rdma_bytes=weights['forward']['traffic']['rdma_payload_bytes'],
                   weight_nvlink_bytes=weights['forward']['traffic']['nvlink_payload_bytes'],
                   weight_us=weights['forward']['schedule']['makespan_us'],
                   gradient_us=weights['gradient']['schedule']['makespan_us'],
                   token_rdma_copies=token['traffic']['rdma_token_copies'],
                   token_nvlink_copies=token['traffic']['nvlink_token_copies'],
                   token_rail_ratio=token['traffic']['rail_statistics']['max_over_mean'],
                   aggregate_rail_ratio=row['aggregate_payload']['rail_statistics']['max_over_mean'],
                   aggregate_rdma_bytes=row['aggregate_payload']['rdma_payload_bytes'],
                   aggregate_nvlink_bytes=row['aggregate_payload']['nvlink_payload_bytes'],
                   aggregate_max_nic_port_bytes=row['aggregate_payload']['max_nic_port_bytes'],
                   training=costs, changed_pairs=row['changed_pairs'])
    return compact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/multinode_rail/weighted_training')
    p.add_argument('--scenarios', nargs='+')
    p.add_argument('--methods', nargs='+', default=['seed','rail_b8_r1','rail_b8_r2','neutral_r1','neutral_r2','limited_b2','limited_b4'])
    p.add_argument('--planner-delta-us', type=float, default=128.0)
    p.add_argument('--gate-us', type=float, default=2.0)
    p.add_argument('--microbatches', nargs='+', type=int, default=[1,4,16,64])
    p.add_argument('--intermediate', type=int, default=2048)
    p.add_argument('--gradient-bytes', type=int, default=2)
    args = p.parse_args()
    if args.planner_delta_us < 0 or min(args.microbatches) < 1:
        p.error('nonnegative planner and positive microbatch count required')
    args.output.mkdir(parents=True, exist_ok=True)
    scenario_dir = ROOT/'outputs/multinode_rail/scenarios'
    previous = ROOT/'outputs/multinode_rail/planner_sparse'
    names = args.scenarios or [m['scenario'] for m in json.loads((scenario_dir/'manifest.json').read_text())['scenarios']]
    config = Config()
    wconfig = WeightConfig(intermediate=args.intermediate, gradient_bytes_per_element=args.gradient_bytes)
    rows = []
    for name in names:
        path = scenario_dir/f'{name}.npz'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as d:
            x, seed, home = d['topk'], d['owner'], d['home_rank']
        meta = json.loads(path.with_suffix('.json').read_text())
        seed_support = initial_support(x, seed, home)
        for method in args.methods:
            support = seed_support
            schedule_seed = None
            start = time.perf_counter()
            if method in ('seed','rail_b8_r1','rail_b8_r2','same_expert','bounded_existing'):
                with np.load(previous/f'{name}_{method}.npz') as d:
                    owner, support = d['owner'], d['support']
                diag = dict(scope='saved previously audited plan', source=str(previous/f'{name}_{method}.npz'))
            elif method.startswith('neutral_reuse_r'):
                owner, diag = optimize_weight_neutral(x, seed, rounds=int(method[-1]))
                schedule_seed = seed
            elif method.startswith('neutral_r'):
                owner, diag = optimize_weight_neutral(x, seed, rounds=int(method[-1]))
            elif method.startswith('limited_b'):
                budget = max(int(method.split('_b')[1]), int((support & (np.arange(32)[:,None] != home)).sum(1).max()))
                support, sd = select_support(x, seed, home, budget=budget, degree=min(budget,4), experts_per_edge=1)
                owner, diag = optimize(x, seed, support, edges=[(e['a'], e['b']) for e in sd['accepted_edges']])
                diag['support_selection'] = sd
            elif method in ('weight_aware','weight_aware_simple'):
                # This is a statistics-only precheck, not an oracle that runs
                # the candidate and then pretends a rejected attempt was free.
                demand = np.bincount(x.ravel(), minlength=len(home))
                home_load = np.bincount(home, weights=demand, minlength=32).astype(np.int64)
                if np.all(home_load == x.shape[1]*x.shape[2]) and np.all(seed == home[x]):
                    owner = seed.copy()
                    diag = dict(method='weight_aware', precheck_no_home_imbalance=True,
                                decision='keep home allocation; no speculative weight replication',
                                gate_us_assumed=args.gate_us)
                else:
                    from cost_aware import optimize_hierarchical
                    owner, support, diag = optimize_hierarchical(x, home, seed_owner=seed)
                    if method == 'weight_aware':
                        schedule_seed = owner
                        owner, nd = optimize_weight_neutral(x, schedule_seed)
                        diag['neutral_exchange'] = nd
                        diag['method'] = 'weight_aware_hierarchical_then_neutral_with_schedule_reuse'
            elif method.startswith('hierarchical'):
                from cost_aware import optimize_hierarchical
                owner, support, diag = optimize_hierarchical(x, home, seed_owner=seed,
                    allocation_order='rank_first' if method=='hierarchical_rank_first' else 'node_first',
                    mapping='fifo' if method=='hierarchical_fifo' else 'locality')
                if method == 'hierarchical_neutral':
                    owner, neutral_diag = optimize_weight_neutral(x, owner)
                    diag['neutral_exchange'] = neutral_diag
                support = initial_support(x, owner, home)
            else:
                raise ValueError(method)
            diag['cpu_reference_seconds_not_kernel_latency_including_load'] = time.perf_counter()-start
            row = save_case(name, method, x, seed, owner, support, home, meta, digest,
                            diag, args.output, config, wconfig, args.planner_delta_us, args.microbatches,
                            schedule_seed=schedule_seed, gate_us=args.gate_us)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('scenario','method','active_extra_copies','weight_us','gradient_us','changed_pairs')} |
                             {'train_m1_us': row['training'][0]['serialized_update_us']}), flush=True)
    for row in rows:
        base = next((b for b in rows if b['scenario']==row['scenario'] and b['method']=='seed'), None)
        if base is not None:
            for cost, bc in zip(row['training'],base['training']):
                cost['savings_vs_seed_us'] = bc['serialized_update_us']-cost['serialized_update_us']
                cost['savings_vs_seed_percent_common_planner_excluded'] = 100*cost['savings_vs_seed_us']/bc['serialized_update_us']
            dt = base['training'][0]['token_training_segment_us_per_microbatch']-row['training'][0]['token_training_segment_us_per_microbatch']-row['training'][0]['incremental_planner_us_per_microbatch']
            upfront = row['weight_us']+row['gradient_us']-base['weight_us']-base['gradient_us']
            row['break_even_same_version_repeated_workload_microbatches'] = max(1,int(np.floor(upfront/dt))+1) if dt>0 else None
            row['available_incremental_planner_budget_m1_us'] = base['training'][0]['serialized_update_us']-(row['training'][0]['serialized_update_us']-row['training'][0]['incremental_planner_us_per_microbatch'])
    sources = ['run_weighted.py','weight_cost.py','cost_aware.py','rail_planner.py','rail_support.py','simulate.py','schedule_reuse.py']
    summary = dict(scope='uncalibrated MoE training-segment CPU model, not whole training iteration or measured GPU speedup',
                   config=asdict(config),weight_config=asdict(wconfig),planner_delta_us_assumed=args.planner_delta_us,
                   microbatches_per_update=args.microbatches,
                   source_sha256={s:hashlib.sha256((Path(__file__).parent/s).read_bytes()).hexdigest() for s in sources},
                   rows=rows)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
