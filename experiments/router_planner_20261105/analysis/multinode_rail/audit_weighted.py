"""Independent replica payload/port audit; imports no cost or planner model.

The exact accounting audit validates declared topology and payload semantics,
not hardware latency, numerical expert execution, or training equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_planner import audit_arrays, audit_physical_layout, exact_traffic, require


ROOT = Path(__file__).resolve().parents[2]


def close(a, b, message):
    require(bool(np.isclose(a, b, rtol=1e-10, atol=1e-7)), message)


def audit_weight_result(topk, owner, home, result, config, *, nodes=4, gpus=8,
                        initial_resident=None):
    """Rebuild each directed expert-transfer edge using Python sets."""
    ranks, experts = nodes * gpus, len(home)
    counts = np.zeros((ranks, experts), np.int64)
    np.add.at(counts, (owner.reshape(-1), topk.reshape(-1)), 1)
    active = counts > 0
    homes = np.arange(ranks)[:, None] == home[None, :]
    initial = homes if initial_resident is None else np.asarray(initial_resident)
    missing, extra = active & ~initial, active & ~homes
    wc = result['weight_config']
    parameters = config['hidden'] * wc['intermediate'] * wc['matrices']
    require(result['parameters_per_expert'] == parameters, 'weight parameter count')
    require(result['weight_bytes_per_expert'] == parameters * wc['bytes_per_element'], 'weight byte count')
    require(result['gradient_bytes_per_expert'] == parameters * wc['gradient_bytes_per_element'], 'gradient byte count')
    require(result['active_extra_copies'] == int(extra.sum()), 'active extra copies')
    require(result['missing_active_copies'] == int(missing.sum()), 'missing active copies')
    require(result['initial_active_warm_extra_copies'] == int(np.count_nonzero(extra & initial)), 'warm extra copies')
    require(np.array_equal(result['missing_copies_per_rank'], missing.sum(1)), 'missing per rank')
    require(np.array_equal(result['active_extra_copies_per_rank'], extra.sum(1)), 'extras per rank')
    checks = {}
    for phase, mask, bytes_per in (
            ('forward', missing, parameters * wc['bytes_per_element']),
            ('gradient', extra, parameters * wc['gradient_bytes_per_element'])):
        rdma_groups, nv_groups = set(), set()
        for target, expert in zip(*np.nonzero(mask)):
            source = int(home[expert])
            sn, sl = divmod(source, gpus)
            dn, dl = divmod(int(target), gpus)
            if dn != sn:
                rdma_groups.add((int(expert), sn, sl, dn))
            if dl != sl:
                nv_groups.add((int(expert), dn, sl, dl))
        rdma = np.zeros((nodes, gpus, nodes), np.int64)
        nv = np.zeros((nodes, gpus, gpus), np.int64)
        for _, sn, sl, dn in rdma_groups:
            a, b = (dn, sn) if phase == 'gradient' else (sn, dn)
            rdma[a, sl, b] += bytes_per
        for _, dn, sl, dl in nv_groups:
            a, b = (dl, sl) if phase == 'gradient' else (sl, dl)
            nv[dn, a, b] += bytes_per
        traffic, sched = result[phase]['traffic'], result[phase]['schedule']
        fields = {
            'rdma_bytes_src_node_rail_dst_node': rdma,
            'nvlink_bytes_node_src_local_dst_local': nv,
            'nic_tx_bytes_node_rail': rdma.sum(2),
            'nic_rx_bytes_node_rail': rdma.sum(0).T,
            'nvlink_tx_bytes_node_local': nv.sum(2),
            'nvlink_rx_bytes_node_local': nv.sum(1),
            'rail_rdma_bytes': rdma.sum((0, 2)),
        }
        for name, expected in fields.items():
            require(np.array_equal(traffic[name], expected), f'{phase} {name}')
        for kind, edge_set, matrix in (('rdma', rdma_groups, rdma), ('nvlink', nv_groups, nv)):
            require(traffic[kind + '_payload_bytes'] == int(matrix.sum()), f'{phase} {kind} payload')
            require(sched[kind + '_payload_bytes'] == int(matrix.sum()), f'{phase} scheduled {kind} payload')
            chunks = len(edge_set) * ((bytes_per + wc['chunk_bytes'] - 1) // wc['chunk_bytes'])
            require(sched[kind + '_chunks'] == chunks, f'{phase} scheduled {kind} chunks')
        peak_nic = int(max(rdma.sum(2).max(), rdma.sum(0).max()))
        peak_nv = int(max(nv.sum(2).max(), nv.sum(1).max()))
        require(traffic['max_nic_port_bytes'] == peak_nic, f'{phase} peak NIC')
        require(traffic['max_nvlink_port_bytes'] == peak_nv, f'{phase} peak NVLink')
        payload_lb = max(peak_nic / (config['rdma_gbytes_per_second'] * 1e3),
                         peak_nv / (config['nvlink_gbytes_per_second'] * 1e3))
        close(traffic['payload_port_lower_bound_us'], payload_lb, f'{phase} payload bound')
        # Independent byte+startup busy-time calculation for each port.
        port_busy = {}
        for kind, edge_set in (('rdma', rdma_groups), ('nvlink', nv_groups)):
            if bytes_per == 0:
                continue
            chunks = (bytes_per + wc['chunk_bytes'] - 1) // wc['chunk_bytes']
            service = bytes_per / (config[kind + '_gbytes_per_second'] * 1e3)
            service += chunks * config[kind + '_startup_us']
            for edge in edge_set:
                if kind == 'rdma':
                    _, sn, local, dn = edge
                    a, b = (dn, sn) if phase == 'gradient' else (sn, dn)
                    keys = (f'rdma_tx/{a}/{local}', f'rdma_rx/{b}/{local}')
                else:
                    _, node, src, dst = edge
                    a, b = (dst, src) if phase == 'gradient' else (src, dst)
                    keys = (f'nvlink_tx/{node}/{a}', f'nvlink_rx/{node}/{b}')
                for key in keys:
                    port_busy[key] = port_busy.get(key, 0) + service
        require(set(port_busy) == set(sched['ports']), f'{phase} scheduled port set')
        for key, value in port_busy.items():
            close(sched['ports'][key]['busy_us'], value, f'{phase} {key} busy time')
        busy_lb = max(port_busy.values(), default=0)
        close(traffic['chunk_service_port_lower_bound_us'], busy_lb, f'{phase} service bound')
        require(sched['makespan_us'] + 1e-7 >= busy_lb >= payload_lb - 1e-7,
                f'{phase} makespan must cover port service')
        checks[phase] = dict(rdma_payload_bytes=int(rdma.sum()),
                             nvlink_payload_bytes=int(nv.sum()),
                             independently_checked_ports=len(port_busy),
                             expert_node_dedup=True, reverse=phase == 'gradient')
    return dict(pass_=True, active_extra_copies=int(extra.sum()),
                missing_active_copies=int(missing.sum()), phases=checks)


def audit_costs(row):
    token, weights = row['token'], row['weights']
    l = token['latency_us']
    wf, gr = weights['forward']['schedule'], weights['gradient']['schedule']
    for cost in row['training']:
        m = cost['microbatches_per_update']
        delta = cost['incremental_planner_us_per_microbatch']
        train = 2*l['dispatch'] + 3*l['compute'] + 2*l['combine']
        close(cost['token_training_segment_us_per_microbatch'], train, 'training token cost')
        expected = wf['makespan_us'] + gr['makespan_us'] + m * (train + delta)
        close(cost['serialized_update_us'], expected, 'training segment additive cost')
        require(cost['common_seed_planner_us'] == 'unmeasured_common_term_omitted', 'common planner boundary')
        require(cost['lower_bound_is_achievable'] is False, 'lower bound cannot assert achievable')
        ports = {}
        for phase, multiplier in ((wf, 1), (gr, 1),
                                  (token['dispatch'], 2*m), (token['combine'], 2*m)):
            for port, info in phase['ports'].items():
                ports[port] = ports.get(port, 0) + multiplier * info['busy_us']
        lower = max(max(ports.values(), default=0), 3*m*l['compute']) + m*delta
        close(cost['relaxed_resource_service_lower_bound_us'], lower, 'training resource bound')
        require(expected + 1e-7 >= lower, 'serialized schedule below resource bound')
    # The saved aggregate is for M=1 with four activation/gradient traversals.
    t, cfg = token['traffic'], token['config']
    require(cfg['input_bytes_per_element'] == cfg['output_bytes_per_element'],
            'current aggregate contract requires equal activation precisions')
    size = cfg['hidden'] * cfg['input_bytes_per_element']
    f = np.asarray(t['rdma_copies_src_node_rail_dst_node'], dtype=np.int64)
    v = np.asarray(t['nvlink_copies_node_src_local_dst_local'], dtype=np.int64)
    rdma = 2*size * (f + f.transpose(2, 1, 0))
    nv = 2*size * (v + v.transpose(0, 2, 1))
    for phase in ('forward', 'gradient'):
        rdma += np.asarray(weights[phase]['traffic']['rdma_bytes_src_node_rail_dst_node'])
        nv += np.asarray(weights[phase]['traffic']['nvlink_bytes_node_src_local_dst_local'])
    aggregate = row['aggregate_payload']
    require(np.array_equal(aggregate['rdma_bytes_src_node_rail_dst_node'], rdma), 'aggregate RDMA edges')
    require(np.array_equal(aggregate['nvlink_bytes_node_src_local_dst_local'], nv), 'aggregate NVLink edges')
    require(aggregate['rdma_payload_bytes'] == int(rdma.sum()), 'aggregate RDMA total')
    require(aggregate['nvlink_payload_bytes'] == int(nv.sum()), 'aggregate NVLink total')
    require(aggregate['max_nic_port_bytes'] == int(max(rdma.sum(2).max(), rdma.sum(0).max())), 'aggregate NIC maximum')
    require(aggregate['max_nvlink_port_bytes'] == int(max(nv.sum(2).max(), nv.sum(1).max())), 'aggregate NVLink maximum')
    rail = rdma.sum((0, 2))
    require(np.array_equal(aggregate['rail_rdma_bytes'], rail), 'aggregate rail totals')
    close(aggregate['rail_statistics']['max_over_mean'], rail.max()/rail.mean(), 'aggregate rail ratio')
    return dict(training_compositions_checked=len(row['training']),
                aggregate_four_activation_traversals_plus_weight_gradient=True)


def audit_directory(directory):
    summary = json.loads((directory/'summary.json').read_text())
    rows, failures = [], []
    scenario_dir = ROOT/'outputs/multinode_rail/scenarios'
    for compact in summary['rows']:
        name, method = compact['scenario'], compact['method']
        try:
            row = json.loads((directory/compact['output_json']).read_text())
            path = scenario_dir/f'{name}.npz'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as src, np.load(directory/row['output_npz']) as result:
                x, seed, home = src['topk'], src['owner'], src['home_rank']
                owner, support = result['owner'], result['support']
                require(row['input_npz_sha256'] == digest, 'JSON input binding')
                require(str(result['input_npz_sha256']) == digest, 'NPZ input binding')
                neutral = method.startswith('neutral_r')
                legal = audit_arrays(x, seed, home, owner, support, same_expert=neutral,
                                     edge_nonincrease=neutral,
                                     saved_counts=result['final_pair_counts_rank_expert'],
                                     saved_padded_counts=result['padded_pair_counts_rank_expert'])
                physical = audit_physical_layout(x, owner, home, support, result)
                f, v = exact_traffic(owner, 4, 8)
                t = row['token']['traffic']
                require(np.array_equal(t['rdma_copies_src_node_rail_dst_node'], f), 'token RDMA traffic')
                require(np.array_equal(t['nvlink_copies_node_src_local_dst_local'], v), 'token NVLink traffic')
                weights = audit_weight_result(x, owner, home, row['weights'], row['token']['config'])
                composition = audit_costs(row)
                require(row['changed_pairs'] == int(np.count_nonzero(owner != seed)), 'changed pair count')
                require(compact['changed_pairs'] == row['changed_pairs'], 'summary changed pairs')
            rows.append(dict(scenario=name, method=method, pass_=True,
                             exact_rank_quotas=legal['pair_loads_per_rank'],
                             physical=physical, weights=weights, costs=composition))
        except Exception as exc:
            failures.append(dict(scenario=name, method=method,
                                 error=f'{type(exc).__name__}: {exc}'))
    # Compare declared savings to the same scenario's seed, not a free baseline.
    by_scenario = {(r['scenario'], r['method']): r for r in summary['rows']}
    for row in summary['rows']:
        base = by_scenario.get((row['scenario'], 'seed'))
        if base is None:
            continue
        try:
            require(len(base['training']) == len(row['training']), 'microbatch sensitivity length')
            for candidate, baseline in zip(row['training'], base['training']):
                require(candidate['microbatches_per_update'] == baseline['microbatches_per_update'], 'microbatch baseline alignment')
                expected = baseline['serialized_update_us'] - candidate['serialized_update_us']
                close(candidate['savings_vs_seed_us'], expected, 'summary savings')
                close(candidate['savings_vs_seed_percent_common_planner_excluded'],
                      100*expected/baseline['serialized_update_us'], 'summary percent savings')
        except Exception as exc:
            failures.append(dict(scenario=row['scenario'], method=row['method'],
                                 error=f'{type(exc).__name__}: {exc}'))
    return dict(all_pass=bool(rows) and not failures, artifact_count=len(rows),
                failures=failures, rows=rows,
                scope='Independent exact owner/layout/replica payload/port arithmetic and cost composition audit; not GPU timing or numerical training validation',
                audited_summary_sha256=hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'outputs/multinode_rail/weighted_training')
    args = parser.parse_args()
    result = audit_directory(args.input)
    (args.input/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('all_pass', 'artifact_count', 'failures')}))
    if not result['all_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
