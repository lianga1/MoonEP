"""Original eight-rank traffic + measured two-GPU goodput -> explicitly modeled times.

No GPU work. Standard library only. Never divides aggregate traffic by eight.
All phases are evaluated per layer before summing sequential layer durations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics


ROOT = Path(__file__).resolve().parents[2]
GIB = 2**30
METHODS = ('native_rt0', 'native_rt2', 'sample256_g2_bundle')
SELECTED = dict(dispatch='dispatch_S65536', combine='combine_S65536',
                weight='weight_8_8', gradient='gradient_8_8')


def transpose(a):
    return [list(x) for x in zip(*a)]


def scale(a, n):
    return [[x*n for x in row] for row in a]


def endpoint(a):
    tx = [sum(row) for row in a]
    rx = [sum(row) for row in transpose(a)]
    return dict(total_bytes=sum(tx), tx_bytes_by_rank=tx, rx_bytes_by_rank=rx,
                max_tx_bytes=max(tx), max_rx_bytes=max(rx),
                bottleneck_bytes=max(tx+rx),
                max_tx_ranks=[i for i, x in enumerate(tx) if x == max(tx)],
                max_rx_ranks=[i for i, x in enumerate(rx) if x == max(rx)])


def stamp(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def strict_min_integer(overhead, saving):
    """Smallest positive integer M with M*saving > overhead; no planner charge."""
    if saving <= 0:
        return 1 if saving > overhead else None
    return max(1, math.floor(overhead/saving)+1)


def read_calibration(raw):
    if raw['status'] != 'complete' or raw.get('failures'):
        raise ValueError('Bandwidth attempt must be complete and free of recorded failures')
    if raw['metadata']['physical_gpus'] != 2:
        raise ValueError('Expected two-physical-GPU calibration')
    rows = {}
    for row in raw['cases']:
        if not row['correctness_checked']:
            raise ValueError(f"Unvalidated bandwidth case: {row['name']}")
        ms = []
        host = []
        for sample in row['raw']:
            if len(sample['per_rank']) != 2:
                raise ValueError('Each bandwidth sample must include two ranks')
            ms.append(max(x['cuda_ms'] for x in sample['per_rank']))
            host.append(max(x['host_ms'] for x in sample['per_rank']))
        if not ms or min(ms) <= 0:
            raise ValueError('Missing or invalid bandwidth timings')
        payload = max(row['tx_bytes_by_rank']+row['rx_bytes_by_rank'])
        median_ms = statistics.median(ms)
        gbps = payload/(median_ms*1e6)
        if not math.isclose(gbps, row['effective_endpoint_GBs'], rel_tol=1e-9):
            raise ValueError('Stored bandwidth disagrees with independent recomputation')
        rows[row['name']] = dict(case=row['name'], payload_endpoint_bytes=payload,
            tx_bytes_by_rank=row['tx_bytes_by_rank'], rx_bytes_by_rank=row['rx_bytes_by_rank'],
            samples=len(ms), median_max_rank_cuda_ms=median_ms,
            median_max_rank_host_ms=statistics.median(host),
            effective_endpoint_GBs=gbps,
            min_sample_endpoint_GBs=payload/(max(ms)*1e6),
            max_sample_endpoint_GBs=payload/(min(ms)*1e6),
            correctness_checked=True, extra=row.get('extra', {}))
    selected = {kind: rows[name] for kind, name in SELECTED.items()}
    ranges = {}
    for kind in SELECTED:
        cases = [x for k, x in rows.items() if k.startswith(kind+'_')]
        ranges[kind] = dict(min_case_effective_endpoint_GBs=min(x['effective_endpoint_GBs'] for x in cases),
                           max_case_effective_endpoint_GBs=max(x['effective_endpoint_GBs'] for x in cases),
                           cases=[x['case'] for x in cases],
                           interpretation='shape/direction sensitivity; not a confidence interval for eight GPUs')
    return dict(selected=selected, all_cases=rows, shape_direction_ranges=ranges)


def check_matrix(a, R, label):
    if len(a) != R or any(len(row) != R for row in a):
        raise ValueError(f'{label}: expected {R} by {R}')
    if any(type(v) is not int or v < 0 for row in a for v in row):
        raise ValueError(f'{label}: noninteger or negative traffic')
    if any(a[i][i] for i in range(R)):
        raise ValueError(f'{label}: local payload must be excluded')


def build_payload(traffic, H, I, weight_refreshes=1):
    R = traffic['protocol']['R']
    if R != 8 or traffic['status'] != 'complete' or traffic.get('failures'):
        raise ValueError('Requires complete, audited original eight-rank traffic')
    wb, gb, tb = 3*H*I*2, 3*H*I*4, H*2
    layers = []
    for layer in traffic['layers']:
        methods = {}
        for name in METHODS:
            raw = layer['methods'][name]
            if not all(raw['audit'].values()):
                raise ValueError(f"Traffic audit failure at layer {layer['layer']} / {name}")
            D, C = raw['token_remote_source_to_destination'], raw['expert_home_to_replica']
            check_matrix(D, R, 'D'); check_matrix(C, R, 'C')
            if transpose(D) != raw['combine_remote_source_to_destination']:
                raise ValueError('combine traffic must transpose dispatch traffic')
            if sum(map(sum, D)) != raw['totals']['remote_token_destinations']:
                raise ValueError('Token total disagrees with matrix')
            if sum(map(sum, C)) != raw['totals']['expert_copies_per_full_prefetch']:
                raise ValueError('Expert copy total disagrees with matrix')
            matrices = dict(dispatch=scale(D, tb), combine=scale(transpose(D), tb),
                            weight=scale(C, wb*weight_refreshes), gradient=scale(transpose(C), gb))
            phases = {k: dict(matrix_bytes=v, **endpoint(v)) for k, v in matrices.items()}
            totals = dict(remote_token_destinations=sum(map(sum, D)),
                          expert_copies=sum(map(sum, C)),
                          expert_weight_transfer_events=sum(map(sum, C))*weight_refreshes,
                          forward_token_bytes=2*phases['dispatch']['total_bytes'],
                          training_token_bytes=4*phases['dispatch']['total_bytes'],
                          weight_bytes=phases['weight']['total_bytes'],
                          gradient_bytes=phases['gradient']['total_bytes'])
            totals['training_M1_bytes'] = sum(totals[k] for k in ('training_token_bytes', 'weight_bytes', 'gradient_bytes'))
            methods[name] = dict(phases=phases, totals=totals)
        layers.append(dict(layer=layer['layer'], methods=methods))
    aggregate = {m: {key: sum(l['methods'][m]['totals'][key] for l in layers)
                      for key in layers[0]['methods'][m]['totals']} for m in METHODS}
    comparisons = {}
    opt = aggregate[METHODS[2]]
    for baseline in METHODS[:2]:
        base = aggregate[baseline]
        saved = base['training_token_bytes']-opt['training_token_bytes']
        ew, eg = opt['weight_bytes']-base['weight_bytes'], opt['gradient_bytes']-base['gradient_bytes']
        comparisons[baseline] = dict(extra_expert_copies=opt['expert_copies']-base['expert_copies'],
            extra_weight_bytes=ew, extra_gradient_bytes=eg, saved_training_token_bytes=saved,
            extra_weight_GiB=ew/GIB, extra_gradient_GiB=eg/GIB, saved_training_token_GiB=saved/GIB,
            net_saved_training_M1_bytes=saved-ew-eg, net_saved_training_M1_GiB=(saved-ew-eg)/GIB,
            ideal_fixed_residency_min_M_for_fewer_bytes=strict_min_integer(ew+eg, saved))
    return dict(layers=layers, aggregate=aggregate, optimized_vs=comparisons)


def shared_service(phases, rates, weight_refreshes=1):
    """Approximate weight + forward-dispatch shared endpoint service demand.

    This is not max(Tweight,Tdispatch), and is not a rigorous hardware bound:
    the effective rates include local SM/HBM work as well as transport.
    """
    # Each refresh has its own dispatch opportunity: forward, then backward.
    # Do not merge a backward refresh into an earlier forward-only window.
    tx = [phases['weight']['tx_bytes_by_rank'][i]/weight_refreshes/(rates['weight']*1e6)+
          phases['dispatch']['tx_bytes_by_rank'][i]/(rates['dispatch']*1e6) for i in range(8)]
    rx = [phases['weight']['rx_bytes_by_rank'][i]/weight_refreshes/(rates['weight']*1e6)+
          phases['dispatch']['rx_bytes_by_rank'][i]/(rates['dispatch']*1e6) for i in range(8)]
    return dict(tx_service_ms_by_rank=tx, rx_service_ms_by_rank=rx,
                estimated_ms=max(tx+rx), dispatch_windows=weight_refreshes,
                planner_prefetch_credit_ms=0.0)


def compare_times(base, opt, extra_planner_ms=0):
    out = {}
    token_saved = base['training_token_ms']-opt['training_token_ms']
    for mode in ('serial', 'ideal_all_weights_hidden', 'shared_endpoint_service'):
        key = 'training_M1_'+mode+'_ms'
        budget = base[key]-opt[key]
        # M>1 is an analytical schedule retaining the chosen fixed update-level
        # refresh count. A shared slot pool does not establish that amortization.
        extra_fixed = (opt[key]-opt['training_token_ms'])-(base[key]-base['training_token_ms'])
        out[mode] = dict(baseline_ms=base[key], optimized_ms=opt[key],
            saved_ms_before_extra_planner=budget,
            saved_pct_before_extra_planner=100*budget/base[key] if base[key] else None,
            allowable_extra_planner_ms=budget,
            extra_planner_charge_ms=extra_planner_ms,
            saved_ms_after_extra_planner=budget-extra_planner_ms,
            positive_in_model_before_extra_planner=budget > 0,
            positive_in_model_after_extra_planner=budget > extra_planner_ms,
            ideal_fixed_residency_min_M_without_planner=strict_min_integer(extra_fixed, token_saved),
            per_microbatch_training_token_saved_ms=token_saved,
            extra_fixed_per_update_ms=extra_fixed)
    return out


def estimate_scenario(payload, calibration, eta, factors, extra_planner_ms, rates_override=None,
                      weight_refreshes=1):
    efficiency = {k: eta*factors[k] for k in SELECTED}
    rates = {k: calibration['selected'][k]['effective_endpoint_GBs']*efficiency[k] for k in SELECTED}
    if rates_override:
        rates.update(rates_override)
    layers = []
    for layer in payload['layers']:
        methods = {}
        for name in METHODS:
            phases = layer['methods'][name]['phases']
            t = {kind+'_ms': phases[kind]['bottleneck_bytes']/(rates[kind]*1e6) for kind in SELECTED}
            t['training_token_ms'] = 2*t['dispatch_ms']+2*t['combine_ms']
            t['forward_token_ms'] = t['dispatch_ms']+t['combine_ms']
            t['training_M1_serial_ms'] = t['weight_ms']+t['training_token_ms']+t['gradient_ms']
            t['training_M1_ideal_all_weights_hidden_ms'] = t['training_token_ms']+t['gradient_ms']
            service = shared_service(phases, rates, weight_refreshes)
            t['weight_and_forward_dispatch_service_ms'] = service['estimated_ms']
            t['training_M1_shared_endpoint_service_ms'] = (weight_refreshes*service['estimated_ms']+
                (2-weight_refreshes)*t['dispatch_ms']+2*t['combine_ms']+t['gradient_ms'])
            methods[name] = dict(times=t, shared_endpoint_service=service)
        comps = {b: compare_times(methods[b]['times'], methods[METHODS[2]]['times']) for b in METHODS[:2]}
        layers.append(dict(layer=layer['layer'], methods=methods, optimized_vs=comps))
    keys = layers[0]['methods'][METHODS[0]]['times']
    total = {m: {k: sum(l['methods'][m]['times'][k] for l in layers) for k in keys} for m in METHODS}
    comparisons = {b: compare_times(total[b], total[METHODS[2]], extra_planner_ms) for b in METHODS[:2]}
    return dict(name=f'efficiency_{eta:g}', common_efficiency=eta, efficiency_by_type=efficiency,
                weight_refreshes_per_layer=weight_refreshes,
                absolute_bandwidth_overrides_GBs=rates_override or {},
                effective_endpoint_GBs=rates, layers=layers, aggregate=total, optimized_vs=comparisons)


def token_bandwidth_thresholds(payload, reference):
    """Hold expert W/G rates fixed; solve a common dispatch/combine GB/s crossover.

    This applies to serial and ideal-hidden brackets; max of mixed service terms
    in the shared-endpoint approximation is piecewise and not claimed by this formula.
    """
    out = {}
    for baseline in METHODS[:2]:
        saved_coefficient = sum(
            2*sum(layer['methods'][baseline]['phases'][k]['bottleneck_bytes'] -
                  layer['methods'][METHODS[2]]['phases'][k]['bottleneck_bytes']
                  for k in ('dispatch', 'combine'))/1e6 for layer in payload['layers'])
        rows = {}
        for mode in ('serial', 'ideal_all_weights_hidden'):
            comp = reference['optimized_vs'][baseline][mode]
            fixed = comp['extra_fixed_per_update_ms']
            critical = saved_coefficient/fixed if saved_coefficient > 0 and fixed > 0 else None
            rows[mode] = dict(common_dispatch_combine_critical_GBs=critical,
                positive_when='common token GB/s < critical GB/s; zero extra planner; expert rates fixed' if critical else 'no positive finite crossover derived',
                saved_token_endpoint_bytes_ms_GBs=saved_coefficient,
                extra_fixed_expert_ms=fixed,
                token_rate_multiplier_critical_preserving_measured_dispatch_combine_ratio=
                    comp['per_microbatch_training_token_saved_ms']/fixed if fixed > 0 else None)
        out[baseline] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--traffic', type=Path, required=True)
    ap.add_argument('--bandwidth', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--efficiencies', default='1,0.75,0.5')
    ap.add_argument('--token-efficiency-factor', type=float, default=1)
    ap.add_argument('--weight-efficiency-factor', type=float, default=1)
    ap.add_argument('--gradient-efficiency-factor', type=float, default=1)
    ap.add_argument('--extra-planner-ms', type=float, default=0,
                    help='Extra planner time over baseline summed over all layers; not measured by this script')
    ap.add_argument('--token-bandwidth-sensitivity', default='200,300,450',
                    help='Hypothetical common dispatch/combine GB/s, holding measured expert rates fixed')
    ap.add_argument('--weight-refreshes-per-layer', type=int, choices=[1, 2], default=1,
                    help='1 assumes copies survive from forward to backward; 2 is a uniform two-refresh sensitivity, not an observed multilayer schedule')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing attempts; select a new --output path')
    traffic = json.loads(args.traffic.read_text())
    bandwidth = json.loads(args.bandwidth.read_text())
    H, I = bandwidth['metadata']['H'], bandwidth['metadata']['I']
    calibration = read_calibration(bandwidth)
    payload = build_payload(traffic, H, I, args.weight_refreshes_per_layer)
    factors = dict(dispatch=args.token_efficiency_factor, combine=args.token_efficiency_factor,
                   weight=args.weight_efficiency_factor, gradient=args.gradient_efficiency_factor)
    etas = list(map(float, args.efficiencies.split(',')))
    token_rates = list(map(float, args.token_bandwidth_sensitivity.split(',')))
    if any(not 0 < x <= 1 for x in etas+list(factors.values())) or args.extra_planner_ms < 0:
        raise ValueError('Efficiency must be in (0,1], additional planner time must be nonnegative')
    if any(not math.isfinite(x) or x <= 0 for x in token_rates):
        raise ValueError('Hypothetical token rates must be finite and positive')
    checks = dict(original_R8=traffic['protocol']['R'] == 8,
                  per_layer_matrices_validated=True, calibration_independently_recomputed=True,
                  layer_times_summed_after_endpoint_max=True)
    # Pin the currently audited cohort without preventing new traffic cohorts.
    if traffic.get('verified_totals', {}).get('expert_copies') == dict(zip(METHODS, (71, 203, 507))) and H == 7168 and I == 2048:
        c = payload['optimized_vs']['native_rt0']
        checks['known_436_copy_byte_audit'] = (c['extra_expert_copies'] == 436 and
            c['extra_weight_GiB'] == 35.765625*args.weight_refreshes_per_layer and c['extra_gradient_GiB'] == 71.53125 and
            c['saved_training_token_GiB'] == 67.424774169921875)
        if not checks['known_436_copy_byte_audit']:
            raise ValueError('Pinned traffic byte audit failed')
    scenarios = [estimate_scenario(payload, calibration, eta, factors, args.extra_planner_ms,
                  weight_refreshes=args.weight_refreshes_per_layer) for eta in etas]
    reference = estimate_scenario(payload, calibration, 1, {k: 1 for k in SELECTED}, args.extra_planner_ms,
                                  weight_refreshes=args.weight_refreshes_per_layer)
    independent_token_scenarios = []
    for rate in token_rates:
        row = estimate_scenario(payload, calibration, 1, {k: 1 for k in SELECTED}, args.extra_planner_ms,
                                rates_override=dict(dispatch=rate, combine=rate),
                                weight_refreshes=args.weight_refreshes_per_layer)
        row['name'] = f'hypothetical_token_{rate:g}_GBs'
        row['interpretation'] = 'Only token operation rate changes; expert rates stay measured. Not an eight-GPU measurement or guarantee.'
        independent_token_scenarios.append(row)
    refresh_scenarios = []
    for refreshes in (1, 2):
        rp = build_payload(traffic, H, I, refreshes)
        row = estimate_scenario(rp, calibration, 1, {k: 1 for k in SELECTED}, args.extra_planner_ms,
                                weight_refreshes=refreshes)
        row['name'] = f'weight_refreshes_{refreshes}_per_layer'
        row['interpretation'] = ('One refresh requires each replica to remain resident until its backward computation.' if refreshes == 1 else
            'Uniform forward + backward refresh sensitivity. Actual shared-pool multilayer execution may retain the last layer, so this is not an assertion of exactly sixteen refresh calls.')
        row['payload_aggregate'] = rp['aggregate']
        row['payload_optimized_vs'] = rp['optimized_vs']
        refresh_scenarios.append(row)
    out = dict(schema_version=1, status='complete_estimate',
        scope='Original eight-rank real routing; two-H200 operation goodput calibration; modeled eight-GPU communication only',
        measured_eight_gpu=False, full_training=False, default_enable_optimized=False,
        inputs=dict(traffic=stamp(args.traffic), bandwidth=stamp(args.bandwidth), source=stamp(__file__),
                    notes=stamp(ROOT/'experiments/eight_rank_estimate/model_notes.txt')),
        protocol=traffic['protocol'],
        model=dict(H=H, I=I, activation_dtype='BF16', activation_gradient_dtype='BF16',
            weight_dtype='BF16', parameter_gradient_dtype='FP32', expert_parameters=3*H*I,
            weight_bytes_per_expert=3*H*I*2, gradient_bytes_per_expert=3*H*I*4,
            weight_refreshes_per_layer=args.weight_refreshes_per_layer,
            weight_lifetime_assumption='One refresh requires single-layer F+B or replicas retained from each forward until its backward; a multilayer shared extra-slot pool does not ensure this. Two refreshes is a uniform sensitivity, not a measured schedule.',
            microbatches_per_update=1, four_token_phases=True,
            endpoint_formula='max(max row bytes, max column bytes) / (effective one-direction GB/s * 1e6)',
            sequential_layers='sum each layer time after its own endpoint max',
            topology_assumption='one nonblocking eight-GPU NVSwitch host; no inter-node traffic',
            extra_planner_ms=args.extra_planner_ms, planner_complete_r8_measured=False,
            extra_planner_allocation='aggregate only; per-layer thresholds exclude aggregate-only charge',
            shared_endpoint_service='For each refresh, max endpoint(one-refresh weight bytes/Bweight + its forward or backward dispatch bytes/Bdispatch); then remaining token phases and full gradient. Service approximation, not rigorous physical bound.',
            ideal_all_weights_hidden='optimistic bound; all baseline and candidate weights hidden equally; gradients fully counted',
            scalar_bytes_in_payload_formula=False, actual_timing_may_include_scalar_work=True,
            calibration_shape_mismatch='Two-GPU token calibration has four local plus four remote experts and one remote destination per token. Original eight-rank trace has fewer tasks per destination and several remote destinations. Same K does not imply equal transport/HBM efficiency.',
            M_gt_1='analytical same-trace repetition only; fixed map, version, persistent layer replicas and gradient accumulation required; shared pool cannot provide this automatically',
            omitted=['GEMM', 'optimizer', 'activation retention', 'recomputation', 'DP/TP communication',
                     'R8 metadata and collectives', 'complete distributed R8 planner', 'routing scalar byte formula']),
        audits=checks, calibration=calibration, traffic_payload=payload,
        scenarios=scenarios,
        independent_token_bandwidth_scenarios=independent_token_scenarios,
        weight_refresh_sensitivity=refresh_scenarios,
        token_bandwidth_crossover=token_bandwidth_thresholds(payload, reference),
        interpretation=['Positive saved_ms means faster only under the stated model.',
            'Efficiency scenarios and two-GPU fluctuations are not confidence bounds for eight-GPU execution.',
            'A negative planner budget means communication already loses even with zero additional planner cost.',
            'FP32 multi-replica accumulation may reduce eight-rank gradient bandwidth beyond other traffic classes.',
            'default_enable_optimized=false is an experimental recommendation, not a production switch; no production setting is changed.',
            'New R8 GPU components must be judged from their separate validation/timing outputs; complete distributed R8 planner remains unmeasured.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(output=str(args.output), audits=checks,
        payload_vs_rt0=payload['optimized_vs']['native_rt0'],
        full_rate=out['scenarios'][0]['optimized_vs']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
