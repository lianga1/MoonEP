"""Offline eligibility plus a conservative decision BEFORE optimized GPU work.

This is a finite-cohort policy, not a guarantee for future inputs or model time.
No router/owner scan, GPU call, or weight transfer occurs in select_before_planning.
The caller must supply an already known input identity and a contract derived
from its actual configured sources/hardware, rather than blindly copying the
saved contract hash. New inputs and unknown weight lifetimes use native MoonEP.

Usage: python preplanning_gate.py --analysis analysis.json --output gate.json
An explicit --raw can locate the exact benchmark JSON on another machine.
Existing files are never overwritten. Raw/analysis evidence is never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


METRICS = ('cuda_ms', 'host_ms')
MODES = {
    'optimized_cold_serial': ('optcoldserial', ('nativecoldserial', 'nativecoldearly')),
    'optimized_cold_early': ('optcoldearly', ('nativecoldserial', 'nativecoldearly')),
    'optimized_hot': ('opthot', ('nativehot',)),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def cohort_contract(metadata, row):
    """Exact replay contract. Computing input hashes is NOT a free online scan.

    Store the input identity when registering an offline replay. A real online
    workload needs separately calibrated pre-planning features and their cost.
    Missing old metadata remains missing; it is not replaced with an assumption.
    """
    keys = ('world', 'gpu', 'torch', 'cuda', 'nccl', 'hidden', 'intermediate',
            'weight_dtype', 'expert_bytes', 'experts', 'extra_slots', 'topk',
            'token_padding', 'replica_target', 'replica_max', 'enable_pdl',
            'rounds', 'source_mapping', 'num_sms', 'prefetch_sms', 'timing_mode',
            'materializer', 'residency', 'residency_variant', 'sources')
    return dict(layer=row['layer'], input_sha256=row['input_sha256'],
                shape=row['shape'], configuration={k: metadata[k] for k in keys if k in metadata})


def comparison_passes(c):
    """Recheck the conservative rule, not only a cached eligibility boolean."""
    try:
        n, guard = c['n'], float(c['guardband_ms'])
        deltas, ci = c['paired_delta_ms'], c['mean_delta_bootstrap_95pct_ms']
        return (type(n) is int and n >= 10 and len(deltas) == n
                and len(c['rounds']) == n and len(set(c['rounds'])) == n
                and c['measured_positive_gate_pass'] is True
                and c['minimum_sample_count_met'] is True
                and math.isfinite(guard) and guard >= 0
                and all(math.isfinite(x) and x > guard for x in deltas)
                and len(ci) == 2 and all(math.isfinite(x) for x in ci)
                and ci[0] > guard and ci[1] >= ci[0])
    except (KeyError, TypeError, ValueError):
        return False


def generate_config(analysis_path, raw_path=None):
    analysis_path = Path(analysis_path)
    analysis_bytes = analysis_path.read_bytes()
    analysis = json.loads(analysis_bytes)
    if raw_path is None:
        named = Path(analysis.get('source_path', '__missing_raw__'))
        candidates = (named, analysis_path.parent / named.name)
        raw_path = next((p for p in candidates if p.is_file()), None)
    raw_bytes = Path(raw_path).read_bytes() if raw_path is not None else None
    raw = json.loads(raw_bytes) if raw_bytes is not None else {}
    raw_sha = hashlib.sha256(raw_bytes).hexdigest() if raw_bytes is not None else None
    verified = raw_sha is not None and raw_sha == analysis.get('source_sha256')
    global_ok = (verified and raw.get('status') == 'complete'
                 and analysis.get('raw_status') == 'complete'
                 and raw.get('failures') == [] and analysis.get('raw_failures') == [])
    rows = {r['layer']: r for r in raw.get('layers', [])}
    metric_rows = {m: {r['layer']: r for r in analysis.get('analysis', {}).get(m, {}).get('layers', [])}
                   for m in METRICS}
    layers = sorted(set(rows).union(*(set(v) for v in metric_rows.values())))
    cohorts = []
    for layer in layers:
        row = rows.get(layer)
        audit = (bool(row) and row.get('post_timing_checked') is True
                 and bool(row.get('correctness')) and all(v is True for v in row['correctness'].values()))
        contract = cohort_contract(raw['metadata'], row) if row else None
        candidates, rejected = [], {}
        for mode, (candidate, baselines) in MODES.items():
            good = global_ok and audit
            lower_bounds, medians, reasons = [], [], []
            if not global_ok: reasons.append('missing/incomplete/failed or hash-mismatched raw evidence')
            if not audit: reasons.append('full-path correctness audit unavailable')
            for metric in METRICS:
                mr = metric_rows[metric].get(layer, {})
                if mr.get('audit_pass') is not True: good = False
                comps = mr.get('comparisons', [])
                for baseline in baselines:
                    matched = [c for c in comps if c.get('baseline') == baseline and c.get('candidate') == candidate]
                    if len(matched) != 1 or not comparison_passes(matched[0]):
                        good = False; reasons.append(metric + ': no positive complete cohort versus ' + baseline)
                    else:
                        lower_bounds.append(matched[0]['mean_delta_bootstrap_95pct_ms'][0])
                        if metric == 'cuda_ms': medians.append(matched[0]['candidate_median_ms'])
            if good:
                candidates.append(dict(mode=mode, minimum_measured_mean_lower_bound_ms=min(lower_bounds),
                                       measured_candidate_cuda_median_ms=max(medians)))
            else:
                rejected[mode] = reasons or ['audit failed']
        candidates.sort(key=lambda x: x['measured_candidate_cuda_median_ms'])
        cohorts.append(dict(layer=layer, cohort_id=digest([raw_sha, layer]),
                            contract=contract, contract_sha256=digest(contract) if contract else None,
                            qualified_modes=candidates, rejected_modes=rejected,
                            preplanning_default='native', allow_analytical_amortization=False))
    return dict(schema_version=1, policy='exact_replay_finite_cohort_before_optimization',
                default='native', future_positive_guaranteed=False,
                skip_optimized_planner_when_native_selected=True,
                decision_overhead_measured=False,
                scope='planner + actual expert copies + dispatch + identity combine; no expert GEMM/training',
                evidence=dict(analysis_path=str(analysis_path), analysis_sha256=hashlib.sha256(analysis_bytes).hexdigest(),
                              raw_path=str(raw_path) if raw_path else None, raw_sha256=raw_sha,
                              raw_hash_verified=verified, raw_status=raw.get('status', 'missing'),
                              raw_failures=raw.get('failures'), analysis_failures=analysis.get('raw_failures')),
                cohorts=cohorts,
                caution='Eligibility applies only to retained measured cohorts. Unknown weight version or mapping lifetime uses native. '
                        'Do not compute the optimized plan to decide whether to run it; its sunk cost cannot be refunded.')


def select_before_planning(config, *, cohort_id=None, observed_contract_sha256=None,
                          weight_state='unknown', weight_version=None,
                          immutable_until_consume=False, exclusive_slots_until_consume=False,
                          mapping_lifetime_known=False, expected_slot_map_sha256=None,
                          current_slot_map_sha256=None, cached_weight_version=None):
    """Return a dispatch choice without doing any GPU optimization work.

    cold: full expert-copy cost is required and already included in its evidence.
    verified_hot: exact slots/version must already be resident and remain valid;
      a reused weight version alone does not establish a reusable slot mapping.
    unknown: always native, with the native caller's normal copy/version protocol.
    All ranks must receive the same choice BEFORE launching collective kernels.
    """
    def native(reason):
        return dict(planner='native', mode='native', reason=reason,
                    run_optimized_work=False, skip_expert_copy=False,
                    future_positive_guaranteed=False)
    if config.get('schema_version') != 1: return native('unsupported gate configuration')
    matches = [c for c in config.get('cohorts', []) if c['cohort_id'] == cohort_id]
    if len(matches) != 1: return native('unregistered input/configuration cohort')
    cohort = matches[0]
    if not observed_contract_sha256 or observed_contract_sha256 != cohort['contract_sha256']:
        return native('runtime code/hardware/input contract not verified')
    if weight_state not in ('cold', 'verified_hot'):
        return native('weight or mapping lifetime is unknown')
    if not weight_version or immutable_until_consume is not True or exclusive_slots_until_consume is not True:
        return native('weight publication/version or destination-slot lifetime is not established')
    if weight_state == 'verified_hot':
        if (mapping_lifetime_known is not True or not expected_slot_map_sha256
                or expected_slot_map_sha256 != current_slot_map_sha256
                or cached_weight_version != weight_version):
            return native('hot weights require the exact current version and physical slot mapping')
        allowed = ('optimized_hot',)
    else:
        allowed = ('optimized_cold_serial', 'optimized_cold_early')
    mode = next((c['mode'] for c in cohort['qualified_modes'] if c['mode'] in allowed), None)
    if mode is None: return native('no measured positive full-path cohort for this weight state')
    return dict(planner='optimized', mode=mode, reason='exact cohort passed both CUDA and host paired criteria',
                run_optimized_work=True, skip_expert_copy=(weight_state == 'verified_hot'),
                future_positive_guaranteed=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', type=Path, required=True)
    ap.add_argument('--raw', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    config = generate_config(args.analysis, args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(config, out, indent=2, allow_nan=False); out.write('\n')
    print(json.dumps(dict(output=str(args.output), cohorts=len(config['cohorts']),
                          qualified_modes=sum(len(c['qualified_modes']) for c in config['cohorts']), default='native')))


if __name__ == '__main__': main()
