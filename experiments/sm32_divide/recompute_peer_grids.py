#!/usr/bin/env python3
"""Independently recompute the complete frozen peer_grids02 evidence cohort.

Standard library only: no benchmark imports, PyTorch, NumPy, GPU, or network.
Reconstruct every rank's input and expected output as little-endian int32
bytes, compare their recorded SHA-256 hashes, validate all 20 raw samples,
and independently recompute the complete per-rank and aggregate summary.
Missing or inconsistent evidence raises an error before output is written.

Defaults, relative to this script:
  input:  components_diagnostics_02/evidence/peer_grids02.json
  output: peer_grids_independent_check.json
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys

HERE = Path(__file__).resolve().parent
WORKLOADS = [
    ('exchange_counts', 1152, 'gather', 4),
    ('materialize_counts', 2816, 'gather', 1),
    ('sample_packet', 4609, 'sum', 1),
    ('global_status', 1, 'max', 1),
]
VARIANTS = [
    ('unfused', 0), ('fused_auto', -1), ('cooperative_1', 1),
    ('cooperative_2', 2), ('cooperative_4', 4), ('cooperative_8', 8),
    ('cooperative_16', 16), ('cooperative_32', 32),
]


def check(ok, label):
    if not ok:
        raise AssertionError(label)


def close(a, b, label):
    check(math.isfinite(a) and math.isfinite(b) and
          math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-9), label)


def digest(values):
    return hashlib.sha256(struct.pack('<%di' % len(values), *values)).hexdigest()


def compare_summary(actual, recorded, label):
    if isinstance(actual, dict):
        check(isinstance(recorded, dict) and set(actual) == set(recorded), label + ' fields')
        for key in actual:
            compare_summary(actual[key], recorded[key], label + '.' + key)
    elif isinstance(actual, list):
        check(isinstance(recorded, list) and len(actual) == len(recorded), label + ' length')
        for i, (a, b) in enumerate(zip(actual, recorded)):
            compare_summary(a, b, label + '[' + str(i) + ']')
    elif isinstance(actual, float):
        close(actual, recorded, label)
    else:
        check(actual == recorded, label)


def reconstruct(phase, words, op, calls):
    # Reconstructed from the recorded input protocol, rather than importing
    # the benchmark's generator. Call-major order matches contiguous [C,N].
    inputs = [[phase * 1000 + rank * 10000 + call * 100 + word * 3 - 1000000
               for call in range(calls) for word in range(words)] for rank in range(8)]
    if op == 'gather':
        output = [value for call in range(calls) for rank in range(8)
                  for value in inputs[rank][call * words:(call + 1) * words]]
    elif op == 'sum':
        output = [sum(inputs[rank][i] for rank in range(8)) for i in range(calls * words)]
    else:
        check(op == 'max', 'unknown operation')
        output = [max(inputs[rank][i] for rank in range(8)) for i in range(calls * words)]
    check(min(output) >= -(2**31) and max(output) < 2**31, 'signed int32 output range')
    input_hashes = [digest(value) for value in inputs]
    check(len(set(input_hashes)) == 8, 'inputs must differ across ranks')
    return input_hashes, digest(output)


def recompute(path):
    raw = path.read_bytes()
    record = json.loads(raw)
    check(record['status'] == 'complete' and record['failures'] == [], 'status/failures')
    meta = record['metadata']
    check(meta['samples'] == 20 and meta['ranks'] == 8 and meta['sm_limit'] == 32, 'matrix metadata')
    check(meta['variants'] == [dict(name=n, grid=g) for n, g in VARIANTS], 'variant matrix')
    check(meta['workloads'] == [dict(name=n, words=w, op=o, calls=c) for n, w, o, c in WORKLOADS],
          'workload matrix')
    check('two untimed COMPLETE alignment collectives' in meta['timing'] and
          'two alignment collectives' in meta['excluded_from_call_timing'],
          'recorded alignment protocol')
    identities = meta['identities']
    check([x['rank'] for x in identities] == list(range(8)), 'identity rank order')
    check(len({x['uuid'] for x in identities}) == 8 and len({x['host'] for x in identities}) == 1,
          'eight distinct physical GPUs on one host')
    check(all(x['sms'] >= 32 for x in identities), 'available SMs')
    check(len(record['samples']) == 640, 'raw row count')
    rows = iter(record['samples'])
    phase = rank_records = call_measurements = 0
    buckets = collections.defaultdict(list)
    for sample in range(20):
        workloads = WORKLOADS if sample % 2 == 0 else list(reversed(WORKLOADS))
        variants = VARIANTS if sample % 2 == 0 else list(reversed(VARIANTS))
        for name, words, op, calls in workloads:
            for position, (variant, grid) in enumerate(variants):
                phase += 1
                row = next(rows)
                actual = (row['sample'], row['phase'], row['workload'], row['variant'],
                          row['grid'], row['position_in_variant_order'])
                check(actual == (sample, phase, name, variant, grid, position),
                      'alternating schedule phase ' + str(phase))
                input_hashes, oracle_hash = reconstruct(phase, words, op, calls)
                check([x['rank'] for x in row['per_rank']] == list(range(8)), 'row rank coverage')
                for rank, item in enumerate(row['per_rank']):
                    label = ' phase/rank ' + str((phase, rank))
                    check(item['phase'] == phase and item['passed'] is True, 'per-rank phase/pass' + label)
                    check(item['input_sha256'] == input_hashes[rank], 'input hash' + label)
                    check(item['output_sha256'] == oracle_hash and item['expected_sha256'] == oracle_hash,
                          'independent output oracle' + label)
                    check(len(item['call_us']) == calls and
                          all(math.isfinite(value) and value > 0 for value in item['call_us']),
                          'raw event durations' + label)
                    close(sum(item['call_us']), item['total_collective_us'], 'raw total' + label)
                    check(math.isfinite(item['group_with_input_reset_us']) and
                          item['group_with_input_reset_us'] + 1e-3 >= item['total_collective_us'],
                          'group interval' + label)
                    rank_records += 1
                    call_measurements += calls
                buckets[name, variant].append(row)

    summary = {}
    for name, words, op, calls in WORKLOADS:
        table = {}
        for variant, grid in VARIANTS:
            cohort = buckets[name, variant]
            check(len(cohort) == 20 and sorted(x['sample'] for x in cohort) == list(range(20)),
                  'cohort sample coverage')
            medians = []
            for rank in range(8):
                items = [row['per_rank'][rank] for row in cohort]
                medians.append(dict(
                    rank=rank,
                    median_total_collective_us=statistics.median(x['total_collective_us'] for x in items),
                    median_group_with_input_reset_us=statistics.median(x['group_with_input_reset_us'] for x in items),
                    median_by_call_us=[statistics.median(x['call_us'][i] for x in items) for i in range(calls)]))
            maximum = max(x['median_total_collective_us'] for x in medians)
            table[variant] = dict(
                grid=grid, per_rank_medians=medians,
                max_rank_median_total_collective_us=maximum,
                max_rank_median_total_divided_by_calls_us=maximum / calls,
                max_rank_median_group_with_input_reset_us=max(x['median_group_with_input_reset_us'] for x in medians))
        baseline = table['unfused']['max_rank_median_total_collective_us']
        for entry in table.values():
            entry['latency_change_vs_unfused_percent'] = 100 * (entry['max_rank_median_total_collective_us'] / baseline - 1)
        summary[name] = dict(words=words, op=op, calls=calls, variants=table)
    compare_summary(summary, record['summary'], 'summary')

    policy = {}
    for name, words, op, calls in WORKLOADS:
        table = summary[name]['variants']
        best = min(table, key=lambda key: table[key]['max_rank_median_total_collective_us'])
        selected = 'fused_auto' if words <= 256 else 'unfused'
        check(best == selected, 'observed selective choice')
        policy[name] = dict(
            words=words, op=op, calls_per_replay=calls, selected=selected,
            legacy_all_fused_per_call_us=table['fused_auto']['max_rank_median_total_divided_by_calls_us'],
            selected_per_call_us=table[selected]['max_rank_median_total_divided_by_calls_us'],
            unfused_per_call_us=table['unfused']['max_rank_median_total_divided_by_calls_us'])
    # Keep provenance relocatable, including explicit inputs outside HERE.
    input_name = os.path.relpath(path.resolve(), HERE)
    return dict(
        status='passed',
        checker='Independent standard-library arithmetic and little-endian int32 SHA-256 reconstruction; does not import benchmark code, torch, or numpy.',
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_file=input_name, source_sha256=hashlib.sha256(raw).hexdigest(),
        checks=dict(
            formal_samples_per_variant=20, workloads=4, variants=8,
            graph_replays_per_rank=640, rank_records=rank_records,
            individual_timed_collectives=call_measurements,
            alternating_schedule_exact=True, unique_phase_on_every_replay=True,
            every_rank_input_hash_reconstructed=True, every_output_and_expected_hash_reconstructed=True,
            all_pass_flags_true=True, signed_int32_sum_range_checked=True,
            all_raw_totals_recomputed=True, all_rank_medians_and_summary_recomputed=True,
            two_untimed_alignment_calls_per_replay_recorded=True, alignment_calls_per_rank=1280),
        hardware=identities, measured_source_files=meta['source_files'],
        recomputed_summary=summary, measured_policy_choices=policy,
        limitations=[
            'Output arrays were not archived in this JSON; their recorded hashes were compared to fully reconstructed deterministic expected int32 byte streams.',
            'This verifies the complete recorded eight-GPU warm-aligned component cohort, not full-planner or training latency.',
            'The <=256 policy is a production boundary reused from the prior single-CTA path; performance optimality at every unmeasured small size is not established.',
            'Two alignment calls and the timing boundary are documented in the measured protocol; this timing JSON is not an independent GPU launch trace or occupancy measurement.',
            'This script recomputes numerical evidence only; it does not repeat the separate source/documentation review.',
            'Frozen source is not modified and no GPU work is run.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / 'components_diagnostics_02/evidence/peer_grids02.json')
    parser.add_argument('--output', type=Path, default=HERE / 'peer_grids_independent_check.json')
    args = parser.parse_args()
    check(args.input.resolve() != args.output.resolve(), 'output must not overwrite original evidence')
    result = recompute(args.input)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: %d rank records, %d timed collective calls; all hashes and summaries agree.\n%s' %
          (result['checks']['rank_records'], result['checks']['individual_timed_collectives'], args.output))


if __name__ == '__main__':
    try:
        main()
    except (AssertionError, KeyError, TypeError, ValueError, OSError, StopIteration, struct.error) as exc:
        print('VALIDATION FAILED: ' + str(exc), file=sys.stderr)
        sys.exit(1)
