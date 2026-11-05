"""CPU-only policy tests; synthetic timings test selection, not performance."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from preplanning_gate import generate_config, select_before_planning, MODES, METRICS


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        raw = dict(status='complete', failures=[], metadata=dict(world=2, gpu='synthetic-test', sources={'kernel.cu':'fixture'}),
                   layers=[dict(layer=0, input_sha256='registered-fixture', shape=[2, 4, 8],
                                post_timing_checked=True, correctness=dict(payload=True, weights=True))])
        self.raw = self.root/'raw.json'; self.raw.write_text(json.dumps(raw))
        comparisons=[]
        for _, (candidate, baselines) in MODES.items():
            for baseline in baselines:
                comparisons.append(dict(candidate=candidate, baseline=baseline, n=10, guardband_ms=0.01,
                    paired_delta_ms=[0.2]*10, rounds=list(range(10)), mean_delta_bootstrap_95pct_ms=[0.2,0.2],
                    measured_positive_gate_pass=True, minimum_sample_count_met=True, candidate_median_ms=1.0))
        self.analysis_data=dict(raw_status='complete', raw_failures=[], source_path=str(self.raw),
            source_sha256=hashlib.sha256(self.raw.read_bytes()).hexdigest(),
            analysis={m:dict(layers=[dict(layer=0,audit_pass=True,comparisons=copy.deepcopy(comparisons))]) for m in METRICS})
        self.analysis = self.root/'analysis.json'
        self.write_analysis()

    def write_analysis(self):
        self.analysis.write_text(json.dumps(self.analysis_data))

    def choose(self, config, **overrides):
        cohort=config['cohorts'][0]
        args=dict(cohort_id=cohort['cohort_id'], observed_contract_sha256=cohort['contract_sha256'],
                  weight_state='cold', weight_version='version-42',
                  immutable_until_consume=True, exclusive_slots_until_consume=True)
        args.update(overrides)
        return select_before_planning(config, **args)

    def test_positive_fixture_enables_only_exact_cold_contract(self):
        config=generate_config(self.analysis)
        self.assertTrue(self.choose(config)['run_optimized_work'])
        self.assertFalse(self.choose(config,observed_contract_sha256='different')['run_optimized_work'])
        self.assertFalse(select_before_planning(config)['run_optimized_work'])

    def test_unknown_lifetime_never_claims_hot_or_cold_savings(self):
        config=generate_config(self.analysis)
        for changes in (dict(weight_state='unknown'),dict(weight_version=None),
                        dict(immutable_until_consume=False),dict(exclusive_slots_until_consume=False)):
            self.assertFalse(self.choose(config,**changes)['run_optimized_work'])

    def test_hot_requires_mapping_and_version(self):
        config=generate_config(self.analysis)
        hot=dict(weight_state='verified_hot',mapping_lifetime_known=True,
                 expected_slot_map_sha256='map-1',current_slot_map_sha256='map-1',cached_weight_version='version-42')
        self.assertEqual(self.choose(config,**hot)['mode'],'optimized_hot')
        for key,value in [('mapping_lifetime_known',False),('current_slot_map_sha256','map-2'),
                          ('cached_weight_version','version-41')]:
            changed=dict(hot);changed[key]=value
            self.assertFalse(self.choose(config,**changed)['run_optimized_work'])

    def test_must_beat_early_native_and_host_time(self):
        # One retained host sample loses against native early; cached flags
        # remain true deliberately, so the gate must recheck the numbers.
        for c in self.analysis_data['analysis']['host_ms']['layers'][0]['comparisons']:
            if c['baseline']=='nativecoldearly': c['paired_delta_ms'][3]=-0.1
        self.write_analysis(); config=generate_config(self.analysis)
        self.assertEqual([c['mode'] for c in config['cohorts'][0]['qualified_modes']],['optimized_hot'])
        self.assertFalse(self.choose(config)['run_optimized_work'])

    def test_failed_or_missing_raw_cannot_qualify(self):
        self.analysis_data['raw_failures']=[dict(error='retained failure')]
        self.write_analysis();config=generate_config(self.analysis)
        self.assertEqual(config['evidence']['analysis_failures'],self.analysis_data['raw_failures'])
        self.assertEqual(config['cohorts'][0]['qualified_modes'],[])
        self.analysis_data['raw_failures']=[]; self.write_analysis()
        self.raw.write_text(self.raw.read_text()+' ')
        config=generate_config(self.analysis)
        self.assertFalse(config['evidence']['raw_hash_verified'])
        self.assertEqual(config['cohorts'][0]['qualified_modes'],[])
        self.raw.unlink();config=generate_config(self.analysis)
        self.assertEqual(config['cohorts'][0]['qualified_modes'],[])

    def test_incomplete_cohort_cannot_qualify(self):
        for metric in METRICS:
            for c in self.analysis_data['analysis'][metric]['layers'][0]['comparisons']:
                c['n']=9;c['paired_delta_ms']=c['paired_delta_ms'][:9];c['rounds']=c['rounds'][:9]
        self.write_analysis()
        self.assertEqual(generate_config(self.analysis)['cohorts'][0]['qualified_modes'],[])


if __name__=='__main__': unittest.main()
