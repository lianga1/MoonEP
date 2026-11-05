"""Original schedule identity and monotonic projection checks."""
import unittest
import numpy as np

from simulate import Config, evaluate
from schedule_reuse import evaluate_reused_schedule


class ScheduleReuseTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(hidden=16, chunk_tokens=3, expert_padding=4)
        self.meta = {"scenario": "small_reference", "provenance": "deterministic test fixture",
                     "topology": {"nodes": 2, "gpus_per_node": 2}, "legal_plan": True}
        self.home = np.arange(8) // 2
        self.support = np.ones((4, 8), bool)
        self.copies = np.array([np.flatnonzero(self.home != r) for r in range(4)])

    def ordinary(self, x, owner):
        return evaluate(dict(topk=x, owner=owner, home_rank=self.home,
                             experts_to_copy=self.copies), self.meta, self.config)

    def reused(self, x, seed, owner):
        return evaluate_reused_schedule(x, seed, owner, self.home, self.support,
                                        self.meta, self.config)

    def fixture(self, rng):
        x = np.array([rng.choice(8, 3, replace=False) for _ in range(4*7)]).reshape(4, 7, 3)
        owner = rng.integers(0, 4, x.shape)
        return x, owner

    def test_seed_identity(self):
        for seed in range(10):
            x, owner = self.fixture(np.random.default_rng(seed))
            ordinary, reused = self.ordinary(x, owner), self.reused(x, owner, owner)
            self.assertEqual(ordinary["traffic"], reused["traffic"])
            self.assertEqual(ordinary["dispatch"], reused["dispatch"])
            self.assertEqual(ordinary["combine"], reused["combine"])
            self.assertEqual(ordinary["latency_us"], reused["latency_us"])

    def test_projected_counts_and_phase_monotonicity(self):
        for seed in range(30):
            rng = np.random.default_rng(seed+20)
            x, owner = self.fixture(rng)
            projected = owner.copy()
            projected[:, :, 1:] = projected[:, :, :1]
            ordinary = self.ordinary(x, owner)
            result = self.reused(x, owner, projected)
            direct = self.ordinary(x, projected)
            self.assertEqual(result["traffic"], direct["traffic"])
            for phase in ("dispatch", "combine"):
                self.assertLessEqual(result[phase]["makespan_us"], ordinary[phase]["makespan_us"]+1e-9)

    def test_free_source_execution_addition(self):
        x = np.tile([0, 1], (4, 2, 1))
        seed = np.zeros_like(x)
        # Source 2's old execution was remote on rank 0; rank 2 is now local.
        owner = seed.copy()
        owner[2, :, 0] = 2
        result = self.reused(x, seed, owner)
        self.assertTrue(result["checks"]["per_token_payload_edge_subset"])

    def test_new_payload_edge_rejected(self):
        x = np.tile([0, 1], (4, 2, 1))
        seed = np.zeros_like(x)
        owner = seed.copy()
        owner[0, 0, 0] = 3
        with self.assertRaisesRegex(ValueError, "RDMA edges"):
            self.reused(x, seed, owner)
        owner = seed.copy()
        owner[0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "NVLink edges"):
            self.reused(x, seed, owner)


if __name__ == "__main__":
    unittest.main()
