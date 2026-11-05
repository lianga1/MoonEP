"""Conservation/property tests for the weight-aware CPU references."""
import unittest
import numpy as np
from cost_aware import optimize_weight_neutral, optimize_hierarchical
from rail_planner import edge_counts

class CostAwareTest(unittest.TestCase):
    def test_neutral_random_counts_and_edges(self):
        for seed in range(40):
            rng=np.random.default_rng(seed)
            x=np.stack([rng.choice(32,4,replace=False) for _ in range(8*32)]).reshape(8,32,4)
            own=np.repeat(np.arange(8),x.size//8)
            rng.shuffle(own)
            own=own.reshape(x.shape)
            saved=own.copy()
            old=np.bincount((own*32+x).ravel(),minlength=8*32)
            f,v=edge_counts(own,2,4)
            out,diag=optimize_weight_neutral(x,own,nodes=2,gpus=4,rounds=2)
            ff,vv=edge_counts(out,2,4)
            self.assertTrue(np.array_equal(own,saved))
            self.assertTrue(np.array_equal(old,np.bincount((out*32+x).ravel(),minlength=8*32)))
            self.assertTrue(np.all(ff<=f) and np.all(vv<=v))
            self.assertTrue(diag['expert_rank_counts_preserved'])
    def test_hierarchical_random(self):
        for seed in range(20):
            rng=np.random.default_rng(seed)
            x=np.stack([rng.choice(32,4,replace=False) for _ in range(8*32)]).reshape(8,32,4)
            home=np.arange(32)//4
            for order in ['node_first','rank_first']:
                for mapping in ['fifo','locality']:
                    out,support,diag=optimize_hierarchical(x,home,nodes=2,gpus=4,budget=8,mapping=mapping,allocation_order=order)
                    self.assertTrue(np.all(np.bincount(out.ravel(),minlength=8)==128))
                    self.assertTrue(np.all(support[out,x]))
                    self.assertTrue(np.all((support & (np.arange(8)[:,None]!=home)).sum(1)<=8))
                    self.assertIsNone(diag['fallback_reason'])
                    self.assertEqual(diag['inter_pairs_moved'],int(np.maximum(np.bincount(home,weights=np.bincount(x.ravel(),minlength=32),minlength=8).reshape(2,4).sum(1)-512,0).sum()))
    def test_budget_fallback(self):
        rng=np.random.default_rng(123)
        x=np.stack([rng.choice(32,4,replace=False) for _ in range(8*32)]).reshape(8,32,4)
        home=np.arange(32)//4
        with self.assertRaises(ValueError):
            optimize_hierarchical(x,home,nodes=2,gpus=4,budget=0)
        own=np.repeat(np.arange(8),128).reshape(x.shape)
        out,support,diag=optimize_hierarchical(x,home,nodes=2,gpus=4,budget=32,seed_owner=own)
        self.assertIsNone(diag['fallback_reason'])

if __name__=='__main__':
    unittest.main()
