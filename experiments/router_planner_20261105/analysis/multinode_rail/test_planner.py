"""Reproducible semantic/property tests for the CPU rail planners.

Run: python3 analysis/multinode_rail/test_planner.py

Random fixtures use R8/N2/G4/E32 and sometimes a replica budget of E=32 to
exercise allocation properties. Those fixtures are not the B8/128K benchmark,
and test runtimes say nothing about GPU planner latency.
"""
from __future__ import annotations

import unittest
import numpy as np

from rail_planner import edge_counts, finalize, optimize
from rail_support import select_support
from same_expert import optimize_same_expert


def naive_edges(owner, nodes=2, gpus=4):
    """Independent token-set oracle; no optimizer bitmask helpers."""
    rdma = np.zeros((nodes, gpus, nodes), np.int64)
    nv = np.zeros((nodes, gpus, gpus), np.int64)
    for source in range(owner.shape[0]):
        sn, rail = divmod(source, gpus)
        for token in owner[source]:
            ranks = set(map(int, token))
            for node in {rank // gpus for rank in ranks} - {sn}:
                rdma[sn, rail, node] += 1
            for rank in ranks:
                dn, dl = divmod(rank, gpus)
                if dl != rail:
                    nv[dn, rail, dl] += 1
    return rdma, nv


def allocations(topk, owner, experts):
    ranks = topk.shape[0]
    return np.bincount((owner.astype(np.int64) * experts + topk).ravel(),
                       minlength=ranks * experts).reshape(ranks, experts)


def random_case(seed, home_only=False):
    rng = np.random.default_rng(seed)
    ranks, per_rank, k, experts = 8, 16, 4, 32
    owner = rng.permutation(np.repeat(np.arange(ranks), per_rank * k)).reshape(ranks, per_rank, k).astype(np.int16)
    home = np.arange(experts, dtype=np.int16) // 4
    if home_only:
        # Unique offsets ensure unique logical experts even when a token has
        # several execution pairs on the same rank.
        offsets = np.stack([rng.permutation(4) for _ in range(ranks * per_rank)]).reshape(owner.shape)
        topk = (owner * 4 + offsets).astype(np.int32)
        support = home[None] == np.arange(ranks)[:, None]
    else:
        topk = np.stack([rng.choice(experts, k, replace=False) for _ in range(ranks * per_rank)]).reshape(owner.shape).astype(np.int32)
        support = (home[None] == np.arange(ranks)[:, None]) | (rng.random((ranks, experts)) < 0.25)
        support[owner, topk] = True
    return topk, owner, home, support


def degree_four_ring(ranks=8):
    return sorted({tuple(sorted((a, (a + jump) % ranks)))
                   for a in range(ranks) for jump in (1, 2)})


class RailPlannerTests(unittest.TestCase):
    def check_rows(self, topk, owner, support, home, budget=32, padding=4):
        finalized = finalize(topk, owner, support, home, budget=budget, padding=padding)
        ranks, experts = topk.shape[0], len(home)
        stride = experts // ranks + budget
        physical = finalized["physical_topk"]
        rows = finalized["destination_row"]
        cu = finalized["physical_cu_seqlens"]
        slot_counts = finalized["physical_pair_counts"]
        np.testing.assert_array_equal(physical // stride, owner)
        slot = physical % stride
        np.testing.assert_array_equal(finalized["final_pair_counts_rank_expert"], allocations(topk, owner, experts))
        self.assertTrue(np.all(cu[:, 0] == 0))
        np.testing.assert_array_equal(np.diff(cu, axis=1), ((slot_counts + padding - 1) // padding) * padding)
        lo, hi = cu[owner, slot], cu[owner, slot + 1]
        self.assertTrue(np.all((rows >= lo) & (rows < hi)))
        # Every real row is in the occupied prefix, not merely padded capacity.
        self.assertTrue(np.all(rows < lo + slot_counts[owner, slot]))
        for rank in range(ranks):
            actual = rows[owner == rank]
            self.assertEqual(len(np.unique(actual)), len(actual))
            local = list(np.flatnonzero(home == rank))
            extra = list(np.flatnonzero(support[rank] & (home != rank)))
            contract = {e: j for j, e in enumerate(local)}
            contract.update({e: experts // ranks + j for j, e in enumerate(extra)})
            for e in np.unique(topk[owner == rank]):
                selected = (owner == rank) & (topk == e)
                expected_slot = contract[int(e)]
                self.assertTrue(np.all(slot[selected] == expected_slot))
                np.testing.assert_array_equal(np.sort(rows[selected]),
                                              cu[rank, expected_slot] + np.arange(selected.sum()))
        return finalized

    def test_random_bounded_planner_properties(self):
        # Alternate the full resident graph and an explicit degree-four graph.
        for seed in range(120):
            with self.subTest(seed=seed):
                x, owner, home, support = random_case(1234 + seed)
                old_x, old_owner, old_support = x.copy(), owner.copy(), support.copy()
                graph = degree_four_ring() if seed % 2 else None
                out, diag = optimize(x, owner, support, nodes=2, gpus=4,
                                     rounds=2, edges=graph)
                np.testing.assert_array_equal(x, old_x)
                np.testing.assert_array_equal(owner, old_owner)
                np.testing.assert_array_equal(support, old_support)
                np.testing.assert_array_equal(np.bincount(out.ravel(), minlength=8), np.full(8, 64))
                self.assertTrue(np.all(support[out, x]))
                before = naive_edges(owner)
                after = naive_edges(out)
                for old, new in zip(before, after):
                    self.assertTrue(np.all(new <= old))
                for exact, optimized in zip(after, edge_counts(out, 2, 4)):
                    np.testing.assert_array_equal(exact, optimized)
                for round_stats in diag["rounds"]:
                    self.assertLessEqual(round_stats["candidate_checks"], 2 * x.size)
                    self.assertLessEqual(round_stats["proposed_tokens"], x.shape[0] * x.shape[1])
                    self.assertLessEqual(round_stats["accepted_tokens"], round_stats["proposed_tokens"])
                    self.assertEqual(round_stats["accepted_tokens"], 2 * round_stats["exchanges"])
                    if graph is not None:
                        self.assertLessEqual(round_stats["maximum_rank_degree"], 4)
                        self.assertLessEqual(round_stats["allowed_undirected_edges"], 8 * 4 // 2)
                self.check_rows(x, out, support, home)

    def test_empty_graph_is_identity(self):
        x, owner, _, support = random_case(100)
        out, diag = optimize(x, owner, support, nodes=2, gpus=4, rounds=2, edges=[])
        np.testing.assert_array_equal(out, owner)
        for stats in diag["rounds"]:
            self.assertEqual(stats["accepted_tokens"], 0)
            self.assertEqual(stats["histogram_counters"], 0)
            self.assertEqual(stats["maximum_rank_degree"], 0)
            self.assertLessEqual(stats["candidate_checks"], 2 * x.size)

    def test_physical_slot_order_regression(self):
        # Rank1 extra e0 must follow home e4 despite its lower logical ID.
        x = np.array([[[0, 4]], [[0, 4]]], np.int32)
        owner = np.array([[[1, 1]], [[0, 0]]], np.int16)
        home = np.arange(8) // 4
        support = home[None] == np.arange(2)[:, None]
        support[1, 0] = support[0, 4] = True
        final = self.check_rows(x, owner, support, home, budget=1, padding=4)
        np.testing.assert_array_equal(final["physical_topk"] % 5, [[[4, 0]], [[0, 4]]])
        np.testing.assert_array_equal(final["destination_row"], [[[4, 0]], [[0, 4]]])

    def test_random_same_expert_preserves_allocation(self):
        for seed in range(60):
            with self.subTest(seed=seed):
                x, owner, home, _ = random_case(8017 + seed)
                old_x, old_owner = x.copy(), owner.copy()
                out, diag = optimize_same_expert(x, owner, nodes=2, gpus=4, rounds=2)
                np.testing.assert_array_equal(x, old_x)
                np.testing.assert_array_equal(owner, old_owner)
                np.testing.assert_array_equal(allocations(x, out, len(home)), allocations(x, owner, len(home)))
                before = naive_edges(owner)
                after = naive_edges(out)
                self.assertLessEqual(int(after[0].sum()), int(before[0].sum()))
                self.assertLessEqual(int(after[1].sum()), int(before[1].sum()))
                self.assertLessEqual(diag["after"]["max_nic_copies"], diag["before"]["max_nic_copies"])
                self.assertLessEqual(diag["after"]["max_rail_copies"], diag["before"]["max_rail_copies"])
                self.assertEqual(diag["new_replicas"], 0)

    def test_same_expert_reduces_last_remote_node(self):
        x = np.array([[[0, 1]], [[0, 2]], [[3, 4]], [[5, 6]]], np.int32)
        owner = np.array([[[3, 0]], [[1, 2]], [[2, 2]], [[3, 3]]], np.int16)
        out, diag = optimize_same_expert(x, owner, nodes=2, gpus=2)
        np.testing.assert_array_equal(allocations(x, out, 7), allocations(x, owner, 7))
        self.assertEqual(diag["operations"]["accepted_swap_pairs"], 1)
        self.assertEqual(diag["before"]["rdma_copies"], 2)
        self.assertEqual(diag["after"]["rdma_copies"], 1)
        self.assertEqual(diag["after"]["nvlink_copies"], 2)

    def test_zero_budget_and_zero_degree_support(self):
        x, owner, home, base = random_case(91, home_only=True)
        originals = x.copy(), owner.copy(), home.copy()
        for opts in ({"budget": 0, "degree": 4},
                     {"budget": 4, "degree": 0},
                     {"budget": 4, "degree": 4, "experts_per_edge": 0}):
            with self.subTest(options=opts):
                support, diag = select_support(x, owner, home, nodes=2, gpus=4, **opts)
                np.testing.assert_array_equal(support, base)
                self.assertEqual(diag["accepted_edges"], [])
                self.assertEqual(sum(diag["new_extra_per_rank"]), 0)
                self.assertLessEqual(diag["candidate_target_checks"], 2 * x.size)
        for current, original in zip((x, owner, home), originals):
            np.testing.assert_array_equal(current, original)
        # Zero budget cannot silently erase mandatory seed replicas.
        x, owner, home, _ = random_case(92)
        with self.assertRaises(ValueError):
            select_support(x, owner, home, nodes=2, gpus=4, budget=0)

    def test_selected_sparse_graph_degree_and_residency(self):
        accepted_total = 0
        for seed in range(10):
            x, owner, home, base = random_case(241 + seed, home_only=True)
            support, diag = select_support(x, owner, home, nodes=2, gpus=4,
                                           budget=4, degree=4, experts_per_edge=1)
            edges = [(edge["a"], edge["b"]) for edge in diag["accepted_edges"]]
            accepted_total += len(edges)
            degrees = np.zeros(8, np.int64)
            for a, b in edges:
                degrees[a] += 1
                degrees[b] += 1
            np.testing.assert_array_equal(degrees, diag["selected_rank_degrees"])
            self.assertTrue(np.all(degrees <= 4))
            self.assertTrue(np.all(support | ~base))
            self.assertTrue(np.all((support & ~base).sum(1) <= 4))
            out, planner = optimize(x, owner, support, nodes=2, gpus=4, edges=edges)
            self.assertLessEqual(planner["rounds"][0]["maximum_rank_degree"], 4)
            np.testing.assert_array_equal(np.bincount(out.ravel(), minlength=8), np.full(8, 64))
            for before, after in zip(naive_edges(owner), naive_edges(out)):
                self.assertTrue(np.all(after <= before))
            self.check_rows(x, out, support, home, budget=4)
        self.assertGreater(accepted_total, 0, "fixture should exercise nonempty sparse graph selection")


if __name__ == "__main__":
    unittest.main(verbosity=2)
