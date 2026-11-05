"""Single-GPU configurable residency checks; no process group is initialized.

Run ordinary pytest (not torchrun):
  python -m pytest -q -s tests/test_sampled_residency_configurable.py

All eight original source packets are summed on one CUDA device using the
component-only API. This checks selector arithmetic, not distributed runtime.
Four compiled specializations cover K=3/7/16/32, E=520/1024, D=37/128/512 and
G=3/8. The separate public matrix covers D=4096 and small-token sample capping.
"""
import os

import numpy as np
import pytest
import torch

from moonep.sampled_residency import Residency
from tests.sampled_residency_reference import reference


SAMPLE_SEED = 20261026
ROUTER_SEED = 932071
R, S = 8, 64
CASES = (
    pytest.param(520, 3, 37, 3, id="E520_K3_D37_G3"),
    pytest.param(1024, 7, 128, 8, id="E1024_K7_D128_G8"),
    pytest.param(520, 16, 512, 3, id="E520_K16_D512_G3"),
    pytest.param(1024, 32, 128, 8, id="E1024_K32_D128_G8"),
)


def structured_routes(experts, topk, sample_ids, seed):
    """Repeated whole-home, striped and skewed expert groups, with wide IDs."""
    rng = np.random.default_rng(seed)
    home_size = experts // R
    prototypes = []
    for index in range(64):
        home = index % R
        mode = (index // R) % 3
        if mode == 0:
            homes = np.full(topk, home, np.int32)
        elif mode == 1:
            homes = (home + np.arange(topk)) % R
        else:
            homes = np.full(topk, home, np.int32)
            homes[-max(1, topk // 4):] = (home + 1) % R
        group = np.empty(topk, np.int32)
        for h in np.unique(homes):
            positions = np.flatnonzero(homes == h)
            group[positions] = h * home_size + rng.choice(home_size, len(positions), replace=False)
        prototypes.append(group)
    prototypes = np.asarray(prototypes, np.int32)
    prototypes[0] = np.arange(topk, dtype=np.int32)
    prototypes[7] = np.arange(experts-topk, experts, dtype=np.int32)
    choices = rng.integers(0, len(prototypes), size=R*S)
    heavy = rng.random(R*S) < 0.5
    choices[heavy] = rng.integers(0, 16, size=np.count_nonzero(heavy))
    flat = prototypes[choices].copy()
    # Force repeated records in the sample itself, and ensure expert zero and
    # expert IDs beyond the former nine-bit encoding appear in sampled rows.
    flat[sample_ids[:8]] = prototypes[3]
    flat[sample_ids[8:16]] = prototypes[7]
    flat[sample_ids[-1]] = prototypes[0]
    result = flat.reshape(R, S, topk)
    assert np.all(np.diff(np.sort(result, axis=-1), axis=-1) > 0)
    return result


def assert_result(planner, expected, label):
    assert int(planner.status.item()) == 0, (label, "status")
    names = ("support", "extra", "classes", "groups", "frequencies", "tops",
             "first", "records", "sizes", "package_offsets", "sig0", "sig1", "masks")
    for name in names:
        actual = getattr(planner, name).cpu().numpy()
        wanted = expected[name]
        if wanted.dtype == np.uint64:
            actual = actual.view(np.uint64)
        np.testing.assert_array_equal(actual, wanted, err_msg=f"{label}: {name}")
    count = int(planner.package_count.item())
    assert count == len(expected["packages"]), (label, "package count")
    np.testing.assert_array_equal(planner.packages[:count].cpu().numpy(), expected["packages"],
                                  err_msg=f"{label}: package order")
    assert planner.accepted_packages() == expected["history"], (label, "complete greedy history")
    copies = planner.experts_to_copy.cpu().numpy()
    home = np.arange(planner.experts) // (planner.experts // R)
    for rank in range(R):
        wanted = np.flatnonzero(expected["support"][rank] & (home != rank))
        np.testing.assert_array_equal(copies[rank][copies[rank] >= 0], wanted)


@pytest.mark.parametrize("experts,topk,samples,groups", CASES)
def test_configurable_selector_exact_and_forced_collision(experts, topk, samples, groups):
    if not torch.cuda.is_available():
        pytest.skip("one CUDA device is required")
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("run this component test with ordinary single-process pytest, not torchrun")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    slots = experts // R
    ids = np.sort(np.random.default_rng(SAMPLE_SEED).choice(R*S, samples, replace=False)).astype(np.int64)
    empty_copies = np.full((R, slots), -1, np.int32)
    ctx = dict(R=R, E=experts, K=topk, S=S, B=slots, rank=0, device=device)
    planner = Residency(ctx, SAMPLE_SEED, sample_size=samples,
                        groups_per_class=groups, component_only=True)
    np.testing.assert_array_equal(planner.ids.cpu().numpy(), ids)
    assert planner.sample_size == samples
    assert planner.packet.numel() == samples * (2 * topk + 2) + 1
    assert planner.history.shape == (R * slots, 6 + 2 * ((experts + 63) // 64))
    routes = torch.empty((R, S, topk), dtype=torch.int32, device=device)
    owner = torch.empty_like(routes)
    copies = torch.as_tensor(empty_copies, device=device)
    route_views, owner_views = list(routes.unbind(0)), list(owner.unbind(0))
    packet_sum = torch.empty_like(planner.packet)

    def execute():
        packet_sum.zero_()
        for source in range(R):
            planner.gather_local(route_views[source], owner_views[source], copies, source_rank=source)
            packet_sum.add_(planner.packet)
        planner.build_from_reduced_samples(copies, packet_sum)

    def inputs(seed):
        x = structured_routes(experts, topk, ids, seed)
        initial_owner = (x // (experts // R)).astype(np.int32)
        expected = reference(x, initial_owner, empty_copies, ids,
                             experts=experts, groups_per_class=groups)
        # This is a nontrivial fixture, including repeated signatures and
        # positive-gain global package selections, rather than an empty plan.
        assert expected["frequencies"].max() > 1
        assert len(expected["packages"]) > 0 and len(expected["history"]) > 0
        assert x.reshape(-1, topk)[ids].max() == experts - 1
        assert x.reshape(-1, topk)[ids].min() == 0
        routes.copy_(torch.as_tensor(x, device=device))
        owner.copy_(torch.as_tensor(initial_owner, device=device))
        return expected

    graph = None
    try:
        expected = inputs(ROUTER_SEED)
        execute()
        torch.cuda.synchronize()
        assert_result(planner, expected, "eager")
        # Reuse allocations while changing the router rows after capture. A
        # precomputed frequency/support table cannot pass this second oracle.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            execute()
        expected = inputs(ROUTER_SEED + 1)
        planner.support.zero_()
        planner.frequencies.fill_(-1)
        planner.history_count.fill_(-1)
        planner.status.fill_(31)
        graph.replay()
        torch.cuda.synchronize()
        assert_result(planner, expected, "graph with new routes")
        # Force every hash pair to collide, then invoke the SAME production
        # tiled frequency kernel through its diagnostic split entry point.
        expected_frequencies = planner.frequencies.clone()
        planner.sig0.zero_()
        planner.sig1.zero_()
        planner.frequencies.fill_(-1)
        planner.ext.count_frequencies(planner.classes, planner.groups, planner.sig0,
                                      planner.sig1, planner.frequencies)
        torch.cuda.synchronize()
        assert torch.equal(planner.frequencies, expected_frequencies), "hash collision changed exact group counts"
        # A normal replay restores real hashes and all downstream tables.
        graph.replay()
        torch.cuda.synchronize()
        assert_result(planner, expected, "replay after forced hash collision")
    finally:
        if graph is not None:
            graph.reset()
