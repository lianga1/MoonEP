"""CPU IPC coverage and opt-in, single-device segmented-VMM kernel tests."""

import ast
import os
import shutil
import socket
import struct
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(autouse=True)
def cleanup_moonep_buffers():
    # This module does not create planner buffers or initialize distributed CUDA.
    yield


def _fd_exchange_runtime(world):
    path = Path(__file__).resolve().parents[1] / "moonep/buffer.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_exchange_ipc_fd_batches"
    ]
    # Annotations refer to ProcessGroup, but no extension import is needed.
    thread = threading.local()
    barrier = threading.Barrier(world, timeout=10)
    values = [None] * world
    broadcast = [None]

    def gather(out, value, group=None):
        values[thread.rank] = value
        barrier.wait()
        out[:] = values
        barrier.wait()

    def bcast(out, src, group=None):
        if thread.rank == src:
            broadcast[0] = out[0]
        barrier.wait()
        out[0] = broadcast[0]
        barrier.wait()

    dist = SimpleNamespace(
        ProcessGroup=object,
        get_rank=lambda g: thread.rank,
        get_world_size=lambda g: world,
        get_global_rank=lambda g, r: r,
        all_gather_object=gather,
        broadcast_object_list=bcast,
        barrier=lambda **kw: barrier.wait(),
    )
    namespace = dict(
        os=os, shutil=shutil, socket=socket, struct=struct, tempfile=tempfile, dist=dist
    )
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["_exchange_ipc_fd_batches"], thread


@pytest.mark.parametrize("count", [1, 129, 300])
def test_fd_batches_preserve_rank_and_handle_order(count):
    exchange, thread = _fd_exchange_runtime(2)
    sources = []
    try:
        for rank in range(2):
            row = []
            sources.append(row)
            for i in range(count):
                fd = os.memfd_create("moonep-ipc-test")
                row.append(fd)
                os.write(fd, struct.pack("ii", rank, i))

        def worker(rank):
            thread.rank = rank
            rows = exchange(sources[rank], None)
            try:
                assert [len(row) for row in rows] == [count, count]
                assert len(set(fd for row in rows for fd in row)) == 2 * count
                for sender, row in enumerate(rows):
                    for i, fd in enumerate(row):
                        assert os.pread(fd, 8, 0) == struct.pack("ii", sender, i)
            finally:
                for row in rows:
                    for fd in row:
                        os.close(fd)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, range(2)))
        # Inputs must still belong to callers, not be consumed by exchange.
        for row in sources:
            for fd in row:
                os.fstat(fd)
    finally:
        for row in sources:
            for fd in row:
                os.close(fd)


def test_fd_batches_cleanup_partial_duplicates(monkeypatch):
    exchange, thread = _fd_exchange_runtime(1)
    thread.rank = 0
    original = os.memfd_create("moonep-ipc-failure")
    duplicated = []
    real_dup = os.dup

    def track(fd):
        result = real_dup(fd)
        duplicated.append(result)
        return result

    monkeypatch.setattr(os, "dup", track)
    try:
        with pytest.raises(OSError):
            exchange([original, -1], None)
        for fd in duplicated:
            with pytest.raises(OSError):
                os.fstat(fd)
        os.fstat(original)
    finally:
        os.close(original)


@pytest.mark.skipif(
    os.environ.get("MOONEP_RUN_VMM_GPU_TESTS") != "1",
    reason="explicit GPU test opt-in required",
)
@pytest.mark.parametrize("use_fabric", [False, True])
def test_unequal_allocations_reorder_alias_and_validate(use_fabric):
    from moonep import _C

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if use_fabric and not _C.nvl_fabric_supported():
        pytest.skip("Fabric allocation is unavailable")
    gran = int(_C.get_vmm_granularity())
    keepalives, shareables = [], []
    try:
        for size, value in [(gran, 11), (2 * gran, 29)]:
            tensor, shareable, owned = _C.nvl_dist_alloc(
                [size], torch.uint8, use_fabric
            )
            keepalives.append(tensor)
            shareables.append(shareable)
            _C.nvl_release_mem_handle(owned)
            tensor.fill_(value)
        kwargs = dict(
            shape=[3 * gran],
            dtype=torch.uint8,
            shareables=torch.stack(shareables),
            segment_bytes=[gran, 2 * gran],
            destination_offsets=[2 * gran, 0],
            use_fabric=use_fabric,
        )
        mapped = _C.nvl_dist_map_segments(**kwargs)
        assert torch.all(mapped[: 2 * gran] == 29)
        assert torch.all(mapped[2 * gran :] == 11)
        mapped[2 * gran :].fill_(37)
        assert torch.all(keepalives[0] == 37)
        for changes in [
            dict(destination_offsets=[0, 0]),
            dict(shape=[4 * gran]),
            dict(destination_offsets=[gran // 2, gran]),
            dict(segment_bytes=[gran - 1, 2 * gran]),
        ]:
            with pytest.raises(RuntimeError):
                _C.nvl_dist_map_segments(**(kwargs | changes))
        del mapped
        # Releasing the alias must not invalidate the original mapping.
        assert torch.all(keepalives[0] == 37)
    finally:
        if not use_fabric:
            for shareable in shareables:
                os.close(int(shareable.item()))
        keepalives.clear()
