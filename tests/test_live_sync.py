"""Wire-protocol sync: version-vector deltas, and the two-process demo."""

import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("loro", reason="Loro CRDT bindings not installed")

from holo import FHRR, HoloReplica, ReplicatedHoloMap  # noqa: E402


def test_updates_since_carries_only_new_local_ops():
    A = HoloReplica(FHRR(2048, seed=0))
    B = HoloReplica(FHRR(2048, seed=0))
    kv_a, kv_b = ReplicatedHoloMap(A), ReplicatedHoloMap(B)
    last_a, last_b = A.version(), B.version()

    # round 1: both write, exchange blobs only (no cross-doc access)
    kv_a.put("x", "1")
    kv_b.put("y", "2")
    d_a, d_b = A.updates_since(last_a), B.updates_since(last_b)
    B.apply(d_a)
    A.apply(d_b)
    last_a, last_b = A.version(), B.version()
    assert kv_b.get("x")[0] == "1"
    assert kv_a.get("y")[0] == "2"
    assert np.array_equal(A.merged("kv"), B.merged("kv"))

    # round 2: frames assemble a third replica in ANY order
    kv_a.put("z", "3")
    d2 = A.updates_since(last_a)
    B.apply(d2)
    assert kv_b.get("z")[0] == "3"
    C = HoloReplica(FHRR(2048, seed=0))
    for frame in (d_b, d2, d_a):       # scrambled delivery order
        C.apply(frame)
    assert np.array_equal(C.merged("kv"), A.merged("kv"))
    # replaying an old delta after later state changes nothing
    before = B.merged("kv").copy()
    B.apply(d_a)
    assert np.array_equal(before, B.merged("kv"))


def test_two_process_live_sync_converges_with_concurrent_undo():
    # the full demo, shrunk: two OS processes, TCP frames, and at round 2
    # BOTH painters concurrently undo the same stroke (observed-remove
    # over a real wire) — digests must still match
    out = subprocess.run(
        [sys.executable, "examples/live_sync.py", "--rounds", "3", "--dim", "1024",
         "--res", "48", "--undo-round", "2", "--no-montage"],
        capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "CONVERGED" in out.stdout
    assert "same stroke chosen independently: True" in out.stdout
