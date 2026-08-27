"""Observed-remove semantics: idempotent deletion, add-wins, undo."""

import numpy as np
import pytest

pytest.importorskip("loro", reason="Loro CRDT bindings not installed")

from holo import (
    FHRR,
    HoloReplica,
    ORHoloMap,
    ORStrokeScene,
    ReplicatedHoloMap,
)


def _pair():
    return HoloReplica(FHRR(2048, seed=0)), HoloReplica(FHRR(2048, seed=0))


def test_arithmetic_retraction_has_the_phantom():
    # the anomaly orset.py exists to fix, pinned as a regression baseline
    A, B = _pair()
    kv_a, kv_b = ReplicatedHoloMap(A), ReplicatedHoloMap(B)
    kv_a.put("k", "v")
    A.sync(B)
    kv_a.delete("k", "v")
    kv_b.delete("k", "v")              # concurrent duplicate retraction
    A.sync(B)
    assert kv_a.get("k")[1] < -0.5     # negative phantom (score ~ -1)


def test_concurrent_double_remove_is_idempotent():
    A, B = _pair()
    or_a, or_b = ORHoloMap(A), ORHoloMap(B)
    or_a.put("k", "v")
    A.sync(B)
    or_a.remove("k")
    or_b.remove("k")                   # same concurrent duplicate removal
    A.sync(B)
    assert abs(or_a.get("k")[1]) < 0.3     # clean zero, not -1
    assert abs(or_b.get("k")[1]) < 0.3
    # allclose, not array_equal: merged() RE-ENCODES tombstoned
    # descriptors locally, and numpy's complex multiply is only
    # reproducible to ~1 ulp across calls (alignment-dependent SIMD
    # paths). Byte determinism holds only for state shipped as bytes.
    assert np.allclose(or_a.store.merged(), or_b.store.merged(), atol=1e-5)


def test_add_wins_over_concurrent_remove():
    A, B = _pair()
    or_a, or_b = ORHoloMap(A), ORHoloMap(B)
    or_a.put("cfg", "v1")
    A.sync(B)
    or_a.remove("cfg")                 # removes the OBSERVED id only
    or_b.put("cfg", "v2")              # concurrent re-add: fresh id
    A.sync(B)
    assert or_a.get("cfg")[0] == "v2"
    assert or_b.get("cfg")[0] == "v2"


def test_remove_covers_all_observed_ids():
    A, B = _pair()
    or_a, or_b = ORHoloMap(A), ORHoloMap(B)
    or_a.put("k", "v1")
    or_b.put("k", "v2")                # two writers, same key
    A.sync(B)
    removed = or_a.remove("k")         # observed both -> tombstones both
    A.sync(B)
    assert len(removed) == 2
    assert or_b.get_all("k") == []


def test_epoch_undo_idempotent_across_peers():
    A, B = _pair()
    sa = ORStrokeScene(A, np.eye(2) * 0.04 ** 2)
    sb = ORStrokeScene(B, np.eye(2) * 0.04 ** 2)
    rng = np.random.default_rng(31)
    for _ in range(10):
        sa.add_splat(rng.uniform(0.1, 0.9, 2), [1.0, 0.5, 0.2])
    doomed = sa.end_stroke()
    for _ in range(10):
        sa.add_splat(rng.uniform(0.1, 0.9, 2), [0.2, 0.5, 1.0])
    kept = sa.end_stroke()
    A.sync(B)
    sa.undo_stroke(doomed)
    sb.undo_stroke(doomed)             # concurrent duplicate undo
    A.sync(B)
    P = rng.uniform(0, 1, size=(300, 2)).astype(np.float32)
    va, vb = sa.eval_rgb(P), sb.eval_rgb(P)
    assert np.allclose(va, vb)
    # exclusion, not double subtraction: field stays near non-negative
    assert va.min() > -0.2
    assert kept is not None and va.max() > 0.3   # surviving stroke intact


def test_compact_preserves_merged_state():
    A, B = _pair()
    or_a, or_b = ORHoloMap(A), ORHoloMap(B)
    for i in range(12):
        or_a.put(f"k{i}", f"v{i % 4}")
    A.sync(B)
    or_b.remove("k3")
    or_b.remove("k7")
    A.sync(B)
    before = or_a.store.merged().copy()
    folded = or_a.store.compact()      # owner folds B's tombstones
    A.sync(B)
    assert folded == 2
    assert np.allclose(or_a.store.merged(), before, atol=1e-4)
    assert np.allclose(or_b.store.merged(), before, atol=1e-4)
    assert or_a.get("k4")[0] == "v0"   # everything else still reads
