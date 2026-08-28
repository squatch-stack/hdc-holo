"""Observed-remove semantics: idempotent deletion, add-wins, undo."""

import json

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


# -- compaction cost under a lossy wire codec -------------------------------

def _lossy_store(n=120, dim=1024):
    """One epoch holding everything — the ORStrokeScene shape."""
    from holo.orset import ORStore
    A = HoloReplica(FHRR(dim, seed=0), codec="hg8")
    vecs = {}

    def encode(desc):
        if desc not in vecs:
            rng = np.random.default_rng(len(vecs))
            vecs[desc] = (rng.standard_normal(dim)
                          + 1j * rng.standard_normal(dim)).astype(np.complex64)
        return vecs[desc]

    store = ORStore(A, "s", encode, epoch_size=10 ** 9, channels=1)
    ids = [store.add("i%d" % i) for i in range(n)]
    store.seal()
    A.flush()
    return store, ids


def test_the_exact_cache_and_the_rebuild_agree_to_float_rounding():
    """The cache exists only to make compaction cheap — 158x on a
    5,000-item epoch — so the fallback must not be a DIFFERENT answer.

    Agreement is to float32 rounding, not byte equality: subtracting the
    doomed from the exact blob and summing the survivors from the index
    add the same terms in different orders, and one code in ~2,000 lands
    on the other side of a quantisation boundary (measured: 1 byte of
    2,076 differing by one level, 1.1e-7 relative). That is the same ~1
    ulp non-determinism this module already documents for merged(), and
    it does not divide peers: only the OWNER compacts its own epochs, so
    there is one writer and its bytes are what everyone reads.
    """
    from holo.crdt import unpack_bundle
    decoded = []
    for clear_cache in (False, True):
        store, ids = _lossy_store()
        for i in range(6):
            store.remove(ids[i])
        if clear_cache:
            store._exact.clear()            # force the index rebuild
        store.compact()
        key = next(k for k in store._blobs().keys() if k.startswith("s/"))
        decoded.append(np.atleast_2d(unpack_bundle(store._blobs().get(key).value)))
    rel = (np.linalg.norm(decoded[0] - decoded[1])
           / np.linalg.norm(decoded[1]))
    assert rel < 1e-5, rel


def test_compaction_stays_cheap_without_losing_exactness():
    """Subtracting from the cached EXACT blob quantises a true value once,
    exactly as the rebuild does — so the drift is the codec's one-shot
    error either way, not something the cheap path gives up."""
    store, ids = _lossy_store()
    errs = []
    for stage in range(5):
        for i in range(stage * 6, stage * 6 + 6):
            store.remove(ids[i])
        store.compact()
        tombs = set(store._tombs().keys())
        truth = store._zeros()
        for key in sorted(store._blobs().keys()):
            if not key.startswith("s/") or key in tombs:
                continue
            for i, desc in enumerate(json.loads(store._index().get(key).value)):
                if f"{key}/{i}" not in tombs:
                    truth += np.atleast_2d(store.encode(desc))
        errs.append(float(np.linalg.norm(store.merged() - truth)
                          / np.linalg.norm(truth)))
    assert max(errs) < 0.012, errs
    assert errs[-1] < errs[0] * 1.2, errs


def test_the_exact_cache_is_bounded():
    """A long editing session would otherwise hold one (channels, d) array
    per stroke for the life of the process."""
    from holo import orset
    from holo.orset import ORStore
    A = HoloReplica(FHRR(256, seed=0), codec="hg8")
    store = ORStore(A, "s", lambda _d: np.ones(256, np.complex64),
                    epoch_size=1, channels=1)
    for i in range(orset.EXACT_CACHE_EPOCHS + 20):
        store.add("i%d" % i)                 # epoch_size=1 seals every add
        A.flush()
    assert len(store._exact) <= orset.EXACT_CACHE_EPOCHS


def test_a_store_with_no_cache_at_all_still_compacts_correctly():
    """The cache is per-instance and empty after a restart, so the rebuild
    path is the one a fresh process takes."""
    store, ids = _lossy_store()
    for i in range(6):
        store.remove(ids[i])
    store._exact.clear()
    assert store.compact() == 6


# -- cell-keyed epochs (capture scale) --------------------------------------
#
# A capture has thousands of cells and a brush stroke crosses dozens. Giving
# each cell its own ORStore would cost one (channels, d) accumulator per cell
# — ~690 MB on saguaro before any editing — and would make undoing one stroke
# N tombstones instead of one. Worse, N tombstones can arrive across N syncs,
# so a peer that saw half of them renders half an undo.
#
# The cell rides in the blob KEY of one flat map rather than in a child
# container per cell, because Loro's own guidance is that two peers lazily
# creating the same child container concurrently get conflicting container
# ids, which "prevents automatic merging and may result in data loss".

def _cell_store(dim=256, name="s"):
    from holo.orset import ORStore
    A = HoloReplica(FHRR(dim, seed=0))
    vecs = {}

    def encode(desc):
        if desc not in vecs:
            rng = np.random.default_rng(len(vecs))
            vecs[desc] = (rng.standard_normal(dim)
                          + 1j * rng.standard_normal(dim)).astype(np.complex64)
        return vecs[desc]

    return ORStore(A, name, encode, epoch_size=10 ** 9, channels=1), A


def test_one_tombstone_undoes_a_stroke_across_every_cell_it_touched():
    store, A = _cell_store()
    for i in range(12):
        store.add("item%d" % i, cell=(i % 4, 0))      # one stroke, four cells
    stroke = store.seal()
    A.flush()
    assert len(store._blobs().keys()) == 4            # four blobs, one stroke

    before = {c: store.merged(cell=(c, 0)).copy() for c in range(4)}
    assert all(np.linalg.norm(v) > 0 for v in before.values())

    store.remove_epoch(stroke)                        # ONE tombstone
    assert len(store._tombs().keys()) == 1
    for c in range(4):
        assert np.allclose(store.merged(cell=(c, 0)), 0, atol=1e-6), c
    assert np.allclose(store.merged(), 0, atol=1e-6)


def test_merged_reads_one_cell_and_the_whole_field():
    store, A = _cell_store()
    store.add("a", cell="left")
    store.add("b", cell="right")
    store.seal()
    A.flush()
    left, right = store.merged(cell="left"), store.merged(cell="right")
    assert not np.allclose(left, right)
    assert np.allclose(store.merged(), left + right, atol=1e-6)


def test_removing_an_item_does_not_corrupt_a_different_cell():
    """An item lives in exactly one cell's blob. Subtracting it while
    summing another cell would remove something never added there."""
    store, A = _cell_store()
    ids = [store.add("item%d" % i, cell="c%d" % (i % 2)) for i in range(6)]
    store.seal()
    A.flush()
    untouched = store.merged(cell="c1").copy()
    store.remove(ids[0])                              # lives in c0
    assert np.allclose(store.merged(cell="c1"), untouched, atol=1e-6)
    # c0 held items 0, 2 and 4; only 0 was removed
    expect = store.encode("item2") + store.encode("item4")
    assert np.allclose(store.merged(cell="c0"), np.atleast_2d(expect),
                       atol=1e-5)


def test_two_peers_editing_overlapping_cells_converge():
    from holo.orset import ORStore
    A, B = HoloReplica(FHRR(256, seed=0)), HoloReplica(FHRR(256, seed=0))
    vecs = {}

    def encode(desc):
        if desc not in vecs:
            rng = np.random.default_rng(len(vecs))
            vecs[desc] = (rng.standard_normal(256)
                          + 1j * rng.standard_normal(256)).astype(np.complex64)
        return vecs[desc]

    sa = ORStore(A, "s", encode, epoch_size=10 ** 9, channels=1)
    sb = ORStore(B, "s", encode, epoch_size=10 ** 9, channels=1)
    for i in range(6):
        sa.add("a%d" % i, cell="shared" if i % 2 else "a-only")
        sb.add("b%d" % i, cell="shared" if i % 2 else "b-only")
    sa.seal()
    sb.seal()
    A.sync(B)
    for cell in ("shared", "a-only", "b-only"):
        assert np.allclose(sa.merged(cell=cell), sb.merged(cell=cell),
                           atol=1e-5), cell
    assert np.allclose(sa.merged(), sb.merged(), atol=1e-5)


def test_the_accumulator_holds_only_the_cells_this_stroke_touched():
    """Bounded by stroke size, not scene size — the whole reason the cell
    is a key rather than a separate store."""
    store, _replica = _cell_store()
    for i in range(200):
        store.add("item%d" % i, cell="c%d" % (i % 5))
    assert len(store._cells) == 5
    store.seal()
    assert store._cells == {}                         # released on seal


def test_an_unkeyed_epoch_still_writes_the_old_bare_list_index():
    """Docs written before cells existed must read back unchanged."""
    import json
    store, A = _cell_store()
    store.add("plain")
    store.seal()
    A.flush()
    key = next(k for k in store._index().keys() if k.startswith("s/"))
    assert isinstance(json.loads(store._index().get(key).value), list)
    assert "@" not in next(k for k in store._blobs().keys())


def test_compaction_folds_within_the_right_cell():
    store, A = _cell_store()
    ids = [store.add("item%d" % i, cell="c%d" % (i % 2)) for i in range(8)]
    store.seal()
    A.flush()
    keep_c1 = store.merged(cell="c1").copy()
    store.remove(ids[0])
    store.remove(ids[2])                              # both in c0
    assert store.compact() == 2
    assert np.allclose(store.merged(cell="c1"), keep_c1, atol=1e-6)
    expect = sum(store.encode("item%d" % i) for i in (4, 6))
    assert np.allclose(store.merged(cell="c0"), np.atleast_2d(expect),
                       atol=1e-5)


# -- ORStrokeScene at capture scale -----------------------------------------

def test_a_partitioned_stroke_scene_writes_one_blob_per_cell_it_touches():
    """The capture-scale shape: a stroke crosses cells, and each gets its
    own blob under one epoch. `cell_of` uses the same floor-divide rule
    as capture.encode_bands, so a stroke lands where a capture would have
    put it."""
    A = HoloReplica(FHRR(512, seed=0))
    scene = ORStrokeScene(A, sigma=0.05, cell_size=0.25)
    for i in range(8):                       # a stroke sweeping in x
        scene.add_splat([i * 0.125, 0.1], [1.0, 0.0, 0.0])
    stroke = scene.end_stroke()
    A.flush()
    assert scene.cell_of([0.3, 0.1]) == (1, 0)
    assert len(scene.cells()) == 4           # 8 splats at 0.125 over 0.25
    # and undoing it is still ONE tombstone
    scene.undo_stroke(stroke)
    assert len(scene.store._tombs().keys()) == 1
    pts = np.array([[0.3, 0.1]], dtype=np.float32)
    assert np.allclose(scene.eval_rgb(pts), 0, atol=1e-5)


def test_reading_one_cell_sees_only_that_cell_s_strokes():
    A = HoloReplica(FHRR(512, seed=0))
    scene = ORStrokeScene(A, sigma=0.05, cell_size=0.5)
    scene.add_splat([0.1, 0.1], [1.0, 0.0, 0.0])        # cell (0, 0)
    scene.add_splat([0.9, 0.1], [0.0, 0.0, 1.0])        # cell (1, 0)
    scene.end_stroke()
    A.flush()
    here = np.array([[0.1, 0.1]], dtype=np.float32)
    near = scene.eval_rgb(here, cell="(0, 0)")[0]
    far = scene.eval_rgb(here, cell="(1, 0)")[0]
    # the red splat is in this cell; the blue one is not
    assert near[0] > far[0]
    assert np.allclose(scene.eval_rgb(here),
                       near + far, atol=1e-4)


def test_an_unpartitioned_stroke_scene_behaves_exactly_as_before():
    """cell_size=None is the 2-D demo path and must not change."""
    A = HoloReplica(FHRR(512, seed=0))
    scene = ORStrokeScene(A, sigma=0.05)
    assert scene.cell_of([0.3, 0.1]) is None
    scene.add_splat([0.3, 0.1], [1.0, 0.0, 0.0])
    scene.end_stroke()
    A.flush()
    assert scene.cells() == []
    assert all("@" not in k for k in scene.store._blobs().keys())
