"""Loro-replicated holographic state: convergence, idempotency, scenes.

Everything here needs the optional `loro` package; the importorskip
below gates the whole file (per TESTING.md, optional-dependency gates
live at file scope in the dedicated file, never mid-file).
"""

import numpy as np
import pytest

pytest.importorskip("loro", reason="Loro CRDT bindings not installed")

from holo import (
    FHRR,
    HoloReplica,
    ReplicatedAttributeScene,
    ReplicatedHoloMap,
    ReplicatedSplatScene,
)


def test_crdt_offline_divergence_converges():
    A = HoloReplica(FHRR(2048, seed=0))
    B = HoloReplica(FHRR(2048, seed=0))
    kv_a, kv_b = ReplicatedHoloMap(A), ReplicatedHoloMap(B)
    for i in range(50):
        kv_a.put(f"a{i}", f"v{i % 10}")
        kv_b.put(f"b{i}", f"v{i % 10}")
    A.sync(B)
    for i in range(50):
        assert kv_a.get(f"b{i}")[0] == f"v{i % 10}"   # A reads B's writes
        assert kv_b.get(f"a{i}")[0] == f"v{i % 10}"
    assert np.array_equal(A.merged("kv"), B.merged("kv"))


def test_crdt_redelivery_is_idempotent():
    A = HoloReplica(FHRR(2048, seed=0))
    B = HoloReplica(FHRR(2048, seed=0))
    kv_a = ReplicatedHoloMap(A)
    kv_a.put("k", "v")
    upd = A.updates_for(B)
    B.apply(upd)
    once = B.merged("kv").copy()
    B.apply(upd)                       # replay the exact same bytes
    assert np.array_equal(once, B.merged("kv"))


def test_crdt_retraction_and_concurrency():
    A = HoloReplica(FHRR(2048, seed=0))
    B = HoloReplica(FHRR(2048, seed=0))
    kv_a, kv_b = ReplicatedHoloMap(A), ReplicatedHoloMap(B)
    kv_a.put("k", "from-A")
    kv_b.put("k", "from-B")            # concurrent write, same key
    A.sync(B)
    assert {l for l, _ in kv_a.get_all("k")} == {"from-A", "from-B"}
    kv_b.delete("k", "from-B")         # B retracts its own write
    A.sync(B)
    assert [l for l, _ in kv_a.get_all("k")] == ["from-A"]


def test_crdt_attributed_scene_converges():
    A = HoloReplica(FHRR(2048, seed=0))
    B = HoloReplica(FHRR(2048, seed=0))
    sa = ReplicatedAttributeScene(A, 0.03)
    sb = ReplicatedAttributeScene(B, 0.03)
    # well-separated grids (spacing 0.1 > 3 sigma) keep vote margins ~1
    placed = []
    for i in range(4):
        for j in range(8):
            mu = np.array([0.05 + 0.1 * i, 0.1 + 0.1 * j])
            lab = ["tree", "water"][(i + j) % 2]
            sa.add_splat(mu, lab)              # A paints the left half
            placed.append((mu, lab))
            mu = np.array([0.6 + 0.1 * i, 0.1 + 0.1 * j])
            lab = ["rock", "path"][(i + j) % 2]
            sb.add_splat(mu, lab)              # B paints the right half
            placed.append((mu, lab))
    A.sync(B)
    for mu, lab in placed:
        la, score_a = sa.what_is_at(mu)
        lb, score_b = sb.what_is_at(mu)
        assert la == lab and lb == lab         # both peers answer correctly
        assert score_a == pytest.approx(score_b, abs=1e-4)
    # B can query a label it never used locally: the label registry plus
    # hash-derived codewords and W rebuild the codebook coordination-free
    P = np.random.default_rng(5).uniform(0, 1, (200, 2)).astype(np.float32)
    assert np.allclose(sa.eval_where("tree", P), sb.eval_where("tree", P))
    own = np.array([mu for mu, lab in placed if lab == "tree"])
    assert sb.eval_where("tree", own).min() > 0.7


def test_crdt_record_payloads_decode_on_remote_peer():
    A = HoloReplica(FHRR(2048, seed=0))
    B = HoloReplica(FHRR(2048, seed=0))
    sa = ReplicatedAttributeScene(A, 0.03)
    sb = ReplicatedAttributeScene(B, 0.03)
    stored = []
    for i in range(3):
        for j in range(5):
            mu = np.array([0.08 + 0.11 * i, 0.1 + 0.11 * j])
            f = {"kind": f"kind{(i + j) % 3}", "owner": "peer-A"}
            sa.add_splat_record(mu, f)
            stored.append((mu, f))
            mu = np.array([0.6 + 0.11 * i, 0.1 + 0.11 * j])
            f = {"kind": f"kind{(i * j) % 4}", "owner": "peer-B"}
            sb.add_splat_record(mu, f)
            stored.append((mu, f))
    A.sync(B)
    # every field of every record decodes on BOTH peers — including
    # records the decoding peer never stored, via codebooks it never built
    for mu, fields in stored:
        for role, filler in fields.items():
            for scene in (sa, sb):
                label, score = scene.record_at(mu, role)
                assert label == filler
                assert score > 0.5
    # the schema itself is discovered from the registry, not agreed on
    assert set(sb.records.known_roles()) == {"kind", "owner"}
    got = sb.fields_at(stored[0][0])
    assert {r: v[0] for r, v in got.items()} == stored[0][1]


def test_crdt_scene_renders_identically_on_both_peers():
    A = HoloReplica(FHRR(2048, seed=0))
    B = HoloReplica(FHRR(2048, seed=0))
    sigma = np.eye(2) * 0.04 ** 2
    sa, sb = (ReplicatedSplatScene(A, sigma),
              ReplicatedSplatScene(B, sigma))
    rng = np.random.default_rng(17)
    for _ in range(40):
        sa.add_splat(rng.uniform([0.1, 0.1], [0.5, 0.9]))
        sb.add_splat(rng.uniform([0.5, 0.1], [0.9, 0.9]))
    A.sync(B)
    P = rng.uniform(0, 1, size=(300, 2))
    va, vb = sa.eval(P), sb.eval(P)
    assert np.allclose(va, vb)
    # and the merged render actually contains BOTH halves
    left = P[:, 0] < 0.45
    assert va[left].max() > 0.3 and va[~left].max() > 0.3


def test_bundle_blob_format_roundtrip():
    from holo.sync import pack_bundle, unpack_bundle
    rng = np.random.default_rng(61)
    one = (rng.normal(size=256) + 1j * rng.normal(size=256)) \
        .astype(np.complex64)
    three = (rng.normal(size=(3, 128)) + 1j * rng.normal(size=(3, 128))) \
        .astype(np.complex64)
    assert np.array_equal(unpack_bundle(pack_bundle(one)), one)
    got = unpack_bundle(pack_bundle(three))
    assert got.shape == (3, 128) and np.array_equal(got, three)
    with pytest.raises(ValueError, match="format tag"):
        unpack_bundle(one.tobytes())          # legacy raw blob, no header
    with pytest.raises(ValueError, match="format tag"):
        unpack_bundle(b"junk")


def test_docs_carry_the_format_record():
    A = HoloReplica(FHRR(2048, seed=0))
    ReplicatedHoloMap(A).put("k", "v")
    A.flush()
    import json
    rec = json.loads(A.doc.get_map("format").get("holo").value)
    from holo.sync import WIRE_VERSION
    assert rec == {"wire": WIRE_VERSION, "dim": 2048, "seed": 0}


def test_mismatched_universe_is_refused():
    A = HoloReplica(FHRR(2048, seed=0))
    B = HoloReplica(FHRR(2048, seed=1))      # different universe
    kv_a = ReplicatedHoloMap(A)
    kv_a.put("k", "v")
    update = A.updates_since(B.version())    # B's own flush wrote ITS record
    with pytest.raises(ValueError, match="wire-format mismatch"):
        B.apply(update)


# -- trimming history -------------------------------------------------------
#
# Loro can discard history before a chosen version (ExportMode.
# ShallowSnapshot), which is the difference between a capture-scale doc
# that grows without bound and one that does not. The catch is that a
# peer BEHIND the trim point cannot be caught up by a delta — and the
# way it fails is silent. These tests pin both halves.

def _seeded(n=6, dim=1024):
    A = HoloReplica(FHRR(dim, seed=0))
    kv = ReplicatedHoloMap(A)
    for i in range(n):
        kv.put("k%d" % i, "v%d" % i)
    A.flush()
    return A


def test_untrimmed_replicas_never_meet_the_guard():
    # the empty shallow_since_vv case: ordinary replicas are untouched
    A, B = _seeded(), HoloReplica(FHRR(1024, seed=0))
    assert not A.doc.is_shallow()
    A.sync(B)
    assert len(B.containers()) == len(A.containers())


def test_a_peer_behind_the_trim_point_is_refused_by_name():
    A = _seeded()
    stale = HoloReplica(FHRR(1024, seed=0))     # never synced
    A.trim_history()
    with pytest.raises(ValueError, match="trimmed"):
        A.updates_for(stale)
    with pytest.raises(ValueError) as exc:
        A.updates_since(stale.doc.oplog_vv)
    # the recovery has to be IN the message: the caller cannot work it
    # out from "refused", and retrying the delta is the wrong move
    assert "snapshot()" in str(exc.value)


def test_the_silent_loss_the_guard_exists_to_prevent():
    """Regression pin, in the style of the arithmetic-retraction phantom.

    Bypassing the guard reproduces Loro's raw behaviour: the stale peer
    receives a plausible, non-empty frame, imports it WITHOUT raising,
    and ends up holding nothing at all. If this ever stops reproducing,
    the guard has become unnecessary — but until then it is the whole
    justification for refusing, and a silent zero is far worse than a
    loud refusal.
    """
    from loro import ExportMode

    A = _seeded()
    stale = HoloReplica(FHRR(1024, seed=0))
    A.trim_history()

    raw = A.doc.export(ExportMode.Updates(from_=stale.doc.oplog_vv))
    assert raw, "the frame is not even empty — it just carries nothing"
    stale.doc.import_(raw)                      # no exception, no error
    assert stale.containers() == []             # and no data either
    assert A.containers() != []                 # while A still has all of it


def test_a_peer_at_the_trim_point_still_syncs():
    A = _seeded()
    B = HoloReplica(FHRR(1024, seed=0))
    A.sync(B)                                   # B is current...
    cut = A.doc.vv_to_frontiers(A.doc.oplog_vv)
    kv = ReplicatedHoloMap(A)
    kv.put("after", "the cut")
    A.flush()
    A.trim_history(cut)                         # ...so the cut is safe for it
    A.sync(B)
    assert sorted(B.containers()) == sorted(A.containers())
    assert np.allclose(B.merged("kv"), A.merged("kv"), atol=1e-5)


def test_a_newcomer_onboards_from_a_trimmed_doc_and_can_write_back():
    A = _seeded()
    A.trim_history()
    C = HoloReplica(FHRR(1024, seed=0))
    C.apply(A.snapshot())                       # snapshot, NOT a delta
    assert sorted(C.containers()) == sorted(A.containers())
    ReplicatedHoloMap(C).put("fresh", "value")
    C.flush()
    A.apply(C.updates_for(A))
    assert np.allclose(A.merged("kv"), C.merged("kv"), atol=1e-5)


def test_trim_history_keeps_this_peer_writing_under_its_own_name():
    A = _seeded()
    peer, keys = A.peer, sorted(A._bundles().keys())
    before, after = A.trim_history()
    assert A.peer == peer                       # a fresh id strands the blobs
    assert sorted(A._bundles().keys()) == keys
    assert after < before
    ReplicatedHoloMap(A).put("post", "trim")
    A.flush()
    # the new write lands under the SAME peer shard, not a second one
    assert {k.rsplit("::", 1)[1] for k in A._bundles().keys()} == {peer}
# -- HG-8 on the wire -------------------------------------------------------

def test_hg8_blobs_are_a_quarter_of_the_bytes_and_round_trip():
    from holo.crdt import pack_bundle, unpack_bundle
    rng = np.random.default_rng(0)
    for shape in ((4096,), (4, 4096)):
        v = (rng.standard_normal(shape)
             + 1j * rng.standard_normal(shape)).astype(np.complex64)
        raw, hg8 = pack_bundle(v), pack_bundle(v, codec="hg8")
        # 2 bytes/component against complex64's 8, plus a 16-byte polar
        # header per channel on top of the 12-byte blob header
        assert 0.24 < len(hg8) / len(raw) < 0.27
        out = unpack_bundle(hg8)
        assert out.shape == v.shape
        assert np.linalg.norm(out - v) / np.linalg.norm(v) < 0.02


def test_raw_stays_byte_identical_so_the_default_did_not_move():
    from holo.crdt import pack_bundle
    v = np.arange(64, dtype=np.complex64)
    assert pack_bundle(v) == pack_bundle(v, codec="raw")


def test_an_unknown_codec_name_is_refused():
    from holo.crdt import pack_bundle
    with pytest.raises(ValueError, match="unknown wire codec"):
        pack_bundle(np.ones(4, np.complex64), codec="nope")
    with pytest.raises(ValueError, match="unknown wire codec"):
        HoloReplica(FHRR(64, seed=0), codec="nope")


def test_an_older_build_refuses_an_hg8_blob_loudly():
    """A new dtype code, not a new layout, so WIRE_VERSION does not move.
    The contract that makes that safe is that a build which does not know
    the code REFUSES rather than reading the payload as complex64."""
    from holo import crdt
    blob = bytearray(crdt.pack_bundle(np.ones(32, np.complex64), codec="hg8"))
    blob[3] = 99                                   # a dtype nobody speaks
    with pytest.raises(ValueError, match="unknown dtype code"):
        crdt.unpack_bundle(bytes(blob))


def test_peers_may_choose_different_codecs_and_still_converge():
    """What replicates is the BLOB, so a peer writing hg8 and a peer
    writing raw still read identical bytes for each other's containers.
    The codec is a local bytes/fidelity trade, not a compatibility
    surface, and needs no coordination."""
    A = HoloReplica(FHRR(1024, seed=0), codec="raw")
    B = HoloReplica(FHRR(1024, seed=0), codec="hg8")
    kv_a, kv_b = ReplicatedHoloMap(A), ReplicatedHoloMap(B)
    kv_a.put("from-a", "x")
    kv_b.put("from-b", "y")
    A.sync(B)
    assert sorted(A.containers()) == sorted(B.containers())
    assert np.allclose(A.merged("kv"), B.merged("kv"), atol=1e-5)


def test_compaction_under_a_lossy_codec_does_not_drift():
    """Regression pin, on the shape where it actually bites.

    compact() used to decode the stored blob, subtract the newly
    tombstoned items and re-pack. Under a lossy codec the subtract moves
    values OFF the quantisation grid, so every compaction re-quantises an
    adjusted approximation and the error accumulates. Re-encoding the
    survivors from the descriptor index quantises the true value exactly
    once however often it runs. Measured here, six staged batches:

        old (subtract + repack)  0.0119 -> 0.0271   2.3x, still climbing
        new (re-encode)          0.0078 -> 0.0079   flat

    Two things this had to get right to be worth having. It measures
    against GROUND TRUTH, not merged() before against merged() after:
    merged() compensates for un-folded tombstones at read time, so it is
    invariant across compaction BY DESIGN and comparing it to itself
    passes against the implementation being pinned. And it uses ONE epoch
    holding everything — the ORStrokeScene shape, and the capture-scale
    stroke case issue #4 is about. ORHoloMap seals every 16 adds, so no
    single epoch is compacted often enough to drift and the bug hides.
    """
    import json

    from holo.orset import ORStore

    A = HoloReplica(FHRR(1024, seed=0), codec="hg8")
    vecs = {}

    def encode(desc):
        if desc not in vecs:
            rng = np.random.default_rng(len(vecs))
            vecs[desc] = (rng.standard_normal(1024)
                          + 1j * rng.standard_normal(1024)
                          ).astype(np.complex64)
        return vecs[desc]

    store = ORStore(A, "s", encode, epoch_size=10 ** 9, channels=1)
    ids = [store.add("item%d" % i) for i in range(40)]
    store.seal()
    A.flush()

    def exact_live():
        tombs = set(store._tombs().keys())
        total = store._zeros()
        for key in sorted(store._blobs().keys()):
            if not key.startswith("s/") or key in tombs:
                continue
            for i, desc in enumerate(json.loads(store._index().get(key).value)):
                if f"{key}/{i}" not in tombs:
                    total += np.atleast_2d(store.encode(desc))
        return total

    errs = []
    for stage in range(6):
        for i in range(stage * 4, stage * 4 + 4):
            store.remove(ids[i])
        store.compact()
        truth = exact_live()
        errs.append(float(np.linalg.norm(store.merged() - truth)
                          / np.linalg.norm(truth)))
    assert max(errs) < 0.012, errs           # the codec's one-shot error
    assert errs[-1] < errs[0] * 1.2, errs    # and it does not grow
