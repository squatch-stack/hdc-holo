"""Loro-replicated holographic state: convergence, idempotency, scenes.

Everything here needs the optional `loro` package; the importorskip
below gates the whole file (per TESTING.md, optional-dependency gates
live at file scope in the dedicated file, never mid-file).
"""

import numpy as np
import pytest

pytest.importorskip("loro", reason="Loro CRDT bindings not installed")

from holo import (FHRR, HoloReplica, ReplicatedAttributeScene,  # noqa: E402
                 ReplicatedHoloMap, ReplicatedRecordSpace,
                 ReplicatedSplatScene)


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
