"""Seeded CPU mechanisms; replication tests require the real Loro bindings."""

import json

import numpy as np
import pytest

from bench.change_detection import drift
from bench.merge_capture import (
    _bands,
    _empty,
    _exchange,
    _flat,
    encode_half,
    main,
    merge_rules,
    replicate,
    run_synthetic,
    split_scene,
    state_growth,
    synthetic_scene,
)
from holo.capture import band_codebooks, exact_slice
from holo.crdt import HAVE_LORO
from holo.spectral import SplatScene

needs_loro = pytest.mark.skipif(not HAVE_LORO, reason="Loro bindings not installed")


@pytest.fixture
def tiny_bundle():
    rng = np.random.default_rng(11)
    value = (rng.normal(size=(1, 256))
             + 1j * rng.normal(size=(1, 256))).astype(np.complex64)
    return {"mid": {(0, 0, 0): value}}


def test_split_covers_scene_and_overlap_is_shared():
    scene = synthetic_scene()
    a, b = split_scene(scene)
    assert not np.intersect1d(a, b).size
    assert np.array_equal(np.union1d(a, b), np.arange(scene.n))
    a, b = split_scene(scene, overlap=1)
    assert np.intersect1d(a, b).size > 0
    with pytest.raises(ValueError):
        split_scene(scene, overlap=-0.1)


@needs_loro
def test_disjoint_halves_merge_to_the_single_encode():
    # Whole-cell ownership preserves the exact reduction order in each bundle.
    mu = np.array([[0.1, 0.1, 0.1], [0.3, 0.1, 0.1],
                   [0.6, 0.1, 0.1], [0.8, 0.1, 0.1]], dtype=np.float32)
    scene = SplatScene(mu, np.tile(np.eye(3, dtype=np.float32) * 0.01**2,
                                  (4, 1, 1)), np.ones((4, 1), np.float32))
    books = band_codebooks(np.random.default_rng(0), dim=64)
    whole, _ = encode_half(scene, np.arange(4), books, 64, 0.125)
    ia, ib = split_scene(scene)
    a, _ = encode_half(scene, ia, books, 64, 0.125)
    b, _ = encode_half(scene, ib, books, 64, 0.125)
    pa, pb = replicate(a, 1), replicate(b, 2)
    _exchange(pa, pb)
    for key, value in _flat(whole).items():
        assert np.array_equal(pa.merged(key), value)


@needs_loro
def test_overlap_doubles_mass_under_sum_and_not_under_mean(tiny_bundle):
    a, b = replicate(tiny_bundle, 1), replicate(tiny_bundle, 2)
    _exchange(a, b)
    rules = merge_rules(a, "mid/0,0,0")
    value = tiny_bundle["mid"][(0, 0, 0)]
    assert np.array_equal(rules["sum"], 2 * value)
    assert np.array_equal(rules["mean"], value)


@needs_loro
def test_mean_rule_is_computable_from_local_shards(tiny_bundle):
    a, b = replicate(tiny_bundle, 1), replicate(tiny_bundle, 2)
    _exchange(a, b)
    receiver = _empty(256, 1, 3)
    receiver.apply(a.updates_since(receiver.version()))
    assert not receiver.local
    assert np.array_equal(merge_rules(receiver, "mid/0,0,0")["mean"],
                          tiny_bundle["mid"][(0, 0, 0)])
    assert np.array_equal(merge_rules(receiver, "mid/0,0,0")["owner"],
                          tiny_bundle["mid"][(0, 0, 0)])


@needs_loro
def test_delta_bytes_shrink_under_hg8(tiny_bundle):
    sizes = {}
    for codec in ("raw", "hg8"):
        a, b = replicate(tiny_bundle, 1, codec), replicate(tiny_bundle, 2, codec)
        sizes[codec] = sum(r["frame_bytes"] for r in _exchange(a, b))
    assert sizes["hg8"] < sizes["raw"]


@pytest.fixture(scope="module")
def study():
    if not HAVE_LORO:
        pytest.skip("Loro bindings not installed")
    return run_synthetic(dim=64)


def test_merge_time_is_reported_for_both_axes(study):
    for row in study["transport"]:
        assert row["dirty_containers"] > 0
        assert row["mean_splats_per_dirty_cell"] > 0
        assert row["fixed_dirty"] > 0
        assert row["fixed_occupancy"] > 0
        assert len(row["fixed_samples_s"]) == 5
    assert set(study["time_fits"]) == {"raw", "hg8"}


@needs_loro
def test_state_is_flat_in_contributions(tiny_bundle):
    rows = state_growth(tiny_bundle, splats=3)
    assert [r["k"] for r in rows] == [1, 2, 4, 8, 16]
    assert len({r["containers"] for r in rows}) == 1
    assert len({r["live_bytes"] for r in rows}) == 1
    # Flat payload must never be mistaken for flat retained CRDT history.
    assert rows[-1]["snapshot_bytes"] > rows[0]["snapshot_bytes"]


def test_drift_degrades_overlap_monotonically():
    # Isolated coincident overlap: the exact Gaussian field decays at its
    # original centre under a nested jitter draw. No universal monotonicity
    # is asserted for Monte Carlo reconstructions of arbitrary captures.
    mu = np.full((8, 3), 0.5, dtype=np.float32)
    scene = SplatScene(mu, np.tile(np.eye(3, dtype=np.float32) * 0.015**2,
                                  (8, 1, 1)), np.ones((8, 1), np.float32))
    books = band_codebooks(np.random.default_rng(0), dim=32)
    errors = []
    for sigma in (0, 0.05, 0.1, 0.2):
        moved = drift(scene, np.random.default_rng(7), sigma, 0.2, 0.1)
        _, members = encode_half(moved, np.arange(moved.n), books, 32, 0.125)
        field = exact_slice(mu[:1], moved, members, bands=_bands(0.125))
        # A keeps the original field; mean combines it with B's drifted field.
        errors.append(float(abs((8 + field[0, 0]) / 2 - 8) / 8))
    assert np.all(np.diff(errors) >= -1e-7)


def test_synthetic_path_runs_end_to_end(study, tmp_path):
    assert study["source"] == "synthetic"
    assert study["overlap_criterion"]["best_rule"] in ("sum", "mean", "owner")
    output = tmp_path / "study.json"
    main([str(output), "--synthetic", "--dim", "64"])
    assert json.loads(output.read_text())["splats"] == 32
