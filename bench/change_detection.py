"""Squatch Stack change detection in one physical frame, using alpha spectra.

Prior art abstracts read: arXiv:2605.07203 (primitive drift and observability)
and arXiv:2512.22830 (multi-view aggregation). The isotropic drift, duplicate
splitting, distance baseline and pooled null below are our definitions, not
reproductions of either method. No camera observability model is available.

Changes smaller than sigma_rec can vanish; drift above sigma_rec can read as
change everywhere. Alpha-only fingerprints are blind to colour changes (RGB
is follow-up work). Non-subset crops are refused. Phasor decoding also adds
the sampling kernel's blur and finite-codebook sidelobes; bbox IoU is not
primitive segmentation accuracy. False mass is threshold-surviving magnitude
outside the true bbox divided by all threshold-surviving magnitude.
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from bench.place_recognition import (
    build_scene_fixed,
    correlate,
    crop_box,
    tile_fingerprints,
    tile_lattice,
)
from holo.capture import ALPHA_MIN, load_scene_file, render_mip, slice_grid
from holo.spectral import (
    SplatScene,
    decode_field_phasor,
    sample_frequencies,
    spectral_bundle,
)


def _take(scene, keep):
    return SplatScene(scene.mu[keep], scene.cov[keep], scene.amp[keep])


def drift(scene, rng, sigma_pos, sigma_amp, split_frac):
    """Independent Gaussian center jitter, positive alpha noise, exact splits.

    Units match scene.mu. Lognormal amplitude noise has mean one and relative
    standard deviation sigma_amp. Splits duplicate covariance/position and
    halve both amplitudes, preserving the perturbed field exactly.
    """
    values = np.asarray([sigma_pos, sigma_amp, split_frac])
    if not np.isfinite(values).all() or np.any(values < 0) or split_frac > 1:
        raise ValueError("invalid drift parameters")
    mu = scene.mu + rng.normal(0, sigma_pos, scene.mu.shape)
    log_sigma = np.sqrt(np.log1p(sigma_amp**2))
    gain = np.exp(rng.normal(-log_sigma**2 / 2, log_sigma, (scene.n, 1)))
    amp = scene.amp * gain
    split = rng.choice(scene.n, int(scene.n * split_frac), replace=False)
    amp[split] *= 0.5
    return SplatScene(np.concatenate([mu, mu[split]]).astype(np.float32),
                      np.concatenate([scene.cov, scene.cov[split]]),
                      np.concatenate([amp, amp[split]]).astype(np.float32))


def _nearest(points, query, radius):
    """Radius-capped nearest distances/indices via spatial bins; no SciPy extra.

    Only neighboring bins can contain a point within radius. Dense bins are
    processed in bounded blocks, so capture size never makes an N*M array.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    bins = {}
    for i, cell in enumerate(np.floor(points / radius).astype(np.int64)):
        bins.setdefault(tuple(cell), []).append(i)
    distances = np.full(len(query), radius, dtype=np.float64)
    indices = np.full(len(query), -1, dtype=np.int64)
    queries = {}
    for i, cell in enumerate(np.floor(query / radius).astype(np.int64)):
        queries.setdefault(tuple(cell), []).append(i)
    offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
    for cell, ids in queries.items():
        candidates = [i for offset in offsets
                      for i in bins.get((cell[0] + offset[0], cell[1] + offset[1],
                                             cell[2] + offset[2]), ())]
        for start in range(0, len(ids), 128):
            qids = np.asarray(ids[start:start + 128])
            for stop in range(0, len(candidates), 512):
                pids = np.asarray(candidates[stop:stop + 512])
                delta = query[qids, None] - points[pids]
                squared = np.einsum("qpi,qpi->qp", delta, delta)
                best = squared.argmin(axis=1)
                ds = np.sqrt(squared[np.arange(len(qids)), best])
                improve = ds < distances[qids]
                indices[qids[improve]] = pids[best[improve]]
                distances[qids[improve]] = ds[improve]
    return distances, indices


def remove_subset(parent_path, crop_path, lo, extent, tol):
    """Verify raw position overlap before filtering; tol is in scene units.

    Remove every parent position within tol of the crop, then apply exactly
    the parent's fixed-frame alpha/cube selection. Bbox is in box units.
    """
    parent = load_scene_file(parent_path)[0]
    crop = load_scene_file(crop_path)[0]
    if not len(crop):
        raise ValueError("empty crop")
    _, matches = _nearest(parent, crop, tol)
    overlap = float(np.mean(matches >= 0))
    if overlap < 0.99:
        raise ValueError(f"crop is not a subset: overlap {overlap:.6f} < 0.99")
    scene = build_scene_fixed(parent_path, lo, extent)[0]
    crop_mu = ((crop - lo) / extent).astype(np.float32)
    _, removed = _nearest(crop_mu, scene.mu, tol / extent)
    keep = removed < 0
    if keep.all():
        raise ValueError("removed crop has no eligible splats inside the frame")
    return _take(scene, keep), _bbox(scene.mu[~keep]), overlap


def _bbox(points):
    if not len(points):
        raise ValueError("change must contain at least one primitive")
    return np.stack([points.min(axis=0), points.max(axis=0)])


def insert(scene, obj_scene, t):
    """Add an object translated by t in the same coordinates as scene.mu."""
    mu = obj_scene.mu + np.asarray(t, dtype=np.float32)
    return SplatScene(np.concatenate([scene.mu, mu]),
                      np.concatenate([scene.cov, obj_scene.cov]),
                      np.concatenate([scene.amp, obj_scene.amp])), _bbox(mu)


def difference_bundle(S_before, S_after):
    """Signed after minus before; retain sign until visualization/scoring."""
    if np.shape(S_before) != np.shape(S_after):
        raise ValueError("bundles must have the same shape and codebook")
    return np.asarray(S_after) - np.asarray(S_before)


def change_map(D, freqs, grid):
    """Magnitude of the real phasor decode on any (..., 3) point grid."""
    grid = np.asarray(grid, dtype=np.float32)
    decoded = decode_field_phasor(np.atleast_2d(D), freqs, grid.reshape(-1, 3))
    return np.abs(decoded[:, 0]).reshape(grid.shape[:-1])


def _encode(scene, freqs):
    return spectral_bundle(SplatScene(scene.mu, scene.cov, scene.amp[:, :1]),
                           freqs)


def null_threshold(  # noqa: PLR0913, PLR0917
        S_before, scene_before, freqs, grid, rng, sigma_pos,
        sigma_amp, split_frac, quantile=0.99):
    """Pooled voxel quantile of three unchanged re-optimizations.

    The public signature follows the lane contract (hence the argument waiver).

    scene_before must already carry recognition blur, as must S_before.
    S_before is the observed before spectrum, held fixed during calibration.
    This is a voxelwise empirical null, not a familywise error guarantee.
    """
    maps = []
    for _ in range(3):
        other = drift(scene_before, rng, sigma_pos, sigma_amp, split_frac)
        maps.append(change_map(difference_bundle(np.atleast_2d(S_before),
                                                  _encode(other, freqs)),
                               freqs, grid))
    return float(np.quantile(maps, quantile))


def _inside(bbox, grid):
    return np.all((grid >= bbox[0]) & (grid <= bbox[1]), axis=-1)


def iou(change_map, threshold, true_bbox, grid):
    truth = _inside(true_bbox, grid)
    predicted = change_map > threshold
    union = np.count_nonzero(truth | predicted)
    return float(np.count_nonzero(truth & predicted) / union) if union else 1.0


def _grid(size):
    axis = np.linspace(0, 1, size, dtype=np.float32)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)


def primitive_baseline(before, after, radius, grid=None):
    """Absolute difference of radius-capped nearest-center distance fields.

    Default grid is 17 cubed in [0,1]; callers pass their evaluation grid.
    Alpha/covariance and exact duplicate splits are invisible to this control.
    Each method gets its own score quantile from the SAME drift realizations,
    not the same numeric threshold on scores with different units.
    """
    grid = _grid(17) if grid is None else np.asarray(grid)
    query = grid.reshape(-1, 3)
    a = _nearest(before.mu, query, radius)[0]
    b = _nearest(after.mu, query, radius)[0]
    return np.abs(a - b).reshape(grid.shape[:-1])


def _evaluate(before, after, bbox, freqs, grid, settings):
    sigma, position, seed = settings
    clean = render_mip(before, sigma)
    changed = render_mip(after, sigma)
    rng = np.random.default_rng(seed)
    observed = drift(clean, rng, position, 0.2, 0.1)
    other = drift(changed, rng, position, 0.2, 0.1)
    spectrum = _encode(observed, freqs)
    delta = difference_bundle(spectrum, _encode(other, freqs))
    field = change_map(delta, freqs, grid)
    null_seed = seed + 10000
    threshold = null_threshold(spectrum, clean, freqs, grid,
                               np.random.default_rng(null_seed),
                               position, 0.2, 0.1)
    rng = np.random.default_rng(null_seed)
    nulls = [primitive_baseline(observed, drift(clean, rng, position, 0.2, 0.1),
                                sigma, grid) for _ in range(3)]
    baseline_threshold = float(np.quantile(nulls, 0.99))
    baseline = primitive_baseline(observed, other, sigma, grid)
    mass = np.where(field > threshold, field, 0)
    false_mass = None
    if bbox is not None:
        false_mass = float(mass[~_inside(bbox, grid)].sum() / mass.sum()) \
            if mass.sum() else 0.0
    row = {"iou_bundle": iou(field, threshold, bbox, grid)
           if bbox is not None else None,
           "iou_baseline": iou(baseline, baseline_threshold, bbox, grid)
           if bbox is not None else None,
           "false_mass": false_mass, "threshold_bundle": threshold,
           "threshold_baseline": baseline_threshold,
           "change_map": field.ravel().tolist(),
           "baseline_map": baseline.ravel().tolist()}
    return row, (delta, threshold, baseline, baseline_threshold)


def synthetic_fixture(seed=0):
    """Four separated, uniformly populated cubes in an eight-unit world."""
    rng = np.random.default_rng(seed)
    centers = np.array([[2, 2, 2], [6, 2, 6], [2, 6, 6], [6, 6, 2]])
    points = np.concatenate([c + rng.uniform(-0.75, 0.75, (128, 3))
                             for c in centers]).astype(np.float32) / 8
    cov = np.tile(np.eye(3, dtype=np.float32) * (0.12 / 8)**2,
                  (len(points), 1, 1))
    before = SplatScene(points, cov, np.ones((len(points), 1), np.float32))
    return before, _take(before, np.arange(128, len(points))), _bbox(points[:128])


def _figure(figure, payload, freqs, bbox):
    delta, threshold, _, _ = payload
    y = float(bbox[:, 1].mean()) if bbox is not None else 0.5
    grid, shape = slice_grid((0, 1), (0, 1), "y", y,
                             pix=1 / 80)
    field = change_map(delta, freqs, grid).reshape(shape)
    truth = _inside(bbox, grid).reshape(shape) if bbox is not None \
        else np.zeros(shape)
    truth_title = "True bbox" if bbox is not None else "No truth supplied"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, data, title in zip(axes, (field, field > threshold, truth),
                               ("Alpha difference", "Above drift null", truth_title)):
        im = ax.imshow(data, origin="lower", extent=(0, 1, 0, 1))
        ax.set(title=title, xlabel="x / extent", ylabel="z / extent")
        fig.colorbar(im, ax=ax, shrink=0.7)
    figure = Path(figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=140)
    plt.close(fig)


def _ladder(scenes, extent, dim, seed, size, sigmas, positions, figure):
    before, after, bbox = scenes
    # A single codebook covers every blur/drift setting in this run.
    freqs = sample_frequencies(dim, 3, extent / min(sigmas),
                               np.random.default_rng(seed))
    grid = _grid(size)
    rows = []
    for sigma, position in itertools.product(sigmas, positions):
        row, payload = _evaluate(before, after, bbox, freqs, grid,
                                 (sigma / extent, position / extent, seed + 1))
        rows.append({"sigma_rec": sigma, "sigma_pos": position, **row})
        if figure and len(rows) == 1:
            _figure(figure, payload, freqs, bbox)
    return {"dim": dim, "seed": seed, "grid": size, "extent": extent,
            "sigma_amp": 0.2, "split_frac": 0.1, "null_draws": 3,
            "quantile": 0.99, "bbox_box": bbox.tolist() if bbox is not None else None,
            "map_order": "C order, ij-indexed linspace(0, 1, grid) on each axis",
            "frequency_sigma_rho": extent / min(sigmas), "ladder": rows}


def run_synthetic(dim=4096, seed=0, figure=None):
    before, after, bbox = synthetic_fixture(seed)
    return _ladder((before, after, bbox), 8.0, dim, seed, 17,
                   (0.25, 0.5, 1.0), (0, 0.05, 0.1, 0.2), figure)


def _real_inputs(args):
    lo, extent = crop_box(args.frame or args.before)
    before = build_scene_fixed(args.before, lo, extent)[0]
    overlap = None
    if args.remove:
        after, bbox, overlap = remove_subset(args.before, args.remove, lo,
                                             extent, args.tol)
    elif args.insert:
        # Restore the object's physical coordinates before mapping into the
        # parent frame. --at is a translation vector in scene units.
        obj_lo, obj_extent = crop_box(args.insert)
        obj = build_scene_fixed(args.insert, obj_lo, obj_extent)[0]
        obj = SplatScene((obj.mu * obj_extent + obj_lo - lo) / extent,
                         obj.cov * (obj_extent / extent)**2, obj.amp)
        after, bbox = insert(before, obj, np.asarray(args.at) / extent)
    else:
        after = build_scene_fixed(args.after, lo, extent)[0]
        bbox = None
    return before, after, bbox, lo, extent, overlap



def _scene_tile(scene, corner, tile):
    """Inclusive center/alpha mask from tile_scenes; normalize to tile units.

    Inputs are physical coordinates with alpha in channel zero. Keep this
    helper local: place_recognition.py is outside this lane's edit matrix.
    """
    mask = (scene.amp[:, 0] >= ALPHA_MIN) & np.all(
        (scene.mu >= corner) & (scene.mu <= corner + tile), axis=1)
    return SplatScene(((scene.mu[mask] - corner) / tile).astype(np.float32),
                      (scene.cov[mask] / tile**2).astype(np.float32),
                      scene.amp[mask])


def _tile_score(before, after, freqs, sigma_box, whiten):
    if not before.n or not after.n:
        return 1.0 if before.n == after.n else 0.0
    fps = tile_fingerprints([(np.zeros(3), s, 1.0) for s in (before, after)],
                            freqs, sigma_box, [0.0])
    score = correlate(fps[0, 0], fps[1, 0], freqs,
                      np.zeros((1, 3), np.float32), whiten)[0]
    # Self-correlation can exceed one by float32 rounding.
    return float(np.clip(score, -1, 1))


def tile_change(  # noqa: PLR0913, PLR0917
        before_scene, after_scene, lo, extent, tile, overlap, freqs,
        sigma_box, rng, drift_kw, yaws=1, whiten=1.0):
    """Fixed-pose change scores on ONE physical lattice; no pose search.

    Scene centers, lo, extent, tile and drift positions are physical units;
    sigma_box and frequencies are in normalized tile units. The signature
    follows the lane contract. yaws must be one: registration is not change.
    A single held-out drift realization supplies each tile's null. The cutoff
    is that tile's null minus three population standard deviations across
    retained tiles' nulls. Empty-before nulls are one (empty stays empty),
    while one-sided occupancy always registers as change. Both-empty tiles
    remain in the output lattice but are excluded from null pooling.
    The primitive control takes maxima on a 17-cubed local distance-field
    grid, using the SAME null scene and per-tile mean-plus-three-sigma rule.
    These empirical definitions are ours, not a reproduced published method.
    """
    if yaws != 1 or not 0 <= whiten <= 1:
        raise ValueError("tiles require yaws=1 and whiten in [0,1]")
    if not np.isfinite(sigma_box) or sigma_box <= 0:
        raise ValueError("sigma_box must be finite and positive")
    corners = np.asarray(tile_lattice(lo, extent, tile, overlap))
    null_scene = drift(before_scene, rng, **drift_kw)
    scores, nulls, baseline, baseline_nulls, counts = [], [], [], [], []
    grid = _grid(17)
    for corner in corners:
        a, b, n = [_scene_tile(s, corner, tile)
                   for s in (before_scene, after_scene, null_scene)]
        counts.append((a.n, b.n))
        scores.append(_tile_score(a, b, freqs, sigma_box, whiten))
        nulls.append(_tile_score(a, n, freqs, sigma_box, whiten))
        baseline.append(float(primitive_baseline(a, b, sigma_box, grid).max()))
        baseline_nulls.append(float(
            primitive_baseline(a, n, sigma_box, grid).max()))
    counts = np.asarray(counts)
    retained = counts.any(axis=1)
    nulls, scores = np.asarray(nulls), np.asarray(scores)
    baseline, baseline_nulls = np.asarray(baseline), np.asarray(baseline_nulls)
    spread = float(nulls[retained].std()) if retained.any() else 0.0
    base_spread = float(baseline_nulls[retained].std()) if retained.any() else 0.0
    thresholds = nulls - 3 * spread
    base_thresholds = baseline_nulls + 3 * base_spread
    # Numerical tolerance is far below the calibrated drift drops.
    changed = retained & (scores < thresholds - 1e-6)
    changed |= (counts[:, 0] == 0) != (counts[:, 1] == 0)
    return {"corners": corners.tolist(), "counts": counts.tolist(),
            "retained": retained.tolist(), "scores": scores.tolist(),
            "nulls": nulls.tolist(), "null_sigma": spread,
            "thresholds": thresholds.tolist(), "drop": (nulls - scores).tolist(),
            "changed": changed.tolist(), "baseline_scores": baseline.tolist(),
            "baseline_nulls": baseline_nulls.tolist(),
            "baseline_thresholds": base_thresholds.tolist(),
            "baseline_changed": (retained & (
                baseline > base_thresholds + 1e-9)).tolist()}


def _tile_truth(corners, tile, bboxes):
    """Quantize the union of true edit bboxes by inclusive tile intersection."""
    corners = np.asarray(corners)
    truth = np.zeros(len(corners), dtype=bool)
    for bbox in np.asarray(bboxes).reshape(-1, 2, 3):
        truth |= np.all((corners <= bbox[1]) &
                        (corners + tile >= bbox[0]), axis=1)
    return truth


def _mask_iou(predicted, truth):
    predicted = np.asarray(predicted, dtype=bool)
    union = np.count_nonzero(predicted | truth)
    return float(np.count_nonzero(predicted & truth) / union) if union else 1.0


def _physical_scene(scene, lo, extent):
    return SplatScene(scene.mu * extent + lo, scene.cov * extent**2, scene.amp)


def _tile_fixture(seed):
    before, removed, bbox = synthetic_fixture(seed)
    obj = _take(before, np.arange(128))
    after, added_bbox = insert(removed, obj, [0.5, 0, 0])
    return before, after, np.stack([bbox, added_bbox])


def _tile_metrics(row, tile, bboxes, threshold):
    if threshold is not None:
        occupancy = np.asarray(row["counts"]) == 0
        row["thresholds"] = [threshold] * len(row["scores"])
        row["changed"] = (np.asarray(row["retained"]) & (
            (np.asarray(row["scores"]) < threshold) |
            (occupancy[:, 0] != occupancy[:, 1]))).tolist()
    truth = _tile_truth(row["corners"], tile, bboxes) if bboxes is not None else None
    row.update({"truth": truth.tolist() if truth is not None else None,
                "iou_bundle_tile": _mask_iou(row["changed"], truth)
                if truth is not None else None,
                "iou_baseline_tile": _mask_iou(row["baseline_changed"], truth)
                if truth is not None else None,
                "false_tiles": int(np.count_nonzero(
                    np.asarray(row["changed"]) & ~truth))
                if truth is not None else None})


def _tile_change_figure(path, row, tile, extent):
    corners = np.asarray(row["corners"])
    layers = np.unique(corners[:, 1])
    layers = layers[np.unique(np.linspace(0, len(layers) - 1,
                                          min(4, len(layers))).astype(int))]
    fig, axes = plt.subplots(len(layers), 3, figsize=(12, 3.5 * len(layers)),
                             squeeze=False, constrained_layout=True)
    for column, (key, title) in enumerate((
            ("drop", "Null minus observed correlation"),
            ("changed", "Detected tiles"), ("truth", "True bbox tiles"))):
        values = np.asarray(row[key] if row[key] is not None
                            else np.zeros(len(corners)), dtype=float)
        norm = plt.Normalize(-1 if key == "drop" else 0, 1)
        cmap = plt.get_cmap("coolwarm" if key == "drop" else "viridis")
        for layer, ax in zip(layers, axes[:, column]):
            for corner, value in zip(corners, values):
                if corner[1] == layer:
                    ax.add_patch(Rectangle((corner[0], corner[2]), tile, tile,
                                           facecolor=cmap(norm(value)),
                                           edgecolor="gray", linewidth=0.5))
            ax.set(xlim=(corners[:, 0].min(), corners[:, 0].max() + tile),
                   ylim=(corners[:, 2].min(), corners[:, 2].max() + tile),
                   title=f"{title}\ny tile [{layer:g}, {layer + tile:g}]",
                   xlabel="x (scene units)", ylabel="z (scene units)",
                   aspect="equal")
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                     ax=axes[:, column].tolist(), shrink=0.8)
    fig.suptitle(f"Tile edge {tile:g}; frame extent {extent:g}; "
                 f"sigma {row['sigma_rec']:g}, drift {row['sigma_pos']:g}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _run_tile_change(args):
    if args.synthetic:
        before, after, bboxes = _tile_fixture(args.seed)
        lo, extent, subset_overlap = np.zeros(3), 8.0, None
    else:
        before, after, bboxes, lo, extent, subset_overlap = _real_inputs(args)
    before, after = [_physical_scene(s, lo, extent) for s in (before, after)]
    bboxes = bboxes * extent + lo if bboxes is not None else None
    freqs = sample_frequencies(args.dim, 3, args.tile / min(args.sigmas),
                               np.random.default_rng(args.seed))
    rows = []
    for sigma, position in itertools.product(args.sigmas, args.positions):
        drift_kw = {"sigma_pos": position, "sigma_amp": 0.2, "split_frac": 0.1}
        rng = np.random.default_rng(args.seed + 1)
        observed, other = [drift(s, rng, **drift_kw) for s in (before, after)]
        row = tile_change(observed, other, lo, extent, args.tile, args.overlap,
                          freqs, sigma / args.tile,
                          np.random.default_rng(args.seed + 10001), drift_kw,
                          whiten=args.whiten)
        _tile_metrics(row, args.tile, bboxes, args.threshold)
        row.update({"sigma_rec": sigma, "sigma_pos": position})
        rows.append(row)
        if args.figure and len(rows) == 1:
            _tile_change_figure(args.figure, row, args.tile, extent)
    return {"mode": "tiles", "dim": args.dim, "seed": args.seed,
            "tile": args.tile, "overlap": args.overlap, "lo": lo.tolist(),
            "extent": extent, "whiten": args.whiten, "yaws": 1,
            "translation": [0, 0, 0], "sigma_amp": 0.2, "split_frac": 0.1,
            "null_draws": 1, "baseline_grid": 17,
            "frequency_sigma_rho": args.tile / min(args.sigmas),
            "bboxes_units": bboxes.tolist() if bboxes is not None else None,
            "subset_overlap": subset_overlap, "ladder": rows}


def _validate_tile_change(parser, args):
    if args.tile is None:
        if args.threshold is not None:
            parser.error("--threshold requires --tile")
        return
    if not np.isfinite(args.tile) or args.tile <= 0:
        parser.error("--tile must be positive and finite")
    if not 0 <= args.overlap < 1 or not 0 <= args.whiten <= 1:
        parser.error("--overlap must be in [0,1); --whiten in [0,1]")
    if args.threshold is not None and not -1 <= args.threshold <= 1:
        parser.error("--threshold must be in [-1,1]")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("before", nargs="?")
    parser.add_argument("after", nargs="?",
                        help="observed capture; replaced by --remove/--insert")
    parser.add_argument("--frame")
    change = parser.add_mutually_exclusive_group()
    change.add_argument("--remove")
    change.add_argument("--insert")
    parser.add_argument("--at", type=float, nargs=3)
    parser.add_argument("--sigma-units", type=float)
    parser.add_argument("--grid", type=int)
    parser.add_argument("--drift", default="0,0.05,0.1,0.2")
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--tile", type=float, help="tile edge in scene units")
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--whiten", type=float, default=1.0)
    parser.add_argument("--threshold", type=float,
                        help="override drift cutoff; one-sided empty stays changed")
    # Positionals may follow options (`out before --dim 32 after`); on
    # Python 3.9 parse_args leaves a trailing positional unrecognised when
    # an earlier one is optional, which failed CI's 3.9 job on every push.
    args = parser.parse_intermixed_args(argv)
    _validate(parser, args)
    _validate_tile_change(parser, args)
    if args.tile is not None:
        result = _run_tile_change(args)
    elif args.synthetic:
        result = _ladder(synthetic_fixture(args.seed), 8.0, args.dim, args.seed,
                         args.grid, args.sigmas, args.positions, args.figure)
    else:
        result = _run_real(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _validate(parser, args):
    args.grid = args.grid if args.grid is not None else (17 if args.synthetic
                                                        else 48)
    args.sigmas = ((0.25, 0.5, 1.0) if args.synthetic else (0.5,)) \
        if args.sigma_units is None else (args.sigma_units,)
    args.sigma_units = args.sigmas[0]
    if args.dim < 1 or args.grid < 2 or args.seed < 0:
        parser.error("dim/grid/seed out of range")
    values = [args.sigma_units, args.tol]
    if not np.isfinite(values).all() or min(values) <= 0:
        parser.error("sigma-units and tol must be finite and positive")
    try:
        args.positions = [float(value) for value in args.drift.split(",")]
    except ValueError:
        parser.error("drift must be comma-separated numbers")
    if not np.isfinite(args.positions).all() or min(args.positions) < 0:
        parser.error("drift must be finite and nonnegative")
    if args.synthetic:
        if any([args.before, args.after, args.frame, args.remove, args.insert,
                args.at]):
            parser.error("synthetic cannot be combined with real inputs")
    elif not args.before or not (args.after or args.remove or args.insert):
        parser.error("supply BEFORE and AFTER, --remove or --insert")
    if bool(args.insert) != bool(args.at):
        parser.error("--insert requires --at, and --at requires --insert")
    if args.at and not np.isfinite(args.at).all():
        parser.error("--at must be finite")


def _run_real(args):
    before, after, bbox, lo, extent, overlap = _real_inputs(args)
    result = _ladder((before, after, bbox), extent, args.dim, args.seed, args.grid,
                     (args.sigma_units,), args.positions, args.figure)
    result.update({"lo": lo.tolist(), "overlap": overlap,
                   "has_truth": bbox is not None})
    return result


if __name__ == "__main__":
    main()
