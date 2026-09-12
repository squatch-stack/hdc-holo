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

from bench.place_recognition import build_scene_fixed, crop_box
from holo.capture import load_scene_file, render_mip, slice_grid
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
    # Positionals may follow options (`out before --dim 32 after`); on
    # Python 3.9 parse_args leaves a trailing positional unrecognised when
    # an earlier one is optional, which failed CI's 3.9 job on every push.
    args = parser.parse_intermixed_args(argv)
    _validate(parser, args)
    if args.synthetic:
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
