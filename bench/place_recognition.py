"""Place recognition by sampled spectra and a searched Fourier shift theorem.

Raw cosine fails because translation multiplies a spectrum by exp(-i w.t).
We search that ramp, including local refinement, and re-encode yaw hypotheses
about the unit cube's y axis. Between yaw steps the peak decays with the blur
autocorrelation: sigma_rec trades angular tolerance for discrimination.
Independent per-capture normalization changes physical scale and cannot be
undone by a phase ramp; build_scene_fixed preserves a supplied physical frame.

Prior art checked at https://arxiv.org/abs/2604.12331 (HyperLiDAR: HDC-based
post-deployment semantic segmentation, not this place-recognition algorithm)
and https://arxiv.org/abs/2209.02000 (Visual Odometry with Neuromorphic Resonator
Networks: VSA working memory and resonator inference of position/orientation).
FTO caveat supplied by the research brief: US patent 12,014,263, VSA encoding
of continuous spaces; research only. This is not a freedom-to-operate opinion.

Two control limitations matter. Shuffling positions preserves total mass and
individual splat envelopes, NOT the summed spectrum's envelope. Identical
splats are unchanged by permutation, so structureless scenes can score like
their scramble. Radial power is exactly translation invariant (up to float32
rounding). Its continuous azimuthal average is yaw invariant, but bins from a
finite iid frequency sample are only approximately so. No exact yaw-invariance
claim is possible for arbitrary spectra sampled at these fixed frequencies.
The max over G*K hypotheses raises the unrelated noise floor roughly by
sqrt(2*log(G*K)); blur and coherent mass further reduce effective dimension.
Empirical scrambled scores use the same search, including refinement, rather
than assuming the nominal 1/sqrt(2*d) noise model is calibrated.

Example CPU smoke run (12 descriptors from three places):
python -m bench.place_recognition /tmp/place.json --synthetic 3 --numpy \
    --dim 512 --grid 9 --yaws 4 --scrambles 4
Real captures must share coordinates; --lo X Y Z --extent E fixes their cube.
Known real partners can be supplied as repeated --partner I J (zero-based).
Offsets[i,j] locate query j after the winning yaw relative to reference i;
they are normalized units, and need multiplying by extent for physical units.

Position scrambling is a weak null on real captures: on the 5090 with
wilsons-creek, its gun crop and cannon (8192-d, 4 yaws), scrambled copies
scored 0.80-0.999 against their own source, because permuting positions
among 400k similar splats leaves the density field nearly unchanged. The
default null is therefore the phase surrogate (random phases, identical
magnitudes): it shares the radial control exactly and has no arrangement,
so it measures precisely what phase correlation adds. --null scramble
keeps the old model for synthetic scenes with distinct landmarks.

The raw cross-power spectrum is dominated by the low-frequency envelope of a
mass-centred capture: on the 5090 the four wide outdoor parents (oak,
redrock, research-library, wilsons-creek; 480k splats, 30-40 unit cubes)
scored 0.70-0.96 against each other at zero offset, and no known partner
ranked first. That peak is the blob, not the arrangement. --whiten applies
the PHAT exponent (Knapp & Carter 1976): each component of conj(A)B is
divided by its magnitude, so only phase votes and the peak measures
arrangement alone. It also trades away every magnitude, which is the
radial control's whole content, so whitened and raw scores answer
different questions and the results note reports both.

Prediction tested here: voxel alpha equalisation preserves a unit-mass uniform
field, balances dense/faint copies, lets a light object find itself beside a
heavy competitor, and retains crop localisation in a shared parent cube.
These are our voxel-support definitions, not a reproduction of prior art.
Equal alpha per voxel does not equal integrated Gaussian mass when covariances
vary. Voxel boundaries also make reweighting translation/yaw dependent.
The bright-landmark crop control loses high similarity under voxel weighting,
although its location and the support-dominant crop control survive.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from holo import accel
from holo.capture import (
    ALPHA_MIN,
    S_HI,
    S_LO,
    build_scene,
    load_scene_file,
    quat_to_rot,
    render_mip,
    weighted_quantile,
)
from holo.spectral import (
    SplatScene,
    decode_field_phasor,
    sample_frequencies,
    spectral_bundle,
)


def build_scene_fixed(path, lo, extent, alpha_min=ALPHA_MIN, s_lo=S_LO,
                      s_hi=S_HI):
    """Mirror capture.build_scene with a fixed cube to retain physical scale."""
    lo = np.asarray(lo, dtype=np.float64)
    if lo.shape != (3,) or not np.isfinite(lo).all():
        raise ValueError("lo must contain three finite coordinates")
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("extent must be finite and positive")
    pos, scale, rgba, quat = load_scene_file(path)
    keep = rgba[:, 3] >= alpha_min
    pos, scale, rgba, quat = pos[keep], scale[keep], rgba[keep], quat[keep]
    hi = lo + extent
    inside = np.all((pos >= lo) & (pos <= hi), axis=1)
    pos, scale, rgba, quat = (pos[inside], scale[inside], rgba[inside],
                              quat[inside])
    if not len(pos):
        raise ValueError("fixed crop contains no splats above the alpha floor")
    pos = ((pos - lo) / extent).astype(np.float32)
    scale = np.clip(scale / extent, s_lo, s_hi)
    rots = quat_to_rot(quat)
    cov = np.einsum("nij,nj,nkj->nik", rots, scale**2, rots).astype(np.float32)
    alpha = rgba[:, 3:4]
    amp = np.concatenate([alpha, alpha * rgba[:, :3]], axis=1).astype(np.float32)
    box = ((hi - lo) / extent).astype(np.float32)
    return SplatScene(pos, cov, amp), scale.max(axis=1), box


def crop_box(path, alpha_min=ALPHA_MIN, crop_quantile=0.75, crop_margin=1.2):
    """Return build_scene's mass-centred cube as (lo, extent) for a capture.

    build_scene computes the cube and does not return it. Recomputing it
    lets a crop be encoded in its parent's frame (--frame parent.spz)
    without retyping the rounded numbers build_scene prints.
    """
    pos, _, rgba, _ = load_scene_file(path)
    keep = rgba[:, 3] >= alpha_min
    pos, a = pos[keep], rgba[keep, 3]
    center = np.array([weighted_quantile(pos[:, i], a, 0.5) for i in range(3)])
    radius = weighted_quantile(np.abs(pos - center).max(axis=1), a, crop_quantile)
    return center - crop_margin * radius, float(2 * crop_margin * radius)


def phase_surrogate(fp, rng):
    """Randomise every phase and keep every magnitude.

    This is the null that shares the radial control exactly and has no
    arrangement at all, which is what phase correlation is supposed to add.
    Position scrambling is not that null on a capture: permuting positions
    among 400k similar splats leaves the density field nearly unchanged.
    """
    fp = np.asarray(fp)
    phases = rng.uniform(-np.pi, np.pi, fp.shape)
    return (fp * np.exp(1j * phases)).astype(np.complex64)


def flatten(scene, sigma, mode="none"):
    """Equalise voxel alpha before blur; preserve geometry and other channels.

    A uniform field is preserved up to a global scale (exactly for unit alpha
    per occupied voxel). Log weights use the median positive voxel alpha mass.
    Nonpositive-mass voxels have zero alpha. The input is never mutated.
    """
    if mode == "none":
        return scene
    if mode not in ("voxel", "log"):
        raise ValueError("flatten mode must be none, voxel or log")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(scene.mu).all() or not np.isfinite(scene.amp[:, 0]).all():
        raise ValueError("positions and alpha must be finite")
    _, inverse = np.unique(np.floor(scene.mu / sigma), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)  # NumPy 2.0 axis-wise inverse shape compatibility.
    mass = np.bincount(inverse, weights=scene.amp[:, 0]).astype(np.float64)
    positive = mass > 0
    target = np.ones_like(mass)
    if mode == "log" and positive.any():
        target = np.log1p(np.maximum(mass, 0) / np.median(mass[positive]))
    weight = np.divide(target, mass, out=np.zeros_like(mass), where=positive)
    amp = scene.amp.astype(np.result_type(scene.amp.dtype, np.float32), copy=True)
    amp[:, 0] = scene.amp[:, 0] * weight[inverse]
    return SplatScene(scene.mu, scene.cov, amp)


_flatten_scene = flatten


def fingerprint(scene, freqs, sigma_rec, flatten="none"):
    """Encode only alpha after mass-preserving blur at recognition resolution."""
    if not np.isfinite(sigma_rec) or sigma_rec <= 0:
        raise ValueError("sigma_rec must be finite and positive")
    alpha = SplatScene(scene.mu, scene.cov, scene.amp[:, :1])
    alpha = _flatten_scene(alpha, sigma_rec, flatten)
    return spectral_bundle(render_mip(alpha, sigma_rec), freqs)[0]


def yaw_scene(scene, theta):
    """Rotate centers and anisotropic covariances through the unit box center."""
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    mu = ((scene.mu - 0.5) @ rot.T + 0.5).astype(np.float32)
    cov = (rot @ scene.cov @ rot.T).astype(np.float32)
    return SplatScene(mu, cov, scene.amp.copy())


def scramble(scene, rng):
    """Reassign positions while retaining each splat's covariance and mass."""
    return SplatScene(scene.mu[rng.permutation(scene.n)].copy(),
                      scene.cov.copy(), scene.amp.copy())


def translation_grid(size, limit=0.25):
    """Return a Cartesian grid; refinement stays within these search bounds."""
    axis = np.linspace(-limit, limit, size, dtype=np.float32)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)


def correlate(a, b, freqs, grid, whiten=0.0):
    """Find B's translation relative to A; undo the decoder's 1/d scaling.

    whiten is the PHAT exponent: 0 keeps the raw cross-power spectrum, 1
    divides every component by its magnitude so only phase, i.e.
    arrangement, votes. Components below 1e-6 of the largest are floored
    rather than amplified. The score is the peak over its aligned maximum.

    A Cartesian grid is recommended. Three 5-per-axis local searches shrink
    the coarse spacing eightfold; singleton axes stay fixed. Zero-energy
    spectra score zero at the first candidate, so empty signals never match.
    """
    a, b = np.asarray(a), np.asarray(b)
    grid = np.asarray(grid, np.float32)
    if a.shape != (len(freqs),) or b.shape != a.shape:
        raise ValueError("fingerprints must match the frequency count")
    if grid.ndim != 2 or grid.shape[1] != freqs.shape[1] or not len(grid):
        raise ValueError("grid must be a nonempty (G, spatial dimensions) array")
    if not 0 <= whiten <= 1:
        raise ValueError("whiten must lie in [0, 1]")
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0:
        return 0.0, grid[0].copy()
    product = np.conj(a) * b
    if whiten:
        mag = np.abs(product)
        product = product * np.maximum(mag, 1e-6 * mag.max()) ** (-whiten)
        norm = float(np.abs(product).sum())
    product = product[None, :].astype(np.complex64)
    values = decode_field_phasor(product, freqs, grid)[:, 0]
    idx = int(np.argmax(values))
    peak, point = float(values[idx]), grid[idx].copy()
    axes = [np.unique(grid[:, i]) for i in range(grid.shape[1])]
    step = np.array([np.max(np.diff(x)) if len(x) > 1 else 0.0 for x in axes])
    lower, upper = grid.min(axis=0), grid.max(axis=0)
    for _ in range(3):
        local_axes = [np.unique(np.clip(p + np.linspace(-h, h, 5), lo, hi))
                      for p, h, lo, hi in zip(point, step, lower, upper)]
        local = np.stack(np.meshgrid(*local_axes, indexing="ij"), -1)
        local = local.reshape(-1, grid.shape[1]).astype(np.float32)
        values = decode_field_phasor(product, freqs, local)[:, 0]
        idx = int(np.argmax(values))
        if values[idx] > peak:
            peak, point = float(values[idx]), local[idx].copy()
        step /= 2
    return float(np.clip(peak * len(freqs) / norm, -1, 1)), point


def radial_power(fp, freqs, n_bins=32):
    """Bin mean power by (horizontal radius, signed vertical frequency).

    Translation cancels exactly in power. Fixed frequency-derived bin edges
    make different descriptors comparable; empty bins are zero. Finite Monte
    Carlo azimuthal coverage makes yaw invariance approximate, not exact.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    radius = np.linalg.norm(freqs[:, [0, 2]], axis=1)
    coords = np.column_stack([radius, freqs[:, 1]])
    edges = [np.linspace(0, max(float(radius.max()), 1e-12), n_bins + 1),
             np.linspace(float(freqs[:, 1].min()) - 1e-12,
                         float(freqs[:, 1].max()) + 1e-12, n_bins + 1)]
    count, _ = np.histogramdd(coords, bins=edges)
    power, _ = np.histogramdd(coords, bins=edges, weights=np.abs(fp)**2)
    return np.divide(power, count, out=np.zeros_like(power),
                     where=count > 0).astype(np.float32).ravel()


def _best_yaw(a, candidates, freqs, grid, whiten=0.0):
    results = [correlate(a, b, freqs, grid, whiten) for b in candidates]
    k = int(np.argmax([r[0] for r in results]))
    return results[k][0], results[k][1], k


def similarity_matrix(fps, freqs, grid, yaw_fps=None, whiten=0.0):
    """Score all ordered pairs; yaw_fps[j,k] rotates query j by angle k.

    Search bounds and discrete rotations can make this matrix asymmetric;
    do not mirror scores or offsets across the diagonal.
    """
    n = len(fps)
    candidates = np.asarray(fps)[:, None, :] if yaw_fps is None else yaw_fps
    scores = np.empty((n, n), np.float64)
    offsets = np.empty((n, n, 3), np.float32)
    for i, a in enumerate(fps):
        for j in range(n):
            scores[i, j], offsets[i, j], _ = _best_yaw(a, candidates[j], freqs,
                                                       grid, whiten)
    return scores, offsets


def synthetic_scene(rng):
    """Sparse landmarks among faint splats exercise position/mass association.

    This deliberately heterogeneous control is not representative of uniform
    point clouds, for which a position permutation is the identical scene.
    """
    n = 160
    mu = rng.uniform(0.1, 0.9, (n, 3)).astype(np.float32)
    scales = rng.uniform(0.006, 0.01, (n, 3))
    cov = np.array([np.diag(s**2) for s in scales], np.float32)
    amp = np.full((n, 1), 0.001, np.float32)
    amp[:12] = rng.uniform(0.5, 1, (12, 1))
    return SplatScene(mu, cov, amp)


def _inputs(args, rng):
    scenes, labels, groups = [], [], []
    if args.paths:
        lo, extent = args.lo, args.extent
        if args.frame is not None:
            lo, extent = crop_box(args.frame)
            print(f"frame from {Path(args.frame).name}: cube of {extent:.4f} "
                  f"scene units at {np.round(lo, 4)}")
        for path in args.paths:
            if lo is None:
                scene, _, _ = build_scene(path)
            else:
                scene, _, _ = build_scene_fixed(path, lo, extent)
            scenes.append(scene)
            labels.append(Path(path).name)
            groups.append(len(groups))
        return scenes, labels, groups
    for i in range(args.synthetic):
        scene = synthetic_scene(rng)
        shifted = SplatScene(scene.mu + np.array([0.07, -0.04, 0.03], np.float32),
                             scene.cov.copy(), scene.amp.copy())
        variants = [scene, shifted, yaw_scene(scene, 2 * np.pi / args.yaws),
                    scramble(scene, rng)]
        scenes.extend(variants)
        labels.extend(f"place{i}/{v}" for v in ("base", "shift", "yaw", "scramble"))
        groups.extend([i * 2, i * 2, i * 2, i * 2 + 1])
    return scenes, labels, groups


def _yaw_fingerprints(scene, angles, freqs, sigma, flatten="none"):
    return np.stack([fingerprint(yaw_scene(scene, t), freqs, sigma, flatten)
                     for t in angles])


def _calibrate(scenes, fps, yaw_fps, angles, freqs, grid, args, rng):
    samples = []
    for m in range(args.scrambles):
        i = m % len(scenes)
        if args.null == "phase":
            candidates = phase_surrogate(yaw_fps[i], rng)
        else:
            candidates = _yaw_fingerprints(scramble(scenes[i], rng), angles,
                                           freqs, args.sigma, args.flatten)
        samples.append(_best_yaw(fps[i], candidates, freqs, grid, args.whiten)[0])
    return {"null": args.null, "samples": samples, "mean": float(np.mean(samples)),
            "sigma": float(np.std(samples)), "p95": float(np.percentile(samples, 95)),
            "max": float(np.max(samples)), "count": len(samples)}


def _retrieval(scores, groups, partners, noise_sigma):
    known = np.equal.outer(groups, groups)
    for i, j in partners:
        known[i, j] = known[j, i] = True
    np.fill_diagonal(known, False)
    reports = []
    for j in range(len(scores)):
        positives = known[:, j]
        if not positives.any():
            continue
        negatives = ~positives
        negatives[j] = False
        column = scores[:, j].copy()
        column[j] = -np.inf
        positive = float(column[positives].max())
        negative = float(column[negatives].max()) if negatives.any() else None
        separation = ((positive - negative) / noise_sigma
                      if negative is not None and noise_sigma > 0 else None)
        reports.append({"query": j, "rank1_hit": bool(positives[column.argmax()]),
                        "positive": positive, "best_negative": negative,
                        "separation_sigma": separation})
    return reports


def tile_lattice(lo, extent, tile, overlap):
    """Cover a cube with fixed-size tiles, including a flush final tile.

    Cubes smaller than one tile receive one tile at lo. Interior strides
    are tile*(1-overlap); the final stride may be shorter to cover the edge.
    """
    lo = np.asarray(lo, dtype=float)
    if lo.shape != (3,) or not np.isfinite(lo).all():
        raise ValueError("lo must contain three finite coordinates")
    if not np.isfinite([extent, tile, overlap]).all():
        raise ValueError("tile lattice parameters must be finite")
    if min(extent, tile) <= 0 or not 0 <= overlap < 1:
        raise ValueError("positive extents and overlap in [0, 1) required")
    end = max(extent - tile, 0)
    axis = np.arange(0, end, tile * (1 - overlap))
    axis = np.unique(np.append(axis, end))
    return [lo + p for p in np.array(np.meshgrid(
        axis, axis, axis, indexing="ij")).reshape(3, -1).T]


def tile_scenes(path, tile, overlap, min_mass, alpha_min=ALPHA_MIN):
    """Load once; retain fixed physical tiles by share of total capture alpha.

    Shares use alpha, not Gaussian volume, and overlap means their sum can
    exceed one. The lattice covers crop_box's cube, not the entire halo.
    """
    if not np.isfinite(min_mass) or not 0 <= min_mass <= 1:
        raise ValueError("min_mass must lie in [0, 1]")
    data = load_scene_file(path)
    pos, _, rgba, _ = data
    alpha = rgba[:, 3]
    total = float(alpha.sum(dtype=np.float64))
    if total <= 0 or not np.any(alpha >= alpha_min):
        return []
    tiles = []
    # Reuse the public fixed-frame loader without rereading a capture per tile.
    with patch(__name__ + ".load_scene_file", return_value=data):
        lo, extent = crop_box(path, alpha_min)
        for corner in tile_lattice(lo, extent, tile, overlap):
            inside = (alpha >= alpha_min) & np.all(
                (pos >= corner) & (pos <= corner + tile), axis=1)
            share = float(alpha[inside].sum(dtype=np.float64)) / total
            if inside.any() and share >= min_mass:
                scene, _, _ = build_scene_fixed(path, corner, tile, alpha_min)
                tiles.append((corner, scene, share))
    return tiles


def tile_fingerprints(tiles, freqs, sigma_box, angles, flatten="none"):
    """Encode every tile with the run's single physical-resolution codebook."""
    if not tiles:
        return np.empty((0, len(angles), len(freqs)), np.complex64)
    return np.stack([_yaw_fingerprints(s, angles, freqs, sigma_box, flatten)
                     for _, s, _ in tiles])


def _tile_candidates(tile_fps, owners, freqs, prefilter):
    # Use every yaw's radial descriptor to reduce finite-sample yaw bias.
    power = np.array([[radial_power(fp, freqs) for fp in yaws]
                      for yaws in tile_fps])
    power /= np.maximum(np.linalg.norm(power, axis=2, keepdims=True), 1e-30)
    candidates = []
    for j, owner in enumerate(owners):
        indices = np.flatnonzero(owners != owner)
        similarity = np.einsum("ib,kb->ik", power[indices, 0], power[j]).max(axis=1)
        order = np.argsort(-similarity, kind="stable")
        candidates.append(indices[order[:prefilter] if prefilter else order])
    return candidates


def _tile_noise(samples):
    values = np.asarray(samples, dtype=float)
    return {"null": "phase", "samples": values.tolist(), "count": len(values),
            "mean": float(values.mean()), "sigma": float(values.std()),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def _capture_maxima(scores, offsets, yaws, owners):
    n_caps = int(owners.max()) + 1
    captures = np.full((n_caps, n_caps), -np.inf)
    best = [[None for _ in range(n_caps)] for _ in range(n_caps)]
    for i, j in zip(*np.where(np.isfinite(scores))):
        a, b = int(owners[i]), int(owners[j])
        if scores[i, j] > captures[a, b]:
            captures[a, b] = scores[i, j]
            best[a][b] = {"reference_tile": int(i), "query_tile": int(j),
                          "offset_box": offsets[i, j].tolist(),
                          "yaw_index": int(yaws[i, j])}
    return captures, best


def tile_matrix(tile_fps, owners, freqs, grid, whiten, prefilter, rng,
                scrambles=1):
    """Columns are queries; masked/unscored pairs are -inf internally.

    Radial cosine is translation invariant; finite-sample yaw invariance is
    approximate. The null randomises query phases, preserving radial selection,
    and repeats exactly the selected database/yaw/translation maximum for each
    query tile. Capture null samples additionally maximise over its query tiles.
    These definitions are ours, using the existing Fourier correlator.
    """
    owners = np.asarray(owners, dtype=int)
    if prefilter < 0 or scrambles < 1:
        raise ValueError("prefilter must be nonnegative; scrambles positive")
    if len(owners) != len(tile_fps) or len(np.unique(owners)) < 2:
        raise ValueError("matching requires tiles from at least two captures")
    if not np.array_equal(np.unique(owners), np.arange(owners.max() + 1)):
        raise ValueError("owners must be contiguous capture indices")
    n = len(owners)
    candidates = _tile_candidates(tile_fps, owners, freqs, prefilter)
    scores = np.full((n, n), -np.inf)
    offsets = np.full((n, n, 3), np.nan, np.float32)
    yaws = np.full((n, n), -1, dtype=int)
    samples = np.empty((scrambles, n))
    started = time.monotonic()
    for j, indices in enumerate(candidates):
        if j % 25 == 0:
            elapsed = time.monotonic() - started
            print(f"tile_matrix: query tile {j}/{n}, {elapsed:.0f} s",
                  file=sys.stderr, flush=True)
        for i in indices:
            scores[i, j], offsets[i, j], yaws[i, j] = _best_yaw(
                tile_fps[i, 0], tile_fps[j], freqs, grid, whiten)
        for draw in range(scrambles):
            surrogate = phase_surrogate(tile_fps[j], rng)
            samples[draw, j] = max(
                _best_yaw(tile_fps[i, 0], surrogate, freqs, grid, whiten)[0]
                for i in indices)
    captures, best = _capture_maxima(scores, offsets, yaws, owners)
    capture_samples = np.array([samples[:, owners == c].max(axis=1)
                                for c in range(len(captures))])
    return {"tile_scores": scores, "capture_scores": captures,
            "tile_offsets": offsets, "tile_yaws": yaws, "best_pairs": best,
            "noise": _tile_noise(samples.ravel()),
            "capture_noise": _tile_noise(capture_samples.ravel()),
            "null_per_tile": samples.T.tolist(),
            "pairs_scored": sum(map(len, candidates)),
            "pairs_possible": int(np.sum(owners[:, None] != owners[None, :]))}


def tile_retrieval(capture_scores, tile_scores, owners, partners, noise_sigma):
    """Report capture ranks and partner hit fractions over all query tiles."""
    owners = np.asarray(owners)
    rows = []
    for a, b in partners:
        for reference, query in ((a, b), (b, a)):
            indices = np.flatnonzero(owners == query)
            hits = sum(np.isfinite(tile_scores[:, j]).any()
                       and owners[np.argmax(tile_scores[:, j])] == reference
                       for j in indices)
            rows.append({"query": query, "partner": reference,
                         "hits": int(hits), "tiles": len(indices),
                         "hit_fraction": float(hits / len(indices))})
    return {"retrieval": _retrieval(capture_scores, list(range(len(capture_scores))),
                                    partners, noise_sigma),
            "tile_retrieval": rows}


def _synthetic_tiles(rng, tile, overlap, min_mass):
    """Three-cell landmark world; two overlapping cubes and an unrelated world.

    Each capture is a cube of edge 2*tile. The second sees world cells 1 and 2
    and rotates its local coordinates by pi/2 about y, then translates by
    tile*[.04, -.03, .02]. Empty space is intentional. Landmark IDs provide
    exact tile correspondences independently of descriptors and scores.
    """
    worlds = []
    for _ in range(2):
        cells = [synthetic_scene(rng) for _ in range(3)]
        for cell in cells:
            cell.amp[:] = 1
            cell.mu[:] = rng.uniform(0.08, 0.92, cell.mu.shape)
        worlds.append(SplatScene(
            np.concatenate([s.mu + np.array([i, 0, 0]) for i, s in enumerate(cells)]),
            np.concatenate([s.cov for s in cells]),
            np.concatenate([s.amp for s in cells])))
    captures, memberships = [], []
    shift = np.array([0.04, -0.03, 0.02], np.float32)
    for owner, (world, start) in enumerate(
            ((worlds[0], 0), (worlds[0], 1), (worlds[1], 0))):
        keep = (world.mu[:, 0] >= start) & (world.mu[:, 0] <= start + 2)
        ids = np.flatnonzero(keep)
        scene = SplatScene((world.mu[keep] - [start, 0, 0]) / 2,
                           world.cov[keep] / 4, world.amp[keep])
        if owner == 1:
            scene = yaw_scene(scene, np.pi / 2)
            scene.mu += shift / 2
        mu = scene.mu * 2
        total = float(scene.amp[:, 0].sum())
        tiles, members = [], []
        for lo in tile_lattice(np.zeros(3), 2 * tile, tile, overlap):
            mask = np.all((mu * tile >= lo) & (mu * tile <= lo + tile), axis=1)
            share = float(scene.amp[mask, 0].sum()) / total
            if mask.any() and share >= min_mass:
                tiles.append((lo, SplatScene(
                    (mu[mask] - lo / tile).astype(np.float32),
                    scene.cov[mask] * 4, scene.amp[mask]), share))
                members.append(tuple(ids[mask]))
        captures.append(tiles)
        memberships.append(members)
    truth = []
    base = len(captures[0])
    for i, ids in enumerate(memberships[0]):
        for j, other in enumerate(memberships[1]):
            if ids == other:
                truth.append({"reference_tile": i, "query_tile": base + j,
                              "offset_box": [-0.02, -0.03, 0.04],
                              "yaw_radians": float(3 * np.pi / 2)})
    return captures, {"capture_offset_units": (np.array([1, 0, 0]) * tile).tolist(),
                      "query_shift_units": (shift * tile).tolist(),
                      "query_yaw_radians": float(np.pi / 2), "pairs": truth}


def _validate_tiles(parser, args):
    sigma = args.tile / 40 if args.sigma_units is None else args.sigma_units
    if not np.isfinite([args.tile, sigma]).all() or min(args.tile, sigma) <= 0:
        parser.error("--tile and --sigma-units must be finite and positive")
    if not 0 <= args.overlap < 1 or not 0 <= args.min_mass <= 1:
        parser.error("--overlap must be in [0,1); --min-mass in [0,1]")
    if args.prefilter < 0:
        parser.error("--prefilter must be nonnegative")
    if args.frame is not None or args.lo is not None or args.null != "phase":
        parser.error("tiles use per-capture lattices and the phase null")


def _nullable(values):
    array = np.asarray(values, dtype=object)
    array[~np.isfinite(np.asarray(values))] = None
    return array.tolist()


def _tile_truth_report(truth, matrix):
    if truth is None:
        return None
    reports = []
    for pair in truth["pairs"]:
        i, j = pair["reference_tile"], pair["query_tile"]
        offset = matrix["tile_offsets"][i, j]
        reports.append({**pair,
                        "rank1_hit": bool(matrix["tile_scores"][:, j].argmax() == i),
                        "reverse_rank1_hit": bool(
                            matrix["tile_scores"][:, i].argmax() == j),
                        "score": float(matrix["tile_scores"][i, j]),
                        "recovered_offset_box": _nullable(offset),
                        "offset_error_box": (float(np.max(np.abs(
                            offset - pair["offset_box"])))
                            if np.isfinite(offset).all() else None)})
        if not np.isfinite(reports[-1]["score"]):
            reports[-1]["score"] = None
    return reports


def _run_tiles(args):
    started = perf_counter()
    rng = np.random.default_rng(args.seed)
    sigma_units = args.tile / 40 if args.sigma_units is None else args.sigma_units
    if args.paths:
        captures = [tile_scenes(p, args.tile, args.overlap, args.min_mass)
                    for p in args.paths]
        labels, truth = [Path(p).name for p in args.paths], None
    else:
        captures, truth = _synthetic_tiles(
            rng, args.tile, args.overlap, args.min_mass)
        labels = ["world/left", "world/right-yawed", "unrelated"]
    if len(captures) < 2 or any(not tiles for tiles in captures):
        raise ValueError("need at least two captures, each with retained tiles")
    owners = np.repeat(np.arange(len(captures)), [len(t) for t in captures])
    angles = np.arange(args.yaws) * (2 * np.pi / args.yaws)
    sigma_box = sigma_units / args.tile
    freqs = sample_frequencies(args.dim, 3, 1 / sigma_box, rng)
    fps = tile_fingerprints([t for ts in captures for t in ts],
                            freqs, sigma_box, angles, args.flatten)
    matrix = tile_matrix(fps, owners, freqs, translation_grid(args.grid, args.limit),
                         args.whiten, args.prefilter, rng, args.scrambles)
    partners = args.partner or ([(0, 1)] if truth is not None else [])
    retrieval = tile_retrieval(matrix["capture_scores"], matrix["tile_scores"],
                               owners, partners, matrix["capture_noise"]["sigma"])
    for row in retrieval["retrieval"]:
        row["above_null_sigma"] = (
            (row["positive"] - matrix["capture_noise"]["mean"])
            / matrix["capture_noise"]["sigma"]
            if matrix["capture_noise"]["sigma"] > 0 else None)
    for row in retrieval["retrieval"]:
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                row[key] = None
    result = {**matrix, **retrieval, "labels": labels, "owners": owners.tolist(),
              "tiles": [{"count": len(ts), "lo": [lo.tolist() for lo, _, _ in ts],
                         "mass_shares": [mass for _, _, mass in ts]}
                        for ts in captures],
              "settings": {"tile": args.tile, "sigma_units": sigma_units,
                           "sigma_box": sigma_box, "overlap": args.overlap,
                           "min_mass": args.min_mass, "prefilter": args.prefilter,
                           "dim": args.dim, "seed": args.seed, "grid": args.grid,
                           "limit": args.limit, "yaws": args.yaws,
                           "angles": angles.tolist(), "whiten": args.whiten,
                           "scrambles": args.scrambles, "numpy": args.numpy,
                           "synthetic": not bool(args.paths)},
              "ground_truth": truth,
              "ground_truth_retrieval": _tile_truth_report(truth, matrix),
              "control_caveat": "Radial yaw invariance is approximate with finite "
                               "samples; unscored/masked entries are null. "
                               "Offsets are box units after rotating the query."}
    for key in ("tile_scores", "capture_scores", "tile_offsets", "tile_yaws"):
        result[key] = _nullable(result[key])
    result["runtime_seconds"] = perf_counter() - started
    return result


def _display_tiles(result, figure):
    print("Capture matrix (columns query; null means unscored):")
    for row in result["capture_scores"]:
        print(" ".join("     -" if x is None else f"{x:6.3f}" for x in row))
    for row in result["tile_retrieval"]:
        print("Tile partner:", json.dumps(row))
    for row in result["retrieval"]:
        print("Capture partner:", json.dumps(row))
    print("Tile null:", json.dumps(result["noise"]))
    print("Capture null:", json.dumps(result["capture_noise"]))
    print(f"Pairs scored: {result['pairs_scored']}/{result['pairs_possible']}")
    print(f"Runtime: {result['runtime_seconds']:.3f} s")
    if figure is not None:
        _tile_figure(result, figure)


def _tile_figure(result, figure):
    hits = np.full((len(result["labels"]), len(result["labels"])), np.nan)
    for row in result["tile_retrieval"]:
        hits[row["partner"], row["query"]] = row["hit_fraction"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, values, title in zip(
            axes, (np.asarray(result["capture_scores"], dtype=float), hits),
            ("Capture correlation", "Partner tile rank-1 fraction")):
        im = ax.imshow(values, vmin=0, vmax=1)
        ax.set(title=title, xlabel="query capture", ylabel="reference capture")
        ax.set_xticks(range(len(result["labels"])), result["labels"], rotation=30,
                      ha="right")
        ax.set_yticks(range(len(result["labels"])), result["labels"])
        fig.colorbar(im, ax=ax)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=150)
    plt.close(fig)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--synthetic", type=int, default=3)
    parser.add_argument("--flatten", choices=("none", "voxel", "log"), default="none")
    parser.add_argument("--flatten-study", action="store_true")
    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--yaws", type=int, default=16)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--limit", type=float, default=0.25)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scrambles", type=int, default=16)
    parser.add_argument("--numpy", action="store_true")
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--lo", type=float, nargs=3)
    parser.add_argument("--whiten", type=float, default=0.0,
                        help="PHAT exponent: 0 raw cross-power (default), 1 "
                        "phase-only correlation")
    parser.add_argument("--frame", help="capture whose mass-centred cube frames "
                        "every path (a crop in its parent's frame)")
    parser.add_argument("--null", choices=("phase", "scramble"), default="phase",
                        help="noise model for calibration: phase-randomised "
                        "fingerprints (default) or position scrambling")
    parser.add_argument("--extent", type=float)
    parser.add_argument("--partner", type=int, nargs=2, action="append", default=[])
    parser.add_argument("--tile", type=float, help="tile edge in scene units")
    parser.add_argument("--sigma-units", type=float, help="blur; default tile/40")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--min-mass", type=float, default=0.01)
    parser.add_argument("--prefilter", type=int, default=20)
    return parser


def _validate(parser, args):
    if args.tile is not None:
        _validate_tiles(parser, args)
    if min(args.synthetic, args.yaws, args.dim, args.scrambles) < 1 or args.grid < 2:
        parser.error("counts must be positive and --grid must be at least 2")
    if not np.isfinite([args.sigma, args.limit]).all() or min(
            args.sigma, args.limit) <= 0:
        parser.error("--sigma and --limit must be finite and positive")
    if (args.lo is None) != (args.extent is None):
        parser.error("--lo and --extent must be supplied together")
    if not 0 <= args.whiten <= 1:
        parser.error("--whiten must lie in [0, 1]")
    if args.frame is not None and args.lo is not None:
        parser.error("--frame and --lo/--extent are alternatives")
    n = len(args.paths) if args.paths else (3 if args.tile else 4 * args.synthetic)
    if any(i == j or min(i, j) < 0 or max(i, j) >= n for i, j in args.partner):
        parser.error("--partner requires distinct valid descriptor indices")


def _display(result, figure):
    if "tiles" in result:
        _display_tiles(result, figure)
        return
    for title, key in (("Phase correlation", "scores"), ("Radial control", "radial")):
        print(title)
        for i, row in enumerate(result[key]):
            print(f"{i:2d} " + " ".join(f"{value:6.3f}" for value in row))
    print("Labels:", dict(enumerate(result["labels"])))
    print("Scrambled noise:", json.dumps(result["noise"]))
    for report in result["retrieval"]:
        print("Known partner:", json.dumps(report))
    if figure is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, key in zip(axes, ("scores", "radial")):
            im = ax.imshow(result[key], vmin=0, vmax=1)
            ax.set(title=key, xlabel="query", ylabel="reference")
            fig.colorbar(im, ax=ax)
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=150)
        plt.close(fig)


def _run(args):
    if args.tile is not None:
        return _run_tiles(args)
    rng = np.random.default_rng(args.seed)
    scenes, labels, groups = _inputs(args, rng)
    freqs = sample_frequencies(args.dim, 3, 1 / args.sigma, rng)
    grid = translation_grid(args.grid, args.limit)
    angles = np.arange(args.yaws) * (2 * np.pi / args.yaws)
    fps = np.stack([fingerprint(s, freqs, args.sigma, args.flatten) for s in scenes])
    yaw_fps = np.stack([_yaw_fingerprints(s, angles, freqs, args.sigma, args.flatten)
                        for s in scenes])
    scores, offsets = similarity_matrix(fps, freqs, grid, yaw_fps, args.whiten)
    power = np.stack([radial_power(fp, freqs) for fp in fps])
    power /= np.maximum(np.linalg.norm(power, axis=1, keepdims=True), 1e-30)
    noise = _calibrate(scenes, fps, yaw_fps, angles, freqs, grid, args, rng)
    return {"labels": labels, "scores": scores.tolist(), "offsets": offsets.tolist(),
            "radial": (power @ power.T).tolist(), "noise": noise,
            "retrieval": _retrieval(scores, groups, args.partner, noise["sigma"]),
            "settings": {"seed": args.seed, "dim": args.dim, "sigma": args.sigma,
                         "grid": args.grid, "limit": args.limit,
                         "yaws": args.yaws, "angles": angles.tolist(),
                         "lo": args.lo, "extent": args.extent,
                         "frame": args.frame, "null": args.null,
                         "whiten": args.whiten,
                         "numpy": args.numpy, "synthetic": not bool(args.paths)},
            "control_caveat": "Translation exact to rounding; finite-sample yaw "
                              "invariance approximate. Permuting equal splats "
                              "does not change a scene."}


def flatten_crop_control(dim=4096, seed=0):
    """Bright landmark crop in the existing faint-background synthetic parent.

    Unlike the support-dominant regression crop, this crop contains only the
    first twelve bright landmarks: it dominates alpha, but not occupied space.
    """
    rng = np.random.default_rng(seed)
    parent = synthetic_scene(rng)
    crop = SplatScene(parent.mu[:12], parent.cov[:12], parent.amp[:12])
    freqs = sample_frequencies(dim, 3, 40, rng)
    grid = translation_grid(9, 0.12)
    result = {}
    for mode in ("none", "voxel", "log"):
        score, offset = correlate(
            fingerprint(crop, freqs, 0.025, mode),
            fingerprint(parent, freqs, 0.025, mode), freqs, grid, whiten=1)
        result[mode] = {"score": score, "offset": offset.tolist()}
    return result


def flatten_study(args):
    """Existing place fixture plus independently arranged dense-core halos."""
    matrices = {}
    for mode in ("none", "voxel", "log"):
        settings = argparse.Namespace(**vars(args))
        settings.flatten = mode
        matrices[mode] = _run(settings)
    rng = np.random.default_rng(args.seed + 400)
    captures = []
    for _ in range(2):
        core = rng.normal(0.5, 0.008, (160, 3))
        halo = rng.uniform(0.1, 0.9, (160, 3))
        mu = np.concatenate([core, halo])
        cov = np.tile(np.eye(3) * 0.004**2, (len(mu), 1, 1))
        amp = np.concatenate([np.full((160, 1), 10.0), np.ones((160, 1))])
        captures.append(SplatScene(mu, cov, amp))
    freqs = sample_frequencies(args.dim, 3, 1 / args.sigma, rng)
    grid = translation_grid(args.grid, args.limit)
    proxy = {}
    for mode in matrices:
        codes = [fingerprint(s, freqs, args.sigma, mode) for s in captures]
        score, offset = correlate(*codes, freqs, grid, whiten=1)
        proxy[mode] = {"score": score, "offset": offset.tolist()}
    return {"settings": {"dim": args.dim, "seed": args.seed},
            "place": matrices, "wide_proxy": proxy,
            "bright_crop": flatten_crop_control(args.dim, args.seed)}


def _flatten_figure(result, path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, (mode, matrix) in zip(axes[:3], result["place"].items()):
        im = ax.imshow(matrix["scores"], vmin=0, vmax=1)
        ax.set(title=mode, xlabel="query", ylabel="reference")
    fig.colorbar(im, ax=list(axes[:3]), shrink=0.7)
    proxy = result["wide_proxy"]
    axes[3].bar(list(proxy), [row["score"] for row in proxy.values()])
    axes[3].set(title="Unrelated core + halo", ylabel="Whitened peak", ylim=(0, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Write reproducible scores and matched-search null calibration as JSON."""
    parser = _parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    if args.paths and args.lo is None and args.tile is None:
        print("Per-capture crop normalization: use --lo and --extent for a shared "
              "physical frame; these scores cannot establish metric alignment.")
    run = flatten_study if args.flatten_study else _run
    if args.flatten_study and (args.paths or args.tile):
        parser.error("--flatten-study requires the non-tile synthetic fixture")
    if args.numpy:
        with patch.object(accel, "active", return_value=False):
            result = run(args)
    else:
        result = run(args)
    if args.flatten != "none":
        result["settings"]["flatten"] = args.flatten
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.flatten_study:
        print(json.dumps(result))
        if args.figure:
            _flatten_figure(result, args.figure)
    else:
        _display(result, args.figure)
    return result


if __name__ == "__main__":
    main()
