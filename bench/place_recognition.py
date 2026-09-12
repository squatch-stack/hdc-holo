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
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

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


def fingerprint(scene, freqs, sigma_rec):
    """Encode only alpha after mass-preserving blur at recognition resolution."""
    if not np.isfinite(sigma_rec) or sigma_rec <= 0:
        raise ValueError("sigma_rec must be finite and positive")
    alpha = SplatScene(scene.mu, scene.cov, scene.amp[:, :1])
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


def _yaw_fingerprints(scene, angles, freqs, sigma):
    return np.stack([fingerprint(yaw_scene(scene, t), freqs, sigma) for t in angles])


def _calibrate(scenes, fps, yaw_fps, angles, freqs, grid, args, rng):
    samples = []
    for m in range(args.scrambles):
        i = m % len(scenes)
        if args.null == "phase":
            candidates = phase_surrogate(yaw_fps[i], rng)
        else:
            candidates = _yaw_fingerprints(scramble(scenes[i], rng), angles,
                                           freqs, args.sigma)
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


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--synthetic", type=int, default=3)
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
    return parser


def _validate(parser, args):
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
    n = len(args.paths) if args.paths else 4 * args.synthetic
    if any(i == j or min(i, j) < 0 or max(i, j) >= n for i, j in args.partner):
        parser.error("--partner requires distinct valid descriptor indices")


def _display(result, figure):
    for title, key in (("Phase correlation", "scores"), ("Radial control", "radial")):
        print(title)
        for i, row in enumerate(result[key]):
            print(f"{i:2d} " + " ".join(f"{value:6.3f}" for value in row))
    print("Labels:", dict(enumerate(result["labels"])))
    print("Scrambled noise:", json.dumps(result["noise"]))
    for report in result["retrieval"]:
        print("Known partner:", json.dumps(report))
    if figure is not None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, key in zip(axes, ("scores", "radial")):
            im = ax.imshow(result[key], vmin=0, vmax=1)
            ax.set(title=key, xlabel="query", ylabel="reference")
            fig.colorbar(im, ax=ax)
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=150)
        plt.close(fig)


def _run(args):
    rng = np.random.default_rng(args.seed)
    scenes, labels, groups = _inputs(args, rng)
    freqs = sample_frequencies(args.dim, 3, 1 / args.sigma, rng)
    grid = translation_grid(args.grid, args.limit)
    angles = np.arange(args.yaws) * (2 * np.pi / args.yaws)
    fps = np.stack([fingerprint(s, freqs, args.sigma) for s in scenes])
    yaw_fps = np.stack([_yaw_fingerprints(s, angles, freqs, args.sigma)
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


def main(argv=None):
    """Write reproducible scores and matched-search null calibration as JSON."""
    parser = _parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    if args.paths and args.lo is None:
        print("Per-capture crop normalization: use --lo and --extent for a shared "
              "physical frame; these scores cannot establish metric alignment.")
    if args.numpy:
        with patch.object(accel, "active", return_value=False):
            result = _run(args)
    else:
        result = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    _display(result, args.figure)
    return result


if __name__ == "__main__":
    main()
