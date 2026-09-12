"""Driver for opt-in analytic projection; see docs/projection.md.

Run a capture with ``python -m examples.run_projection_pipeline SCENE``.
Legacy positional keep and --sweep forms remain available; settings run
sequentially through the public API. --synthetic is a seeded CPU smoke test.
Use --dim 2048 for a CPU-scale fixture, or --spectrum for band spectra.
"""
import os
import sys
import time

import numpy as np

from holo import runlog
from holo.capture import (
    BANDS,
    DIM,
    band_codebooks,
    build_scene,
    decode_slice,
    encode_bands,
    exact_slice,
    mass_mode,
    slice_grid,
)
from holo.projection import (
    Referee,
    band_errors,
    build_gram,
    project_cells,
    rel_err,
)
from holo.spectral import SplatScene

#: What `--spectrum` reports. The eps grid spans the regime the
#: pipeline actually operates in: at d=8192 the shipped keep=0.25 cuts
#: at 2.56e-4 on `xfine` and 1.77e-5 on `fine`, so a grid of 1e-8 and
#: below — chosen from a d=2048 rehearsal — sits entirely on the loose
#: side of every setting ever run and answers nothing.
SPECTRUM_KEEPS = (0.25, 0.40, 0.55, 0.70)
SPECTRUM_EPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8)

#: Where --spectrum leaves the spectra themselves. Committed, because
#: the eigendecompositions behind them cost 11 minutes and every later
#: threshold question is a lookup against this file rather than a rerun.
SPECTRUM_NPZ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "out", "gram_spectrum_d%d.npz")


def spectrum_table(dim=DIM, bands=None):
    """Each band's Gram spectrum: magnitudes, descending, scaled to 1.

    Needs no capture — the Gram depends on (codebook, cell size, window
    width) alone, which is also why a sweep can share it.

    Uses `eigvalsh` rather than `eigen`: the eigenvectors cost a second
    d x d buffer and most of the runtime, and nothing here applies an
    operator. The eigenvalues agree to machine precision, far tighter
    than anything read off this.
    """
    books = band_codebooks(np.random.default_rng(42), bands, dim)
    for name, _cap, cell in (bands or BANDS):
        freqs, _rho, _weights = books[name]
        a = np.abs(np.linalg.eigvalsh(build_gram(freqs.astype(np.float64),
                                                 (cell / 2) / 2)))
        a = np.sort(a)[::-1]
        yield name, a / a[0]


def at_keep(spectrum, frac):
    """The relative eigenvalue a RANK fraction cuts at — the smallest
    one the truncated operator will divide by, and therefore the thing
    that actually sets the regularisation."""
    return float(spectrum[max(1, round(frac * len(spectrum))) - 1])


def at_eps(spectrum, eps):
    """The rank fraction a THRESHOLD implies."""
    return int((spectrum > eps).sum()) / len(spectrum)


def report_spectrum(dim=DIM, run=None, save=True):
    """Whether `keep` is the right knob, in two tables and one file.

    `keep` truncates by rank, which says nothing about how SMALL the
    smallest survivor is. Each band draws its frequencies at its own
    scale cap, so one rank fraction lands at a different eigenvalue in
    every band — and the band whose spectrum decays fastest is the one
    a shared `keep` regularises LEAST.
    """
    spectra = {}
    for name, spectrum in spectrum_table(dim):
        spectra[name] = spectrum
        if run is not None:
            run.result(**{name: {
                "eigenvalue_at_keep": {"%.2f" % k: float("%.3e"
                                                         % at_keep(spectrum, k))
                                       for k in SPECTRUM_KEEPS},
                "keep_at_eps": {"%.0e" % e: round(at_eps(spectrum, e), 3)
                                for e in SPECTRUM_EPS}}})
        print("  %-7s measured" % name, flush=True)

    _spectrum_tables(spectra, dim)
    if save:
        path = SPECTRUM_NPZ % dim
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **spectra)
        print("\n  spectra -> %s" % os.path.relpath(path, os.getcwd()))
    return spectra


def _spectrum_tables(spectra, dim):
    """The two readings of the same spectra, in both directions."""
    print("\nRelative eigenvalue at the cut  (d=%d)" % dim)
    print("  %-7s %s" % ("band", " ".join("%11s" % ("keep=%.2f" % k)
                                          for k in SPECTRUM_KEEPS)))
    for name, a in spectra.items():
        print("  %-7s %s" % (name, " ".join("%11.2e" % at_keep(a, k)
                                            for k in SPECTRUM_KEEPS)))

    print("\nRank fraction a threshold implies")
    print("  %-7s %s" % ("band", " ".join("%11s" % ("eps=%.0e" % e)
                                          for e in SPECTRUM_EPS)))
    for name, a in spectra.items():
        print("  %-7s %s" % (name, " ".join("%11.3f" % at_eps(a, e)
                                            for e in SPECTRUM_EPS)))


def report_band_error(name, got, base):
    """One band's line, with the change against forward where both exist."""
    cells = []
    for i, v in enumerate(got):
        if v is None:
            cells.append("%16s" % "no signal")
            continue
        b = None if base is None else base[i]
        cells.append("%16s" % ("%.4f (%+.1f%%)" % (v, 100 * (b - v) / b)
                               if b else "%.4f" % v))
    print("      %-8s %s" % (name, " ".join(cells)), flush=True)


def label_of(setting):
    kind, val = setting
    return "keep=%.2f" % val if kind == "keep" else "%s=%.0e" % (kind, val)


def estimate_gb(n_settings):
    """What this run will need, for the headroom guard.

    Deliberately a loose UPPER bound. The guard's failure direction is
    under-protecting — an estimate that is too low reads as a check
    while providing none — and report_peak prints the real peak against
    this on every run, so being wrong is visible and cheap.

    Measured on saguaro at d=8192: 5.28 GB for one setting, 5.00 GB for
    a three-setting sweep. The peak is the eigendecomposition (the Gram,
    its eigenvectors and LAPACK's workspace), which the setting count
    does not move, because it lands on the first band before any solved
    bundles have accumulated. Settings still earn a term: on a denser
    capture each one's bundles are larger (train holds 1.3 GB against
    saguaro's 0.65 GB) and eventually outgrow the eigendecomposition.
    """
    return 5.5 + 0.7 * (n_settings - 1)


def build_referee(scene, members, books, slices, fwd_bundles):
    """Both referees and forward encoding's score on each.

    The aggregate slice error is what every projection number in this
    repo is quoted in. The per-band one exists because the aggregate is
    dominated by whichever band holds the splats, and is therefore not
    an acceptance test on its own — see `band_errors`.

    Per-band ground truth is not four times the work: the bands
    PARTITION the splats, so four single-band referee passes touch the
    same splats one full pass does. The only repeated cost is
    `exact_slice`'s whole-scene covariance inverse, which is seconds.
    """
    truth = {n: exact_slice(pts, scene, members) for n, (pts, _) in slices}
    band_truth = {n: {b[0]: exact_slice(pts, scene, members, bands=[b])
                      for b in BANDS}
                  for n, (pts, _) in slices}

    def err(bundles):
        return [rel_err(decode_slice(pts, bundles, books)[:, 0],
                        truth[n][:, 0])
                for n, (pts, _) in slices]

    def band_err(bundles):
        return band_errors(bundles, books, slices, band_truth)

    return Referee(err, band_err, err(fwd_bundles), band_err(fwd_bundles))


def band_result(per_band):
    """Per-band errors as JSON for the run record."""
    return {k: [None if v is None else round(v, 4) for v in got]
            for k, got in per_band.items()}


def report_forward(path, referee, seconds):
    """The baseline every setting is measured against, per band too."""
    print("%s  |  forward encoding: %.4f / %.4f  (%.0fs)"
          % (path.split("/")[-1], referee.base[0], referee.base[1], seconds),
          flush=True)
    for name, got in referee.band_base.items():
        report_band_error(name, got, None)


def report_setting(lab, cells, ref, also_shrink, run):
    """One setting's slice error, per band as well as in aggregate."""
    a = ref.err(cells)
    print("  %-16s analytic (window s=h/2): %.4f / %.4f"
          "   top-down %+.1f%%  side %+.1f%%"
          % (lab, a[0], a[1], 100 * (ref.base[0] - a[0]) / ref.base[0],
             100 * (ref.base[1] - a[1]) / ref.base[1]))
    per = ref.band_err(cells)
    for name, got in per.items():
        report_band_error(name, got, ref.band_base.get(name))
    if run is not None:
        run.result(**{lab: {"top_down": round(a[0], 4),
                            "side": round(a[1], 4),
                            "vs_forward_pct": round(
                                100 * (ref.base[0] - a[0]) / ref.base[0], 1),
                            "bands": band_result(per)}})
    if not also_shrink:
        return
    # Does shrinkage add anything to an ALREADY-SOLVED bundle? Prediction
    # was "little or negative" — the solve is L2-optimal on the window and
    # already regularised. Measured +6.4% / +3.6%: the objective (windowed
    # L2 per cell) is not the evaluation (slice error with cross-cell
    # contributions).
    from holo.denoise import percentile_threshold, shrink
    for pct in (10, 25):
        sh = {b: {k: shrink(v, percentile_threshold(v, pct))
                  for k, v in band.items()} for b, band in cells.items()}
        e = ref.err(sh)
        del sh
        print("    + shrink p%-2d       : %.4f / %.4f  (%+.1f%% / %+.1f%%)"
              % (pct, e[0], e[1], 100 * (a[0] - e[0]) / a[0],
                 100 * (a[1] - e[1]) / a[1]))


def synthetic_scene():
    """Seeded sparse fixture, available without capture files."""
    rng = np.random.default_rng(42)
    mu = rng.uniform(0.35, 0.65, (12, 3)).astype(np.float32)
    scales = np.full(12, 0.012, np.float32)
    cov = np.broadcast_to(np.eye(3) * scales[0] ** 2, (12, 3, 3)).copy()
    return (SplatScene(mu, cov.astype(np.float32), np.ones((12, 1), np.float32)),
            scales, np.ones(3, np.float32))


def main(path, settings, also_shrink=False, run=None,
         allow_divergence=False, dim=DIM):
    t0 = time.time()
    scene, smax, box = (synthetic_scene() if path == "synthetic"
                        else build_scene(path, verbose=False))
    books = band_codebooks(np.random.default_rng(42), dim=dim)
    fwd_bundles, members = encode_bands(scene, smax, books, verbose=False)

    w = scene.amp[:, 0]
    slices = [("top-down", slice_grid((0, box[0]), (0, box[2]), "y",
                                      mass_mode(scene.mu[:, 1], w, box[1]))),
              ("side", slice_grid((0, box[2]), (0, box[1]), "x",
                                  mass_mode(scene.mu[:, 0], w, box[0])))]
    referee = build_referee(scene, members, books, slices, fwd_bundles)
    base = referee.base
    if run is not None:
        run.result(forward={"top_down": round(base[0], 4),
                            "side": round(base[1], 4),
                            "bands": band_result(referee.band_base)})
        run.stage("forward encode", time.time() - t0)
    report_forward(path, referee, time.time() - t0)
    del fwd_bundles
    for setting in settings:
        lab = label_of(setting)
        cells = project_cells(scene, members, books,
                              setting=setting,
                              allow_divergence=allow_divergence)
        report_setting(lab, cells, referee, also_shrink, run)
    print("  total %.0fs" % (time.time() - t0))


def parse_settings(argv):
    """Back-compatible: a positional keep_frac and --tikhonov still work.
    --sweep keep=a,b / --sweep eps=a,b / --sweep tikhonov=a,b add
    settings evaluated sequentially through project_cells. Each call builds
    its own Gram; use BandSolver directly to reuse a Gram across settings."""
    settings, rest = [], []
    i = 0
    while i < len(argv):
        if argv[i] == "--tikhonov":
            settings.append(("tikhonov", float(argv[i + 1])))
            i += 2
        elif argv[i] == "--sweep":
            kind, _, vals = argv[i + 1].partition("=")
            if kind not in ("keep", "eps", "tikhonov"):
                raise SystemExit(
                    "--sweep takes keep=..., eps=... or tikhonov=...")
            settings += [(kind, float(v)) for v in vals.split(",")]
            i += 2
        elif not argv[i].startswith("--"):
            rest.append(argv[i])
            i += 1
        else:
            i += 1
    if not settings:
        settings = ([("keep", float(rest[1]))] if len(rest) > 1
                    else [("eps", 1e-3)])
    return rest[0], settings


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--spectrum" in argv:
        # Measures the KNOB, not a scene: the Gram depends only on the
        # codebook, the cell size and the window width, so this loads no
        # capture and holds one d x d buffer at a time.
        with runlog.record("spectrum", need_gb=1.5,
                           force="--force-memory" in argv) as run:
            report_spectrum(run=run)
        sys.exit(0)
    dim = DIM
    if "--dim" in argv:
        index = argv.index("--dim")
        dim = int(argv[index + 1])
        del argv[index:index + 2]
    if "--synthetic" in argv:
        argv[argv.index("--synthetic")] = "synthetic"
    path, settings = parse_settings(argv)
    with runlog.record(path.split("/")[-1],
                       need_gb=estimate_gb(len(settings)) * (dim / DIM) ** 2,
                       force="--force-memory" in argv) as run:
        run.result(forward=None)          # replaced once the baseline lands
        main(path, settings, also_shrink="--shrink" in argv, run=run,
             allow_divergence="--allow-divergence" in argv, dim=dim)
