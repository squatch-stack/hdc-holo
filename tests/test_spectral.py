"""holo/spectral.py — spectral encoder, mixture codebooks, translation."""

import numpy as np
import pytest

from holo.spectral import (
    SplatScene,
    decode_field,
    decode_field_phasor,
    eval_scene_exact,
    phasor_bundle,
    random_scene,
    sample_frequencies,
    spectral_bundle,
    translate_bundle,
)


def _single_splat(sigma=0.03):
    return SplatScene(
        mu=np.array([[0.5, 0.5]], dtype=np.float32),
        cov=np.array([np.eye(2) * sigma**2], dtype=np.float32),
        amp=np.array([[1.0]], dtype=np.float32))


def test_spectral_decode_matches_analytic_gaussian():
    rng = np.random.default_rng(0)
    scene = _single_splat()
    rho = 1.3 / 0.03
    freqs = sample_frequencies(1 << 14, 2, rho, rng)
    bundle = spectral_bundle(scene, freqs)
    pts = rng.uniform(0.35, 0.65, size=(400, 2)).astype(np.float32)
    approx = decode_field(bundle, freqs, rho, pts)
    exact = eval_scene_exact(scene, pts)
    # MC error ~ chi/sqrt(d) ~ 0.01 at d=16384; 0.06 is >4 sigma of margin
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 0.06


def test_phasor_bundle_is_the_shared_kernel_special_case():
    rng = np.random.default_rng(1)
    scene = _single_splat()
    freqs = sample_frequencies(1 << 14, 2, 1.0 / 0.03, rng)
    bundle = phasor_bundle(scene, freqs)
    pts = rng.uniform(0.35, 0.65, size=(400, 2)).astype(np.float32)
    approx = decode_field_phasor(bundle, freqs, pts)
    exact = eval_scene_exact(scene, pts)
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 0.06


def test_mixture_codebook_decodes_unbiased():
    rng = np.random.default_rng(2)
    scene = _single_splat()
    rho = [0.4 / 0.03, 1.3 / 0.03, 4.0 / 0.03]   # 3-component mixture
    freqs = sample_frequencies(1 << 14, 2, rho, rng)
    bundle = spectral_bundle(scene, freqs)
    pts = rng.uniform(0.35, 0.65, size=(400, 2)).astype(np.float32)
    approx = decode_field(bundle, freqs, rho, pts)
    exact = eval_scene_exact(scene, pts)
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 0.08


def test_translate_bundle_shifts_the_decoded_field():
    rng = np.random.default_rng(3)
    scene = random_scene(20, dim=2, rng=rng, scale_range=(0.03, 0.05))
    rho = 1.3 / 0.03
    freqs = sample_frequencies(1 << 14, 2, rho, rng)
    bundle = spectral_bundle(scene, freqs)
    t = np.array([0.12, -0.07], dtype=np.float32)
    shifted = translate_bundle(bundle, freqs, t)
    pts = rng.uniform(0.25, 0.75, size=(400, 2)).astype(np.float32)
    # decoding the bound-translated bundle at p equals the original
    # mixture at p - t: the shift theorem is exact on the bundle
    approx = decode_field(shifted, freqs, rho, pts)
    exact = eval_scene_exact(scene, pts - t)
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 0.15   # same MC noise budget as the untranslated decode


def test_error_shrinks_with_dimension():
    rng = np.random.default_rng(4)
    scene = random_scene(30, dim=2, rng=rng, scale_range=(0.03, 0.05))
    rho = 1.3 / 0.03
    rels = []
    for d in (1 << 10, 1 << 14):
        book_rng = np.random.default_rng(7)
        freqs = sample_frequencies(d, 2, rho, book_rng)
        bundle = spectral_bundle(scene, freqs)
        pts = np.random.default_rng(8).uniform(
            0.2, 0.8, size=(400, 2)).astype(np.float32)
        exact = eval_scene_exact(scene, pts)
        approx = decode_field(bundle, freqs, rho, pts)
        rels.append(np.linalg.norm(approx - exact) / np.linalg.norm(exact))
    assert rels[1] < rels[0]           # ~1/sqrt(d)
    assert rels[1] < 0.15


def test_gpu_and_numpy_paths_agree():
    from holo import accel
    if not accel.active():
        pytest.skip("MLX backend not active")
    rng = np.random.default_rng(5)
    scene = random_scene(10, dim=3, rng=rng, scale_range=(0.03, 0.05))
    freqs = sample_frequencies(1 << 12, 3, 1.3 / 0.03, rng)
    fast = accel.spectral_bundle(scene, freqs)
    # reference: the pure-NumPy formulation, computed inline
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    wq = np.stack([freqs[:, i] * freqs[:, j] * (1.0 if i == j else 2.0)
                   for i, j in pairs], axis=1)
    cq = np.stack([scene.cov[:, i, j] for i, j in pairs], axis=1)
    norm = ((2 * np.pi) ** 1.5
            * np.sqrt(np.linalg.det(scene.cov.astype(np.float64))))
    env = np.exp(-0.5 * (cq @ wq.T))
    ref = (scene.amp * norm[:, None]).T.astype(np.complex64) @ \
        (env * np.exp(-1j * (scene.mu @ freqs.T))).astype(np.complex64)
    # identical math, different silicon: float32 rounding only
    assert np.max(np.abs(fast - ref)) / np.max(np.abs(ref)) < 1e-5


# --- orthogonal frequency coupling (issue #3) ------------------------------
#
# The variance win is free, but it rests entirely on the coupled rows
# keeping their Gaussian marginal: decode_weights evaluates rho at each
# drawn frequency, so a drifted marginal makes every decode subtly wrong
# WITHOUT raising anything. These tests exist to make that failure loud.

def _ks_vs_standard_normal(x):
    """Two-sided KS statistic against N(0,1), and the 5% critical value.
    Hand-rolled because scipy is not a dependency here."""
    import math
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    d_plus = np.max(np.arange(1, n + 1) / n - cdf)
    d_minus = np.max(cdf - np.arange(0, n) / n)
    return max(d_plus, d_minus), 1.36 / math.sqrt(n)


def test_orthogonal_coupling_preserves_the_gaussian_marginal():
    from holo.spectral import sample_frequencies
    sigma, dim = 2.5, 3
    w = sample_frequencies(30000, dim, sigma, np.random.default_rng(0),
                           coupling="orthogonal")
    for axis in range(dim):
        stat, crit = _ks_vs_standard_normal(w[:, axis] / sigma)
        assert stat < crit, ("axis %d marginal drifted: KS %.5f >= %.5f"
                             % (axis, stat, crit))


def test_the_haar_sign_correction_is_what_preserves_it():
    """The correction is one line and its absence is silent, so pin it by
    showing the UNCORRECTED construction fails the test above. Without
    this, a well-meaning simplification of _haar_orthogonal passes
    everything and quietly biases every codebook."""
    rng = np.random.default_rng(0)
    sigma, dim, n = 2.5, 3, 30000
    blocks = []
    for _ in range(-(-n // dim)):
        radii = np.sqrt(rng.chisquare(dim, size=dim))
        q, _r = np.linalg.qr(rng.standard_normal((dim, dim)))   # no sign fix
        blocks.append((radii[:, None] * q) * sigma)
    uncorrected = np.vstack(blocks)[:n]
    worst = max(_ks_vs_standard_normal(uncorrected[:, a] / sigma)[0]
                for a in range(dim))
    _, crit = _ks_vs_standard_normal(uncorrected[:, 0] / sigma)
    assert worst > crit, ("plain QR should FAIL the marginal test; if this "
                          "assertion breaks, the marginal test has stopped "
                          "discriminating and is no longer protecting anything")


def test_orthogonal_coupling_actually_couples():
    from holo.spectral import sample_frequencies
    dim = 3
    w = sample_frequencies(300, dim, 1.0, np.random.default_rng(1),
                           coupling="orthogonal").astype(np.float64)
    u = w / np.linalg.norm(w, axis=1, keepdims=True)
    # rows within a block are mutually orthogonal; across blocks they are not
    within = [abs(float(u[i] @ u[j]))
              for b in range(0, 300, dim)
              for i in range(b, b + dim) for j in range(i + 1, b + dim)]
    across = abs(u[0] @ u[dim:].T)
    assert max(within) < 1e-5, "within-block rows are not orthogonal"
    assert across.mean() > 0.1, "across-block rows should stay independent"


def test_decode_weights_needs_no_change_under_coupling():
    """Coupling changes the joint distribution, not the marginal, so the
    importance weights keep the same distribution. Compared by quantile:
    decode_weights is 1/density, whose MEAN is tail-dominated (mean/median
    ~1e3) and swings by 2x between draws of either kind."""
    from holo.spectral import decode_weights, sample_frequencies
    sig = [10.0, 40.0, 160.0, 640.0]
    a, b = [], []
    for s in range(4):
        a.append(decode_weights(sample_frequencies(
            20000, 3, sig, np.random.default_rng(100 + s)), sig))
        b.append(decode_weights(sample_frequencies(
            20000, 3, sig, np.random.default_rng(200 + s),
            coupling="orthogonal"), sig))
    a, b = np.concatenate(a), np.concatenate(b)
    for q in (25, 50, 75, 90, 99):
        ratio = np.percentile(b, q) / np.percentile(a, q)
        assert 0.85 < ratio < 1.15, ("p%d ratio %.3f — the coupled draw's "
                                     "importance weights have shifted" % (q, ratio))


def test_iid_remains_the_default():
    """Every committed measurement was taken under iid; changing the
    default silently would renumber the whole repo."""
    from holo.spectral import sample_frequencies
    explicit = sample_frequencies(64, 3, 1.0, np.random.default_rng(5),
                                  coupling="iid")
    default = sample_frequencies(64, 3, 1.0, np.random.default_rng(5))
    assert np.array_equal(explicit, default)


def test_unknown_coupling_is_refused():
    from holo.spectral import sample_frequencies
    with pytest.raises(ValueError, match=r"iid.*orthogonal"):
        sample_frequencies(16, 3, 1.0, np.random.default_rng(0),
                           coupling="antithetic")


# --- per-cell Gram factorisation (issue #2) --------------------------------

def test_cell_gram_factorises_so_one_decomposition_serves_a_band():
    """`G_c = D G0 D^H` with D a unitary diagonal of cell-centre phases.

    The analytic projection's whole cost model rests on this: cell
    position enters only as a phase, so ONE eigendecomposition serves
    every cell in a band instead of one per cell — 1,624 of them on the
    saguaro fine band. If cell geometry ever stops being a translation
    of a common box, this should fail loudly rather than silently making
    the projection quadratically more expensive.
    """
    rng = np.random.default_rng(0)
    freqs = rng.normal(0, 8.0, size=(48, 3))
    half = 0.0625

    def gram(centre):
        delta = freqs[:, None, :] - freqs[None, :, :]
        phase = np.exp(1j * (delta @ np.asarray(centre, dtype=float)))
        sinc = np.prod(2 * half * np.sinc(delta * half / np.pi), axis=2)
        return phase * sinc

    origin = gram([0.0, 0.0, 0.0])
    for centre in ([0.3, -0.7, 1.1], [5.0, 2.0, -3.0], list(rng.normal(size=3))):
        d = np.diag(np.exp(1j * (freqs @ np.asarray(centre, dtype=float))))
        assert np.allclose(gram(centre), d @ origin @ d.conj().T, atol=1e-10)
        # the consequence that matters: identical spectra, so a single
        # decomposition is reusable across cells
        assert np.allclose(np.linalg.svd(origin, compute_uv=False),
                           np.linalg.svd(gram(centre), compute_uv=False),
                           rtol=1e-9)


def test_a_gaussian_window_conditions_better_than_a_hard_box():
    """Why the windowed objective is the usable one. The Gram IS the
    window's Fourier transform, so a box gives slowly-decaying sincs and
    a Gaussian gives Gaussian decay.

    The threshold here is deliberately modest because the test runs at
    d=512, where the measured ratio is ~460x and BOTH Grams are still
    full rank. The gap widens sharply with dimension — at d=2048 it is
    6e19 against 1.2e11, and the box has lost 400 modes — but asserting
    that here would mean carrying a 2048x2048 eigendecomposition in the
    unit suite for a fact the driver already reports."""
    rng = np.random.default_rng(1)
    sigmas = 1.0 / np.geomspace(0.002, 0.004, 5)      # real xfine codebook
    comp = rng.integers(0, len(sigmas), size=512)
    freqs = sigmas[comp, None] * rng.standard_normal((512, 3))
    half = (1 / 32) / 2
    delta = freqs[:, None, :] - freqs[None, :, :]
    box = np.prod(2 * half * np.sinc(delta * half / np.pi), axis=2)
    win = np.exp(-0.5 * (half / 2) ** 2 * (delta ** 2).sum(2))
    assert np.linalg.cond(win) < np.linalg.cond(box) / 100
