"""Gaussian splats as complex64 hypervector bundles.

FHRR / Vector Function Architecture.

The bridge is Bochner's theorem / random Fourier features:

    a Gaussian splat  g(p) = a * exp(-1/2 (p-mu)^T Sigma^{-1} (p-mu))
    has Fourier transform
    g_hat(w) = a * (2*pi)^{D/2} |Sigma|^{1/2} * exp(-1/2 w^T Sigma w) * e^{-i w.mu}

Sample d random frequencies w_j ~ rho = N(0, sigma_rho^2 I) once (the shared
codebook). A splat's hypervector is its spectrum sampled at those frequencies:

    s_k[j] = g_hat_k(w_j)                      (complex64, length d)

A scene is the *bundle* (superposition) of its splats:  S = sum_k s_k.
Decoding the field at a point p is Monte-Carlo inversion of the Fourier
transform with importance weights 1/rho:

    f(p) ~= Re[ (1/d) sum_j S[j] e^{i w_j.p} / ((2*pi)^D rho(w_j)) ]

which is a single inner product between the bundle and a query hypervector.

The classical FHRR / fractional-power-encoding case falls out as the special
case where every splat shares one covariance Sigma0: draw w_j ~ N(0, Sigma0^{-1})
and the magnitude envelope is constant, so splat vectors are *unit phasors*
e^{-i w_j.mu} and decoding needs no importance weights.

Translation of the whole scene is binding: shifting f by t multiplies the
spectrum by e^{-i w.t}, i.e. one elementwise complex multiply on the bundle.
"""

from dataclasses import dataclass

import numpy as np

from . import accel as _accel  # Metal GPU backend when available

# ---------------------------------------------------------------------------
# Scene container
# ---------------------------------------------------------------------------

@dataclass
class SplatScene:
    """A set of anisotropic Gaussian splats with per-channel amplitudes."""

    mu: np.ndarray    # (N, D) float32 centers
    cov: np.ndarray   # (N, D, D) float32 covariances
    amp: np.ndarray   # (N, C) float32 per-channel peak amplitudes

    @property
    def n(self):
        return self.mu.shape[0]

    @property
    def dim(self):
        return self.mu.shape[1]

    @property
    def channels(self):
        return self.amp.shape[1]


def rotation_matrix_2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def random_rotations_3d(n, rng):
    """Uniform-ish random 3D rotations via QR of Gaussian matrices."""
    a = rng.standard_normal((n, 3, 3))
    rots = np.empty((n, 3, 3), dtype=np.float32)
    for i in range(n):
        q, r = np.linalg.qr(a[i])
        q *= np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        rots[i] = q
    return rots


def random_scene(n, dim, rng, scale_range=(0.02, 0.045), channels=1,
                 amp_range=(0.3, 1.0), box=1.0):
    """Random anisotropic splats in a box of side `box`."""
    mu = rng.uniform(0.08, box - 0.08, size=(n, dim)).astype(np.float32)
    scales = rng.uniform(*scale_range, size=(n, dim)).astype(np.float32)
    if dim == 2:
        thetas = rng.uniform(0.0, np.pi, size=n)
        rots = np.stack([rotation_matrix_2d(t) for t in thetas])
    elif dim == 3:
        rots = random_rotations_3d(n, rng)
    else:
        raise ValueError("dim must be 2 or 3")
    # Sigma = R diag(s^2) R^T
    cov = np.einsum("nij,nj,nkj->nik", rots, scales**2, rots).astype(np.float32)
    amp = rng.uniform(*amp_range, size=(n, channels)).astype(np.float32)
    return SplatScene(mu=mu, cov=cov, amp=amp)


# ---------------------------------------------------------------------------
# Frequency codebook
# ---------------------------------------------------------------------------

def sample_frequencies(d, dim, sigma_rho, rng):
    """d random frequencies from rho = N(0, sigma_rho^2 I). Shape (d, dim).

    sigma_rho may also be a sequence of stds: then rho is an equal-weight
    mixture of those Gaussians -- a multi-scale codebook whose importance
    weights stay bounded across a spread of splat scales, instead of one
    sigma_rho that must cover the narrowest splat and pays a large variance
    penalty on the widest.
    """
    sigmas = np.atleast_1d(np.asarray(sigma_rho, dtype=np.float64))
    if sigmas.size == 1:
        return (float(sigmas[0]) * rng.standard_normal((d, dim))).astype(np.float32)
    comp = rng.integers(0, sigmas.size, size=d)
    return (sigmas[comp, None] * rng.standard_normal((d, dim))).astype(np.float32)


def decode_weights(freqs, sigma_rho):
    """Importance weights c_j = 1 / (d * (2 pi)^D * rho(w_j)) for decoding.

    sigma_rho: the same scalar or sequence the frequencies were sampled
    with; a sequence means rho is the equal-weight Gaussian mixture.
    """
    d, dim = freqs.shape
    sigmas = np.atleast_1d(np.asarray(sigma_rho, dtype=np.float64))
    r2 = (freqs.astype(np.float64) ** 2).sum(axis=1)
    comp_log = np.stack([-0.5 * r2 / s**2 - 0.5 * dim * np.log(2 * np.pi * s**2)
                         for s in sigmas])
    peak = comp_log.max(axis=0)
    log_rho = peak + np.log(np.exp(comp_log - peak).mean(axis=0))
    return np.exp(-log_rho - dim * np.log(2 * np.pi) - np.log(d)).astype(np.float32)


# ---------------------------------------------------------------------------
# Encoding: splats -> complex64 bundle
# ---------------------------------------------------------------------------

def spectral_bundle(scene, freqs, chunk=512):
    """Bundle anisotropic splats: S[c, j] = sum_k amp[k,c] * g_hat_k(w_j).

    Returns (C, d) complex64. Handles per-splat covariance because the
    covariance lives in the sampled spectrum's magnitude envelope, not in the
    frequency codebook. The quadratic form w^T Sigma w is evaluated as one
    matmul over the dim*(dim+1)/2 unique covariance entries (a BLAS sgemm),
    which is what makes million-splat scenes tractable.
    """
    if _accel is not None and _accel.active():
        return _accel.spectral_bundle(scene, freqs)
    d, dim = freqs.shape
    pairs = [(i, j) for i in range(dim) for j in range(i, dim)]
    wq = np.stack([freqs[:, i] * freqs[:, j] * (1.0 if i == j else 2.0)
                   for i, j in pairs], axis=1)       # (d, dim*(dim+1)/2)
    bundle = np.zeros((scene.channels, d), dtype=np.complex64)
    for lo in range(0, scene.n, chunk):
        mu = scene.mu[lo:lo + chunk]
        cov = scene.cov[lo:lo + chunk]
        amp = scene.amp[lo:lo + chunk]
        # (2 pi)^{D/2} |Sigma|^{1/2}, folded into the amplitude per splat
        norm = ((2 * np.pi) ** (dim / 2)
                * np.sqrt(np.linalg.det(cov.astype(np.float64)))).astype(np.float32)
        cq = np.stack([cov[:, i, j] for i, j in pairs], axis=1)
        env = np.exp(-0.5 * (cq @ wq.T))              # == w^T Sigma w, (n, d)
        phase = mu @ freqs.T                          # (n, d)
        splat_vecs = (norm[:, None] * env) * np.exp(-1j * phase).astype(np.complex64)
        bundle += (amp.T.astype(np.complex64)) @ splat_vecs
    return bundle


def phasor_bundle(scene, freqs, chunk=4096):
    """FHRR/FPE bundle for the shared-covariance case: unit phasors only.

    freqs must be drawn from N(0, Sigma0^{-1}) for the shared kernel Sigma0
    (for isotropic Sigma0 = sigma0^2 I that means sigma_rho = 1/sigma0).
    S[c, j] = sum_k amp[k,c] * e^{-i w_j . mu_k}
    """
    d, _ = freqs.shape
    bundle = np.zeros((scene.channels, d), dtype=np.complex64)
    for lo in range(0, scene.n, chunk):
        mu = scene.mu[lo:lo + chunk]
        amp = scene.amp[lo:lo + chunk]
        phase = mu @ freqs.T
        bundle += ((amp.T.astype(np.complex64))
                   @ np.exp(-1j * phase).astype(np.complex64))
    return bundle


# ---------------------------------------------------------------------------
# Decoding: bundle x query point -> field value
# ---------------------------------------------------------------------------

def decode_field(bundle, freqs, sigma_rho, points, chunk=1024):
    """Evaluate the encoded field at query points via inner products.

    Returns (P, C) float32: f_c(p) ~= Re[ sum_j S[c,j] c_j e^{i w_j . p} ].
    """
    weights = decode_weights(freqs, sigma_rho)
    return _decode(bundle, freqs, weights, points, chunk)


def decode_field_phasor(bundle, freqs, points, chunk=1024):
    """Decode an FHRR/phasor bundle: uniform weights 1/d."""
    d = freqs.shape[0]
    weights = np.full(d, 1.0 / d, dtype=np.float32)
    return _decode(bundle, freqs, weights, points, chunk)


def _decode(bundle, freqs, weights, points, chunk):
    if _accel is not None and _accel.active():
        return _accel.decode(bundle, freqs, weights, points)
    out = np.empty((points.shape[0], bundle.shape[0]), dtype=np.float32)
    weighted = (bundle * weights[None, :]).T          # (d, C) complex64
    for lo in range(0, points.shape[0], chunk):
        phase = points[lo:lo + chunk] @ freqs.T       # (p, d) float32
        query = np.exp(1j * phase).astype(np.complex64)
        out[lo:lo + chunk] = (query @ weighted).real
    return out


# ---------------------------------------------------------------------------
# Bundle algebra
# ---------------------------------------------------------------------------

def translate_bundle(bundle, freqs, t):
    """Shift the entire encoded scene by t: one elementwise complex multiply.

    f'(p) = f(p - t)  <=>  S'[j] = S[j] * e^{-i w_j . t}   (binding with the
    conjugate position encoding of t).
    """
    shift = np.exp(-1j * (freqs @ np.asarray(t, dtype=np.float32))).astype(np.complex64)
    return bundle * shift[None, :]


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def eval_scene_exact(scene, points, iso_sigma=None, chunk=256):
    """Evaluate the Gaussian mixture directly. Returns (P, C) float32.

    If iso_sigma is given, every splat uses the isotropic kernel sigma^2 I
    instead of its own covariance (the phasor/FPE ground truth).
    """
    out = np.zeros((points.shape[0], scene.channels), dtype=np.float32)
    for lo in range(0, scene.n, chunk):
        mu = scene.mu[lo:lo + chunk]
        amp = scene.amp[lo:lo + chunk]
        diff = points[None, :, :] - mu[:, None, :]    # (n, P, D)
        if iso_sigma is None:
            cov64 = scene.cov[lo:lo + chunk].astype(np.float64)
            inv_cov = np.linalg.inv(cov64).astype(np.float32)
            quad = np.einsum("npi,nij,npj->np", diff, inv_cov, diff)
        else:
            quad = (diff**2).sum(axis=2) / iso_sigma**2
        out += np.exp(-0.5 * quad).T @ amp
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Single 2D splat, large d: decoded field must match the analytic Gaussian."""
    rng = np.random.default_rng(0)
    sigma = 0.03
    scene = SplatScene(
        mu=np.array([[0.5, 0.5]], dtype=np.float32),
        cov=np.array([np.eye(2) * sigma**2], dtype=np.float32),
        amp=np.array([[1.0]], dtype=np.float32),
    )
    sigma_rho = 1.3 / sigma
    freqs = sample_frequencies(1 << 17, 2, sigma_rho, rng)
    bundle = spectral_bundle(scene, freqs)
    pts = rng.uniform(0.35, 0.65, size=(500, 2)).astype(np.float32)
    approx = decode_field(bundle, freqs, sigma_rho, pts)
    exact = eval_scene_exact(scene, pts)
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 0.02, f"spectral self-test failed: rel err {rel:.4f}"

    freqs_p = sample_frequencies(1 << 17, 2, 1.0 / sigma, rng)
    bundle_p = phasor_bundle(scene, freqs_p)
    approx_p = decode_field_phasor(bundle_p, freqs_p, pts)
    rel_p = np.linalg.norm(approx_p - exact) / np.linalg.norm(exact)
    assert rel_p < 0.02, f"phasor self-test failed: rel err {rel_p:.4f}"

    mix = [0.4 / sigma, 1.3 / sigma, 4.0 / sigma]
    freqs_m = sample_frequencies(1 << 17, 2, mix, rng)
    bundle_m = spectral_bundle(scene, freqs_m)
    approx_m = decode_field(bundle_m, freqs_m, mix, pts)
    rel_m = np.linalg.norm(approx_m - exact) / np.linalg.norm(exact)
    assert rel_m < 0.02, f"mixture self-test failed: rel err {rel_m:.4f}"
    print(f"self-test OK  (spectral rel err {rel:.4f}, phasor {rel_p:.4f}, "
          f"mixture-codebook {rel_m:.4f})")


if __name__ == "__main__":
    _self_test()
