"""Fitting holograms FROM data: the bundle is a regression weight vector.

Everything else in this package builds bundles forward — we know the
splats, we superpose their codewords. But look at the readout:

    f(p) = Re< e^{i W p}, conj(S) > / D

f is LINEAR in S. A hologram is the weight vector of a random Fourier
features regression model (Rahimi & Recht), and the codeword map
p -> e^{i W p} is the feature map. So S can be FIT to samples
{(p_i, y_i)} of any target field by ridge regression — no splats, no
mixture model, no knowledge of how the data was generated. This is the
holographic analog of 3DGS's training loop: where 3DGS optimizes
explicit splat parameters through a differentiable rasterizer, here the
representation itself is the parameter vector and the problem is convex
with a closed-form optimum.

In real coordinates the features are [cos(Wp), sin(Wp)] (2D of them,
plus a bias) and we solve

    (A^T A + lambda I) theta = A^T y

by conjugate gradients, then fold theta back into a complex bundle:
S = D * (theta_cos + i theta_sin). The fitted S is a first-class citizen:
render it, chunk it, replicate it over Loro — nothing downstream knows
it was learned rather than bundled.

Bandwidth is the one modeling decision: a single Sigma is a single
kernel scale, and photographs are broadband. FrequencyBands concatenates
several W blocks with different sigmas (coarse-to-fine, like spatial.py's
bands but fit jointly) so one hologram carries all scales.

Because regression AVERAGES rather than memorizes, fitting from noisy
samples denoises: with n samples per effective degree of freedom the
recovered field beats the per-sample noise floor — something forward
bundling cannot do, since it never sees data at all.
"""

import numpy as np


class FrequencyBands:
    """Concatenated frequency blocks, one per kernel scale."""

    def __init__(self, dims, sigmas, ndim=2, seed=0):
        assert len(dims) == len(sigmas)
        rng = np.random.default_rng(seed)
        blocks = []
        for d_b, sigma in zip(dims, sigmas):
            sigma = np.atleast_2d(np.asarray(sigma, dtype=np.float64))
            if sigma.shape == (1, 1):
                sigma = np.eye(ndim) * sigma.item() ** 2   # scalar = iso std
            B = np.linalg.cholesky(np.linalg.inv(sigma))
            blocks.append((rng.standard_normal((d_b, sigma.shape[0]))
                           @ B.T).astype(np.float32))
        self.W = np.concatenate(blocks, axis=0)
        self.D = self.W.shape[0]
        self.dims, self.sigmas = list(dims), list(sigmas)


class HoloRegressor:
    """Ridge-fit a holographic field to point samples of any target."""

    def __init__(self, bands):
        self.bands = bands
        self.W = bands.W
        self.D = bands.D
        self.S = np.zeros(self.D, dtype=np.complex64)
        self.bias = 0.0

    def _design(self, points):
        theta = np.asarray(points, dtype=np.float32) @ self.W.T
        A = np.empty((len(points), 2 * self.D + 1), dtype=np.float32)
        np.cos(theta, out=A[:, :self.D])
        np.sin(theta, out=A[:, self.D:2 * self.D])
        A[:, -1] = 1.0
        return A

    def fit(self, points, values, lam=1e-3):
        """Exact ridge regression. lam is relative (actual ridge lam * n).

        values: (n,) for a scalar field or (n, c) for c channels (e.g.
        RGB). All channels share ONE design matrix and ONE factorization
        — color costs extra right-hand sides, not extra solves.

        Solves the primal normal equations (A^T A + ridge I) theta = A^T y
        when features <= samples, else the equivalent dual/kernel form
        theta = A^T (A A^T + ridge I)^{-1} y — the same optimum, whichever
        Gram matrix is smaller. Direct float64 solves: the convex problem
        deserves its closed form (CG in float32 diverges at this scale)."""
        y = np.asarray(values, dtype=np.float32)
        single = y.ndim == 1
        Y = y[:, None] if single else y                       # (n, c)
        A = self._design(points)
        n, m = A.shape
        ridge = lam * n
        # NOTE: numpy is pinned to 1.26.4 (OpenBLAS wheels) because the
        # Accelerate-backed numpy 2.0 wheels on macOS corrupt float32
        # GEMV (A.T @ y and y @ A alike) with heap-layout-dependent NaNs
        # at this size. If you lift the pin, run the test suite several
        # times in a row before trusting a fit.
        if m <= n:
            G = (A.T @ A).astype(np.float64)
            G[np.diag_indices_from(G)] += ridge
            Theta = np.linalg.solve(G, (Y.T @ A).astype(np.float64).T)
        else:
            K = (A @ A.T).astype(np.float64)
            K[np.diag_indices_from(K)] += ridge
            Beta = np.linalg.solve(K, Y.astype(np.float64))
            Theta = (Beta.astype(np.float32).T @ A).astype(np.float64).T
        S = self.D * (Theta[:self.D] + 1j * Theta[self.D:2 * self.D])
        if single:
            self.S = S[:, 0].astype(np.complex64)             # (D,)
            self.bias = float(Theta[-1, 0])
        else:
            self.S = S.T.astype(np.complex64)                 # (c, D)
            self.bias = Theta[-1].astype(np.float32)          # (c,)
        return self

    def eval(self, points, chunk=8192):
        """Same readout as every other field in this package (routed
        through accel.readout — GPU when present). Returns (n,) for
        scalar fits, (n, c) for multi-channel fits."""
        from .accel import readout
        return readout(points, self.W, self.S, chunk=chunk) + self.bias


def _load_photo(res=128):
    """Grace Hopper's portrait (ships with matplotlib), grayscale [0,1]."""
    import matplotlib
    from PIL import Image
    import os
    path = os.path.join(os.path.dirname(matplotlib.__file__),
                        "mpl-data", "sample_data", "grace_hopper.jpg")
    img = Image.open(path).convert("L").resize((res, res))
    return np.asarray(img, dtype=np.float32) / 255.0


def _psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def demo(dim=4096, seed=0, save_png=True):  # dim: total D of largest fit
    print("== Fitting holograms from data (ridge regression) ==")

    # -- a real photograph, multi-band ----------------------------------
    res = 128
    img = _load_photo(res)
    xs = np.linspace(0, 1, res)
    P = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    y = img.reshape(-1)
    rng = np.random.default_rng(seed)
    train = rng.choice(len(P), size=len(P) * 3 // 4, replace=False)
    test = np.setdiff1d(np.arange(len(P)), train)

    # finest band ~ one pixel at res=128: a narrower kernel than the
    # sample spacing can only memorize training pixels, never interpolate
    sigma_scales = [0.12, 0.04, 0.016, 0.008]
    proportions = np.array([1, 2, 4, 9])
    panels = [("original photo", img)]
    print(f"  {'total D':>8} {'bytes':>9} {'train PSNR':>11} "
          f"{'test PSNR':>10}")
    for total in [2048, 8192, max(16384, dim)]:
        dims = np.maximum(proportions * total // proportions.sum(), 32)
        bands = FrequencyBands(dims, sigma_scales, ndim=2, seed=seed)
        reg = HoloRegressor(bands).fit(P[train], y[train], lam=1e-2)
        pred = reg.eval(P)
        print(f"  {bands.D:>8} {8*bands.D:>9,} "
              f"{_psnr(pred[train], y[train]):>11.1f} "
              f"{_psnr(pred[test], y[test]):>10.1f}")
        panels.append((f"hologram, D={bands.D}\n({8*bands.D//1024}KB)",
                       pred.reshape(res, res)))
    print(f"  (the {res}x{res} photo itself is {res*res//1024}KB as uint8; "
          "the fitted bundle is a field — evaluate it anywhere, "
          "bind it, replicate it)")

    # -- fit vs. forward bundling on a known mixture --------------------
    print("  -- regression vs. forward bundling (same d, same kernel) --")
    from .field import GaussianSplatField
    rng = np.random.default_rng(seed + 1)
    sigma, d = 0.04, 4096
    truth_field = GaussianSplatField(d, np.eye(2) * sigma ** 2, seed=seed)
    for _ in range(200):
        truth_field.add_splat(rng.uniform(0.05, 0.95, 2),
                              float(rng.uniform(0.5, 1.0)))
    Ptr = rng.uniform(0, 1, size=(12000, 2)).astype(np.float32)
    Pte = rng.uniform(0, 1, size=(4000, 2)).astype(np.float32)
    y_clean = truth_field.exact(Ptr).astype(np.float32)
    y_te = truth_field.exact(Pte)
    bundle_rmse = float(np.sqrt(np.mean(
        (truth_field.eval(Pte) - y_te) ** 2)))
    print(f"  {'source':>28} {'held-out RMSE':>14}")
    print(f"  {'forward bundle (knows splats)':>28} {bundle_rmse:>14.4f}")
    for noise in [0.0, 0.1]:
        bands = FrequencyBands([d], [sigma], ndim=2, seed=seed)
        y_fit = y_clean + rng.normal(0, noise, len(y_clean)) \
            .astype(np.float32)
        reg = HoloRegressor(bands).fit(Ptr, y_fit)
        r = float(np.sqrt(np.mean((reg.eval(Pte) - y_te) ** 2)))
        label = f"fit, sample noise {noise:.1f}"
        print(f"  {label:>28} {r:>14.4f}")
    print("  (the fit never saw a splat — only samples; with noisy "
          "samples it still beats the forward bundle: regression "
          "averages, bundling can't)")

    if save_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print()
            return
        fig, axes = plt.subplots(1, len(panels),
                                 figsize=(3.6 * len(panels), 4.1))
        for ax, (title, im) in zip(axes, panels):
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.suptitle("A photograph fit as one complex vector: "
                     "ridge regression whose weights ARE the hologram",
                     fontsize=11)
        import os
        os.makedirs("out", exist_ok=True)
        fig.savefig("out/fit_photo.png", dpi=110)
        plt.close(fig)
        print("  saved out/fit_photo.png")
    print()


__all__ = ["FrequencyBands", "HoloRegressor"]
