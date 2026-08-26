"""Gaussian-splat field as a holographic bundle (the repo's namesake).

Fractional power encoding + Bochner's theorem: encode a point p as the
phasor vector e^{i W p} with frequency rows w_j ~ N(0, Sigma^{-1}).
Then for a splat centered at mu,

    (1/d) Re< e^{i W p}, e^{i W mu} > ~ exp(-1/2 (p-mu)^T Sigma^{-1} (p-mu))

i.e. the inner product IS the Gaussian kernel (random Fourier features,
in VSA clothing). A whole scene of N splats bundles into ONE complex64
vector S = sum_k alpha_k e^{i W mu_k}, and evaluating the mixture at any
point is a single inner product — with Monte Carlo noise ~sqrt(1/(2d))
per splat mass, the capacity trade of every other structure here.

Limitation worth knowing: one shared W bakes in one shared Sigma. True
3DGS gives every splat its own covariance; with a single shared basis
that requires multiple frequency 'bands' (one W per covariance class) or
per-splat frequencies, which reintroduces per-primitive storage.
"""

import numpy as np


class GaussianSplatField:
    def __init__(self, dim, sigma, seed=0):
        """sigma: (k, k) covariance shared by all splats. A scalar is a
        standard deviation: it becomes the isotropic covariance
        sigma^2 * I in 2-D."""
        sigma = np.atleast_2d(np.asarray(sigma, dtype=np.float64))
        if sigma.shape == (1, 1):
            sigma = np.eye(2) * float(sigma[0, 0]) ** 2
        self.k = sigma.shape[0]
        self.dim = dim
        self.sigma_inv = np.linalg.inv(sigma)
        rng = np.random.default_rng(seed)
        B = np.linalg.cholesky(self.sigma_inv)  # B B^T = Sigma^-1
        self.W = (rng.standard_normal((dim, self.k)) @ B.T).astype(np.float32)
        self.S = np.zeros(dim, dtype=np.complex64)
        self.splats = []  # (mu, alpha), kept only for exact ground truth

    def add_splat(self, mu, alpha=1.0):
        mu = np.asarray(mu, dtype=np.float32)
        self.S += np.complex64(alpha) * np.exp(1j * (self.W @ mu)) \
                                          .astype(np.complex64)
        self.splats.append((mu, float(alpha)))

    def eval(self, points, chunk=8192):
        """Holographic evaluation of the mixture at points (n, k),
        GPU-dispatched through accel.readout when MLX is present."""
        from . import accel
        return accel.readout(points, self.W, self.S, chunk=chunk)

    def exact(self, points):
        """Ground-truth mixture, from the explicit splat list."""
        points = np.asarray(points, dtype=np.float64)
        out = np.zeros(len(points))
        for mu, alpha in self.splats:
            delta = points - mu
            out += alpha * np.exp(-0.5 * np.einsum(
                "ij,jk,ik->i", delta, self.sigma_inv, delta))
        return out


def _make_scene(dim, seed, n_splats):
    theta = np.deg2rad(30)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    sigma = R @ np.diag([0.06 ** 2, 0.02 ** 2]) @ R.T  # anisotropic, shared
    field = GaussianSplatField(dim, sigma, seed=seed)
    rng = np.random.default_rng(seed + 7)
    for _ in range(n_splats):
        field.add_splat(rng.uniform(0.1, 0.9, size=2),
                        alpha=float(rng.uniform(0.5, 1.0)))
    return field


def demo(dim=4096, seed=0, save_png=True):
    print("== Gaussian splat field: a 'scene' as one complex64 vector ==")
    n_splats = 80
    grid = 120
    xs = np.linspace(0, 1, grid)
    P = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)

    truth = None
    images = {}
    print(f"{'dim d':>7} {'scene bytes':>12} {'RMSE':>7} {'rel to peak':>12}")
    for d in [1024, 4096, 16384]:
        field = _make_scene(d, seed, n_splats)
        if truth is None:
            truth = field.exact(P)
        approx = field.eval(P)
        rmse = float(np.sqrt(np.mean((approx - truth) ** 2)))
        images[d] = approx.reshape(grid, grid)
        print(f"{d:>7} {8 * d:>12} {rmse:>7.3f} {rmse/truth.max():>12.1%}")
    explicit = n_splats * 3 * 4  # mu (2 floats) + alpha, float32
    print(f"(explicit splat list would be {explicit} bytes for "
          f"{n_splats} splats; the hologram is fixed-size at any N)")

    if save_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping image")
            print()
            return
        panels = [("ground truth\n(explicit mixture)", truth.reshape(grid, grid))]
        panels += [(f"holographic, d={d}", img) for d, img in images.items()]
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img, origin="lower", extent=(0, 1, 0, 1),
                      cmap="magma", vmin=0, vmax=truth.max())
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle(f"{n_splats} anisotropic Gaussian splats bundled into one "
                     "complex vector — evaluated by inner product", fontsize=11)
        fig.tight_layout()
        import os
        os.makedirs("out", exist_ok=True)
        path = os.path.join("out", "field_comparison.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"saved {path}")
    print()
