"""Dynamic holograms: animation as algebra inside the bundle.

Three questions, measured (2-D scenes, the spectral encoder's own API):

1. MOTION IS FREE — rigid translation of an object's sub-bundle is one
   elementwise phase ramp (`translate_bundle`), algebraically identical
   to re-encoding the moved splats. Verify exactness; compare cost.
2. ONE VECTOR HOLDS THE WHOLE ANIMATION — bind each frame's bundle with
   a time codeword e^{i w_t t} and SUM. Querying time t* is one unbind;
   frames within the time-kernel width blend (continuous playback /
   motion blur). What does the capacity law charge per frame?
3. WHERE WAS OBJECT k AT TIME t? — additionally bind each object with
   an id phasor: one vector answers (id, t) queries by two unbinds and
   an argmax. Localization error vs ground-truth trajectories.

Run:  python examples/dynamic_prototype.py            (figure -> out/)
"""

import time

import numpy as np

from holo.spectral import (SplatScene, decode_field, eval_scene_exact,
                           sample_frequencies, spectral_bundle,
                           translate_bundle)

RNG = np.random.default_rng(7)
D = 1 << 14                    # hypervector dimensionality
S_MIN, S_MAX = 0.02, 0.035     # splat axis scales (unit box)
SIGMA_RHO = 1.3 / S_MIN        # codebook per the house recipe
K, SPLATS = 5, 25              # objects x splats each


def make_object(rng):
    """A small anisotropic splat cluster in LOCAL coordinates."""
    mu = rng.normal(0.0, 0.03, (SPLATS, 2)).astype(np.float32)
    th = rng.uniform(0, np.pi, SPLATS)
    a = rng.uniform(S_MIN, S_MAX, SPLATS)
    b = rng.uniform(S_MIN, S_MAX, SPLATS)
    R = np.stack([np.stack([np.cos(th), -np.sin(th)], -1),
                  np.stack([np.sin(th), np.cos(th)], -1)], 1)
    cov = np.einsum("nab,nb,ncb->nac", R, np.stack([a**2, b**2], -1), R)
    amp = rng.uniform(0.6, 1.0, (SPLATS, 1)).astype(np.float32)
    return SplatScene(mu, cov.astype(np.float32), amp)


def place(obj, x):
    """The re-encode baseline: the object's splats moved to x."""
    return SplatScene((obj.mu + np.asarray(x, np.float32)), obj.cov,
                      obj.amp)


def trajectory(k, t):
    """Object k's position at time t in [0, 1]: a Lissajous loop."""
    cx, cy = 0.5, 0.5
    r = 0.16 + 0.03 * k
    return np.array([cx + r * np.cos(2 * np.pi * (t + k / K)),
                     cy + r * np.sin(2 * np.pi * (2 * t + k / K) *
                                     (1 if k % 2 else -1))],
                    dtype=np.float32)


def merged_exact(objs, t):
    scenes = [place(o, trajectory(k, t)) for k, o in enumerate(objs)]
    return SplatScene(np.concatenate([s.mu for s in scenes]),
                      np.concatenate([s.cov for s in scenes]),
                      np.concatenate([s.amp for s in scenes]))


def main():
    print(f"== dynamic holograms: motion as algebra (d={D}, "
          f"{K} objects x {SPLATS} splats) ==")
    freqs = sample_frequencies(D, 2, SIGMA_RHO, RNG)
    objs = [make_object(RNG) for _ in range(K)]
    subs = [spectral_bundle(o, freqs) for o in objs]     # (1, d) each
    pts = RNG.uniform(0.05, 0.95, (2000, 2)).astype(np.float32)

    # -- 1: phase-ramp motion vs re-encode ------------------------------
    t0 = 0.31
    tic = time.time()
    ramped = sum(translate_bundle(subs[k], freqs, trajectory(k, t0))
                 for k in range(K))
    ramp_s = time.time() - tic
    tic = time.time()
    reenc = spectral_bundle(merged_exact(objs, t0), freqs)
    reenc_s = time.time() - tic
    bundle_gap = np.abs(ramped - reenc).max() / np.abs(reenc).max()
    est = decode_field(ramped, freqs, SIGMA_RHO, pts)
    exact = eval_scene_exact(merged_exact(objs, t0), pts)
    rel = np.linalg.norm(est - exact) / np.linalg.norm(exact)
    print(f"  1. ramped-vs-reencoded bundle gap {bundle_gap:.1e} "
          f"(shift theorem is exact); decode rel err {rel:.3f}")
    print(f"     cost/frame at {K*SPLATS} splats: ramps {1e3*ramp_s:.1f} "
          f"ms vs re-encode {1e3*reenc_s:.1f} ms — a wash at toy scale")
    # the ramp is O(K d) REGARDLESS of splat count; re-encode is O(N d).
    # measure at capture scale: one object, 20k splats
    big = make_object(RNG)
    reps = 20_000 // SPLATS
    big = SplatScene(np.tile(big.mu, (reps, 1))
                     + RNG.normal(0, 0.05, (reps * SPLATS, 2))
                     .astype(np.float32),
                     np.tile(big.cov, (reps, 1, 1)),
                     np.tile(big.amp, (reps, 1)))
    tic = time.time()
    Sbig = spectral_bundle(big, freqs)
    enc_s = time.time() - tic
    tic = time.time()
    for _ in range(10):
        translate_bundle(Sbig, freqs, [0.1, 0.2])
    ramp_big = (time.time() - tic) / 10
    print(f"     at {big.n:,} splats: ramps {1e3*ramp_big:.1f} ms vs "
          f"re-encode {1e3*enc_s:.0f} ms ({enc_s/ramp_big:.0f}x — the "
          "ramp cost never grows with splat count)")

    # -- 2: the whole animation in ONE vector ---------------------------
    print("  2. time-binding: T frames summed into one bundle "
          "(query = one unbind)")
    print(f"     {'T':>4} {'stored-frame err':>17} {'mid-frame err':>14} "
          f"{'sqrt(T) guide':>14}")
    base = None
    for T in [4, 8, 16, 32]:
        sig_t = 0.55 / T                       # kernel ~ frame spacing
        wt = RNG.normal(0.0, 1.0 / sig_t, D).astype(np.float32)
        times = (np.arange(T) + 0.5) / T
        S4 = sum(np.exp(1j * wt * t).astype(np.complex64)
                 * sum(translate_bundle(subs[k], freqs, trajectory(k, t))
                       for k in range(K))
                 for t in times).astype(np.complex64)

        def query(tq, S4=S4, wt=wt, times=times, sig_t=sig_t):
            Sq = S4 * np.exp(-1j * wt * tq).astype(np.complex64)
            norm = np.exp(-0.5 * ((times - tq) / sig_t) ** 2).sum()
            return decode_field(Sq, freqs, SIGMA_RHO, pts) / norm

        errs = []
        for tq in times[::max(T // 4, 1)]:
            e = eval_scene_exact(merged_exact(objs, tq), pts)
            errs.append(np.linalg.norm(query(tq) - e) / np.linalg.norm(e))
        mids = []
        for tq in (times[:-1] + times[1:])[::max(T // 4, 1)] / 2:
            e = eval_scene_exact(merged_exact(objs, tq), pts)
            mids.append(np.linalg.norm(query(tq) - e) / np.linalg.norm(e))
        err = float(np.mean(errs))
        base = base or err
        print(f"     {T:>4} {err:>17.3f} {float(np.mean(mids)):>14.3f} "
              f"{base * np.sqrt(T / 4):>14.3f}")

    # -- 3: where_is(object, t) from one vector -------------------------
    T = 16
    sig_t = 0.55 / T
    wt = RNG.normal(0.0, 1.0 / sig_t, D).astype(np.float32)
    times = (np.arange(T) + 0.5) / T
    ids = np.exp(1j * RNG.uniform(-np.pi, np.pi, (K, D))) \
        .astype(np.complex64)
    S = sum(ids[k] * np.exp(1j * wt * t).astype(np.complex64)
            * translate_bundle(subs[k], freqs, trajectory(k, t))
            for k in range(K) for t in times).astype(np.complex64)

    g = np.linspace(0.02, 0.98, 96, dtype=np.float32)
    grid = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    locs, truth, tracks = [], [], []
    qts = np.linspace(0.05, 0.95, 10)
    for k in range(K):
        row = []
        for tq in qts:                       # incl. BETWEEN stored frames
            Sq = S * np.conj(ids[k]) * np.exp(-1j * wt * tq) \
                .astype(np.complex64)
            f = decode_field(Sq, freqs, SIGMA_RHO, grid)[:, 0]
            p = grid[int(np.argmax(f))]
            row.append(p)
            locs.append(p)
            truth.append(trajectory(k, tq))
        tracks.append(np.array(row))
    err = np.linalg.norm(np.array(locs) - np.array(truth), axis=1)
    print(f"  3. where_is(id, t) over {K} objects x {len(qts)} times "
          f"(one vector, {S.nbytes // 1024} KB): median loc err "
          f"{np.median(err):.3f} box units (~{np.median(err)/S_MAX:.1f} "
          f"splat scales), worst {err.max():.3f}")

    # -- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.1))
    fig.patch.set_facecolor("#f9f9f7")
    show = np.linspace(0.1, 0.9, 4)
    S4 = sum(np.exp(1j * wt * t).astype(np.complex64)
             * sum(translate_bundle(subs[k], freqs, trajectory(k, t))
                   for k in range(K)) for t in times).astype(np.complex64)
    for ax, tq in zip(axes[:4], show):
        Sq = S4 * np.exp(-1j * wt * tq).astype(np.complex64)
        norm = np.exp(-0.5 * ((times - tq) / sig_t) ** 2).sum()
        f = decode_field(Sq, freqs, SIGMA_RHO, grid)[:, 0] / norm
        ax.imshow(f.reshape(96, 96), origin="lower", cmap="magma",
                  extent=[0, 1, 0, 1])
        ax.set_title(f"decoded at t={tq:.2f}\n(one 4-D bundle, T={T})",
                     fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    ax = axes[4]
    tt = np.linspace(0, 1, 200)
    for k in range(K):
        c = plt.cm.viridis(k / (K - 1))
        gt = np.array([trajectory(k, t) for t in tt])
        ax.plot(gt[:, 0], gt[:, 1], "-", color=c, lw=1, alpha=0.5)
        ax.plot(tracks[k][:, 0], tracks[k][:, 1], "o", color=c, ms=4)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("where_is(id, t): argmax tracks (dots)\n"
                 "vs true trajectories (lines)", fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("dynamic holograms: the whole animation is one complex64 "
                 "vector; playback and object queries are unbindings",
                 fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("out/dynamic_prototype.png", dpi=140, bbox_inches="tight",
                facecolor="#f9f9f7")
    print("  figure: out/dynamic_prototype.png")


if __name__ == "__main__":
    main()
