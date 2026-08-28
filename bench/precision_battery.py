"""Does the projection solve actually need float64? — a serial battery.

The question came from asking why the solve runs on NumPy when
holo/accel.py puts encode and decode on Metal. The first answer given
was "Metal has no float64 and our Gram is conditioned 1.6e20", and half
of that was wrong: 1.6e20 describes a matrix this pipeline never
inverts. Truncation to 25% of the spectrum — which is mandatory anyway —
leaves an operator conditioned ~4e3, which float32 can carry.

So the fp64 question was never settled, and this battery settles it,
along with what Metal can actually run (MLX refuses eigh/inv/solve/svd
on the GPU at EVERY dtype, which is a different and larger obstacle than
precision) and whether bits and rank should be chosen together rather
than separately (arXiv:1811.00155).

WHY A RUNNER AND NOT A SCRIPT. This machine is shared with splat
training and with other agent sessions. Four measurement runs died in
one evening: two OOM kills, one Metal command-buffer fault as the
innocent victim of another process, and one silent SIGKILL. So:

  - one experiment at a time, never two;
  - headroom checked before each, and on refusal it WAITS rather than
    dying, because the contention is someone else's real work;
  - one JSON line appended after each, so a kill costs one experiment
    rather than the batch;
  - --resume skips what is already recorded.

RUN IT WITH `HDC_BACKEND=numpy`. holo/accel.py picks MLX/Metal at
import time, so spectral_bundle would share the GPU with any splat
training on the box and die as an innocent victim of that process's
faults — which has already happened once. B2 is unaffected by the
setting because it imports mlx.core directly, so forcing the library
backend off costs nothing and the one experiment that must touch Metal
still does.

NOT RUN, AND WHY. Three planned experiments were dropped once A1 landed,
recorded here rather than quietly omitted:

  A3, end-to-end float32 — decode is LINEAR in the solved coefficients,
      so A1's measured 1.4e-5 perturbation bounds the slice error's
      change at the same order. It would confirm linearity, not
      precision, and cost a full pipeline run to do it.
  A4, iterative refinement — gated on float32 factorisation failing.
      See A2.
  E,  Ozaki-style split GEMM — gated on float32 being INSUFFICIENT.
      A1 says it is sufficient for every setting the pipeline ships, so
      emulating float64 on Metal solves a problem we do not have.

Usage:
    HDC_BACKEND=numpy python -m bench.precision_battery --list
    python -m bench.precision_battery --only A1 --only B1
    python -m bench.precision_battery --phase A --resume
    python -m bench.precision_battery --report
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from collections import namedtuple

import numpy as np

from holo import budget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "out", "precision", "battery.jsonl")

#: `data/` is gitignored (it holds real captures), so a worktree does not
#: have one — pass --capture, or set HOLO_CAPTURE. Defaulting silently to
#: a missing path is how the first run of this battery died.
#: The real results path as it was at import — the one worth protecting.
#: Tests point RESULTS elsewhere, and writing THERE is always allowed.
_REAL_RESULTS = RESULTS

CAPTURE = os.environ.get("HOLO_CAPTURE",
                         os.path.join(ROOT, "data", "scan-tucson.spz"))

#: How long to wait for a shared machine to free up before giving up.
HEADROOM_POLL_S = 60
HEADROOM_LIMIT_S = 3600

Experiment = namedtuple("Experiment", "id phase need_gb minutes summary fn")
REGISTRY = []


def experiment(eid, phase, need_gb, minutes, summary):
    def wrap(fn):
        REGISTRY.append(Experiment(eid, phase, need_gb, minutes, summary, fn))
        return fn
    return wrap


# ---------------------------------------------------------------------------
# Shared fixtures — loaded once, because build_scene + encode_bands is a
# minute and most experiments want the same cells.
# ---------------------------------------------------------------------------

_CACHE = {}


def pipeline():
    """The SHIPPED projection pipeline, imported rather than copied.

    The battery must measure the code that runs in production; a battery
    that re-implements build_gram or BandSolver measures its own copy and
    proves nothing about the pipeline.
    """
    if "rp" not in _CACHE:
        path = os.path.join(ROOT, "examples", "run_projection_pipeline.py")
        spec = importlib.util.spec_from_file_location("_rp", path)
        mod = importlib.util.module_from_spec(spec)
        # exec_module sets __name__ to the spec name, so the driver's
        # __main__ block does not run and argv is not touched — do NOT
        # "defensively" clear sys.argv here, it silently breaks the
        # runner's own arguments if an experiment ever loads first
        spec.loader.exec_module(mod)
        _CACHE["rp"] = mod
    return _CACHE["rp"]


def capture_path():
    return _CACHE.get("capture") or CAPTURE


def set_capture(path):
    _CACHE["capture"] = path


def scene():
    """(scene, members, books) for the saguaro capture."""
    if "scene" not in _CACHE:
        from holo.capture import band_codebooks, build_scene, encode_bands
        if not os.path.exists(capture_path()):
            raise SystemExit(
                "capture not found: %s\n"
                "data/ is gitignored, so a worktree has none. Pass "
                "--capture /abs/path/to/scan-tucson.spz (or set "
                "HOLO_CAPTURE)." % capture_path())
        sc, smax, box = build_scene(capture_path(), verbose=False)
        books = band_codebooks(np.random.default_rng(42))
        _fwd, members = encode_bands(sc, smax, books, verbose=False)
        _CACHE["scene"] = (sc, members, books, box)
    return _CACHE["scene"]


def band_rhs(band, ncells=64):
    """Real right-hand sides for `ncells` cells of one band, plus the
    geometry needed to build that band's operator."""
    key = ("rhs", band, ncells)
    if key not in _CACHE:
        from holo.capture import BANDS
        rp = pipeline()
        sc, members, books, _box = scene()
        name, _cap, cell = next(b for b in BANDS if b[0] == band)
        freqs, _rho, weights = books[name]
        fd = freqs.astype(np.float64)
        s = (cell / 2) / 2
        keys = list(members[name].keys())[:ncells]
        centres = [(np.array(k, np.float64) + 0.5) * cell for k in keys]
        rhs = np.concatenate(
            [rp.window_bundle(sc, members[name][k], c, s, freqs)
             for k, c in zip(keys, centres)], axis=0)
        _CACHE[key] = (fd, s, rhs, weights)
    return _CACHE[key]


def cell_scenes(band="xfine", ncells=24, seed=7):
    """Real cells as CELL-LOCAL scenes, plus evaluation points.

    Per-cell reconstruction rather than a whole-scene slice, because the
    (d, bits) question is about what one cell's bundle can hold. A slice
    restricted to the sparse bands would referee against 347 splats;
    xfine cells carry ~330 splats EACH, which is the load that matters.
    Mirrors examples/run_analytic_projection.py's measure_cell.
    """
    key = ("cells", band, ncells, seed)
    if key not in _CACHE:
        from holo.capture import BANDS
        from holo.spectral import SplatScene
        sc, members, _books, _box = scene()
        _n, _cap, cell = next(b for b in BANDS if b[0] == band)
        half = cell / 2
        rng = np.random.default_rng(seed)
        out = []
        for k in list(members[band].keys())[:ncells]:
            ids = members[band][k]
            centre = (np.array(k, np.float64) + 0.5) * cell
            local = SplatScene(mu=(sc.mu[ids] - centre).astype(np.float32),
                               cov=sc.cov[ids], amp=sc.amp[ids])
            pts = (rng.uniform(-1, 1, (1500, 3)) * half).astype(np.float32)
            out.append((local, pts))
        _CACHE[key] = (out, half)
    return _CACHE[key]


def as_complex32(v):
    """complex64 -> 2x float16 -> back. 4 bytes/component."""
    parts = np.stack([v.real, v.imag], -1).astype(np.float16)
    return (parts[..., 0].astype(np.float32)
            + 1j * parts[..., 1].astype(np.float32)).astype(np.complex64)


def as_bfp16(v, block):
    """Block floating point: one shared scale per `block` components,
    float16 mantissas. `block=len(v)` is a single shared exponent for the
    whole vector, which is what pack_polar already does with its
    `scale = m.max()` — so the sweep over block size measures whether
    smaller blocks buy anything, not whether scaling helps at all."""
    v = np.asarray(v, np.complex64).ravel()
    n = len(v)
    pad = (-n) % block
    w = np.concatenate([v, np.zeros(pad, np.complex64)]).reshape(-1, block)
    scale = np.abs(w).max(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    q = as_complex32((w / scale).astype(np.complex64))
    return (q * scale).ravel()[:n].astype(np.complex64)


def rel(got, ref):
    n = np.linalg.norm(ref)
    return float(np.linalg.norm(got - ref) / n) if n else float("nan")


# ---------------------------------------------------------------------------
# Phase A — does the solve need float64?
# ---------------------------------------------------------------------------

SETTINGS = [("keep", 0.25), ("keep", 0.10),
            ("tikhonov", 1e-1), ("tikhonov", 1e-3), ("tikhonov", 1e-6)]


@experiment("A1", "A", 6.0, 12,
            "fp32 APPLICATION of the fp64 operator, per band x setting")
def a1_fp32_application():
    """Operator built in float64, applied in float32.

    This is the cheapest thing that could close the question, so it runs
    first. If applying in fp32 is clean, the per-cell solve — which is
    the part that scales with cell count — can move to a float32 device
    even though the factorisation cannot.
    """
    rp = pipeline()
    out = {}
    for band in ("xfine", "fine"):
        fd, s, rhs, _w = band_rhs(band)
        solver = rp.BandSolver(rp.build_gram(fd, s))
        rhs64, rhs32 = rhs.astype(np.complex128), rhs.astype(np.complex64)
        for kind, val in SETTINGS:
            m64, how = solver.operator((kind, val))
            ref = (m64 @ rhs64.T).T
            got = (m64.astype(np.float32) @ rhs32.T).T.astype(np.complex128)
            out["%s/%s" % (band, how)] = rel(got, ref)
            del m64
        solver.close()
    return out


@experiment("A2", "A", 6.0, 25,
            "fp32 FACTORISATION: eigh/inv themselves in single precision")
def a2_fp32_factorisation():
    """The harder ask: compute the operator in fp32, not just apply it.

    This is what decides whether a float32-only device could own the
    whole solve. Reported two ways — deviation of the operator itself,
    and deviation of what it produces — because an operator can be far
    off in norm while still acting correctly on the vectors we feed it.
    """
    rp = pipeline()
    out = {}
    for band in ("xfine", "fine"):
        fd, s, rhs, _w = band_rhs(band)
        g64 = rp.build_gram(fd, s)
        rhs64 = rhs.astype(np.complex128)
        # ONE solver per precision per band, not one per setting: a fresh
        # BandSolver throws away the cached eigendecomposition, which is
        # 106 s at d=8192. Building it inside the loop cost eight eigh
        # calls where two would do.
        s64 = rp.BandSolver(g64)
        s32 = rp.BandSolver(g64.astype(np.float32))
        print("    %s: gram built, factorising both precisions" % band,
              flush=True)
        for kind, val in SETTINGS:
            m64, how = s64.operator((kind, val))
            m32, _ = s32.operator((kind, val))
            ref = (m64 @ rhs64.T).T
            got = (m32.astype(np.float64) @ rhs64.T).T
            out["%s/%s" % (band, how)] = {
                "operator": rel(m32.astype(np.float64), m64),
                "action": rel(got, ref),
            }
            print("      %-22s operator=%.2e action=%.2e"
                  % (how, out["%s/%s" % (band, how)]["operator"],
                     out["%s/%s" % (band, how)]["action"]), flush=True)
            del m64, m32
        s64.close()
        s32.close()
    return out


# ---------------------------------------------------------------------------
# Phase B — what can Metal actually run?
# ---------------------------------------------------------------------------

@experiment("B1", "B", 1.0, 2,
            "MLX capability census: op x dtype x device")
def b1_mlx_census():
    """Which MLX operations run on Metal, at which dtypes.

    Recorded rather than assumed, because the answer is the real
    obstacle: precision is not what stops the factorisation moving to
    the GPU, the absence of GPU linalg is.
    """
    try:
        import mlx.core as mx
    except ImportError:
        return {"mlx": "not installed"}
    n, out = 64, {"mlx_version": getattr(mx, "__version__", "?")}
    a = np.eye(n) + 0.01 * np.random.default_rng(0).standard_normal((n, n))
    a = a @ a.T
    dtypes = [("float16", mx.float16), ("float32", mx.float32),
              ("float64", mx.float64), ("complex64", mx.complex64)]
    ops = ["matmul", "eigh", "inv", "solve", "cholesky", "qr", "svd"]
    for dname, dt in dtypes:
        for dev, devname in ((mx.gpu, "gpu"), (mx.cpu, "cpu")):
            for op in ops:
                key = "%s/%s/%s" % (op, dname, devname)
                try:
                    with mx.stream(dev):
                        x = mx.array(a, dtype=dt)
                        if op == "matmul":
                            r = x @ x
                        elif op == "solve":
                            r = mx.linalg.solve(x, x)
                        else:
                            r = getattr(mx.linalg, op)(x)
                        mx.eval(r)
                    out[key] = "ok"
                except Exception as exc:
                    msg = str(exc).replace("\n", " ")
                    out[key] = ("unsupported" if "not supported" in msg
                                or "not yet supported" in msg else
                                msg[:90])
    return out


# ---------------------------------------------------------------------------
# Phase C — dynamic range, which is what decides whether fp16 is viable
# ---------------------------------------------------------------------------

@experiment("C1", "C", 6.0, 10,
            "dynamic-range census of every array the solve touches")
def c1_dynamic_range():
    """float16 bottoms out at 6.1e-5 normal / 6.0e-8 subnormal.

    A first probe found one band's right-hand side spanning 1.22e-14 to
    2.14e-04 — ten orders in a single array — which is why this is
    measured per array rather than assumed. The Apple-Silicon result
    (arXiv:2605.28451) is that range, not mantissa, is what breaks fp16
    here; this says whether that applies to us.
    """
    rp = pipeline()
    f16 = np.finfo(np.float16)
    out = {"float16_tiny": float(f16.tiny), "float16_max": float(f16.max),
           "float16_eps": float(f16.eps)}
    for band in ("xfine", "fine"):
        fd, s, rhs, _w = band_rhs(band)
        g = rp.build_gram(fd, s)
        m, _how = rp.BandSolver(g.copy()).operator(("keep", 0.25))
        for label, arr in (("gram", g), ("operator", m), ("rhs", rhs)):
            mag = np.abs(arr)
            nz = mag[mag > 0]
            out["%s/%s" % (band, label)] = {
                "min_nonzero": float(nz.min()), "max": float(mag.max()),
                "decades": float(np.log10(mag.max() / nz.min())),
                "below_f16_tiny_pct": float(100 * np.mean(nz < f16.tiny)),
            }
        del g, m
    return out



@experiment("B2", "B", 4.0, 5,
            "Metal matmul at the shapes the per-cell solve actually uses")
def b2_metal_gemm():
    """Timing only, so the operands are random — the clock does not care.

    The census says complex64 matmul runs on Metal while every
    factorisation does not, so this measures the only piece that could
    actually move: applying an already-built operator to a batch of
    right-hand sides.
    """
    try:
        import mlx.core as mx
    except ImportError:
        return {"mlx": "not installed"}
    d, nb, reps = 8192, 1024, 3
    rng = np.random.default_rng(0)
    m32 = rng.standard_normal((d, d)).astype(np.float32)
    r64 = (rng.standard_normal((nb, d))
           + 1j * rng.standard_normal((nb, d))).astype(np.complex128)
    out = {}

    def clock(fn):
        fn()                                    # warm up
        t = time.time()
        for _ in range(reps):
            fn()
        return (time.time() - t) / reps

    out["numpy_fp64_shipped_s"] = clock(
        lambda: m32.astype(np.float64) @ r64.T)
    r32 = r64.astype(np.complex64)
    out["numpy_fp32_s"] = clock(lambda: m32 @ r32.T)
    mm, mr = mx.array(m32), mx.array(r32)
    mx.eval(mm, mr)

    def metal():
        mx.eval(mm @ mr.T)
    out["metal_complex64_s"] = clock(metal)
    out["metal_speedup_vs_shipped"] = round(
        out["numpy_fp64_shipped_s"] / out["metal_complex64_s"], 2)
    ref = m32.astype(np.float64) @ r64.T
    out["metal_vs_fp64_rel"] = rel(np.array(mm @ mr.T, copy=False)
                                   .astype(np.complex128), ref)
    return out


@experiment("C2", "C", 4.0, 8,
            "complex32 vs HG-8 vs BFP16 on real cell bundles")
def c2_codecs():
    """Rate-distortion on the bundles this pipeline actually stores.

    Reported as bytes AND error rather than a single pairing, because
    the formats do not line up at equal width: complex32 is 4 bytes per
    component, HG-8 is 2, HG-4 is 1.
    """
    from holo.phase import pack_polar, unpack
    from holo.spectral import spectral_bundle
    _sc, _members, books, _box = scene()
    freqs = books["xfine"][0]
    cells, _half = cell_scenes("xfine", ncells=12)
    acc = {}
    for local, _pts in cells:
        b = spectral_bundle(local, freqs)[0].astype(np.complex64)
        cand = {"complex64 (8 B/comp)": (8.0, b),
                "complex32 2xf16 (4 B/comp)": (4.0, as_complex32(b))}
        for bits in (8, 4):
            cand["HG-%d (%.1f B/comp)" % (bits, bits / 4.0)] = (
                bits / 4.0, unpack(pack_polar(b, bits=bits)))
        for blk in (64, 256):
            cand["BFP16 blk=%d (4 B/comp)" % blk] = (4.0, as_bfp16(b, blk))
        for label, (bpc, q) in cand.items():
            acc.setdefault(label, [bpc, []])[1].append(rel(q, b))
    return {k: {"bytes_per_component": v[0],
                "median_rel_err": float(np.median(v[1]))}
            for k, v in acc.items()}


@experiment("D1", "D", 4.0, 12,
            "LP-RFF: rank and bits chosen together at a fixed byte budget")
def d1_rank_vs_bits():
    """Fixed bytes per cell; spend them on more dimensions or more bits.

    The standing negative says doubling d bought 2-4% for +600 MB — but
    that was measured at CONSTANT bits. arXiv:1811.00155 says that is
    the wrong axis to hold fixed.

    The budget is set from ACTUAL packed bytes, not nominal bits, because
    pack_polar has a nibble floor: _pack_block puts anything at or below
    4 bits into a nibble, so a 2-bit code occupies the same space as a
    4-bit one (measured: 80 bytes for both at d=64). That caps this sweep
    at 4 bits — d=32768 @ 2 bit would cost twice the budget, not the
    same — and it is why the fourth row is a control rather than a rate
    point. A sub-nibble packer is what it would take to go further, and
    2-bit round-trips at 0.50 relative on its own, so it is not obviously
    worth writing.
    """
    from holo.capture import BANDS, S_LO
    from holo.phase import pack_polar, unpack
    from holo.spectral import (
        decode_field,
        eval_scene_exact,
        sample_frequencies,
        spectral_bundle,
    )
    _n, cap, _cell = BANDS[0]                      # xfine
    cells, _half = cell_scenes("xfine", ncells=12)
    n_comp = 3 + max(2, round(np.log2(cap / S_LO)))
    rho = list(1.0 / np.geomspace(S_LO, cap, n_comp))
    out = {}
    for d, bits in ((4096, 16), (8192, 8), (16384, 4), (32768, 4)):
        freqs = sample_frequencies(d, 3, rho, np.random.default_rng(42))
        errs = []
        for local, pts in cells:
            truth = eval_scene_exact(local, pts)[:, 0]
            norm = np.linalg.norm(truth)
            if norm == 0:
                continue
            b = spectral_bundle(local, freqs)[0].astype(np.complex64)
            buf = pack_polar(b, bits=bits)
            est = decode_field(unpack(buf)[None, :], freqs, rho, pts)[:, 0]
            errs.append(float(np.linalg.norm(est - truth) / norm))
            nbytes = len(buf)
        out["d=%d @ %d bit" % (d, bits)] = {
            "packed_bytes": nbytes,
            "median_rel_err": float(np.median(errs)),
        }
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def done_ids(path=None):
    # resolved at CALL time: binding RESULTS as a default argument makes
    # every later override of it silently ineffective, which is how the
    # resume test caught this
    path = path or RESULTS
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    continue
    return ids


def wait_for_headroom(need_gb, poll=HEADROOM_POLL_S, limit=HEADROOM_LIMIT_S):
    """Block until the machine has room, rather than failing.

    A refusal here means someone else's training run is using the box.
    That is not an error and the right response is patience, not death —
    but it IS bounded, so a wedged machine does not hold the battery
    open forever.
    """
    waited = 0
    while True:
        try:
            return budget.require_headroom(need_gb)
        except MemoryError:
            if waited >= limit:
                raise
            print("  no headroom yet (%ds/%ds waited); sleeping %ds"
                  % (waited, limit, poll), flush=True)
            time.sleep(poll)
            waited += poll


def record(exp, result, wall, peak, path=None):
    """Append one result line.

    An id that is not in the REGISTRY cannot be written to the DEFAULT
    results file. That is not defensive programming for its own sake:
    test fixtures called DONE, TODO and Z9 have twice ended up in the
    real out/precision/battery.jsonl looking exactly like measurements —
    once from the default-argument bug, once from deliberately
    sabotaging the code to prove a test caught it. Tests pass an
    explicit path and are unaffected.
    """
    path = path or RESULTS
    if path == _REAL_RESULTS and exp.id not in {e.id for e in REGISTRY}:
        raise ValueError(
            "refusing to write unregistered id %r into the real results "
            "file; point RESULTS elsewhere for test fixtures" % exp.id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"id": exp.id, "phase": exp.phase, "summary": exp.summary,
           "wall_s": round(wall, 1), "peak_rss_gb": round(peak, 2),
           "result": result}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def run(selected, resume, dry_run):
    already = done_ids() if resume else set()
    todo = [e for e in selected if e.id not in already]
    skipped = sorted(already & {e.id for e in selected})
    if skipped:
        print("resume: skipping %s" % ", ".join(skipped))
    print("%d experiment(s), ~%d min, one at a time"
          % (len(todo), sum(e.minutes for e in todo)), flush=True)
    for exp in todo:
        print("\n=== %s  %s" % (exp.id, exp.summary), flush=True)
        if dry_run:
            print("  (dry run: needs ~%.1f GB, ~%d min)"
                  % (exp.need_gb, exp.minutes))
            continue
        wait_for_headroom(exp.need_gb)
        t0 = time.time()
        result = exp.fn()
        row = record(exp, result, time.time() - t0, budget.peak_rss_gb())
        print("  done in %.0fs, peak %.2f GB" % (row["wall_s"],
                                                 row["peak_rss_gb"]),
              flush=True)
        rows = sorted(result.items())
        for k, v in rows[:8]:
            print("    %-28s %s" % (k, v), flush=True)
        if len(rows) > 8:
            # the JSONL has all of them; only the console is trimmed
            print("    ... and %d more (--report)" % (len(rows) - 8),
                  flush=True)


def report(path=RESULTS):
    if not os.path.exists(path):
        print("no results at %s" % path)
        return
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    for row in rows:
        print("\n## %s — %s  (%.0fs, %.2f GB peak)"
              % (row["id"], row["summary"], row["wall_s"], row["peak_rss_gb"]))
        for k, v in sorted(row["result"].items()):
            print("  %-34s %s" % (k, v))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--phase", action="append", default=[])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--capture", help="absolute path to the capture file; "
                                      "data/ is gitignored so a worktree "
                                      "has none")
    args = ap.parse_args(argv)
    if args.capture:
        set_capture(args.capture)

    if args.list:
        for e in REGISTRY:
            print("%-4s %-6s ~%4.1f GB ~%3d min  %s"
                  % (e.id, "phase " + e.phase, e.need_gb, e.minutes,
                     e.summary))
        return 0
    if args.report:
        report()
        return 0
    sel = REGISTRY
    if args.only:
        sel = [e for e in sel if e.id in args.only]
    if args.phase:
        sel = [e for e in sel if e.phase in args.phase]
    if not sel:
        print("nothing selected; --list shows the registry")
        return 1
    run(sel, args.resume, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
