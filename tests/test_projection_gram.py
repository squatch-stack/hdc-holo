"""The projection pipeline's Gram build must be BIT-identical, not close.

`build_gram` exists to hold two 537 MB arrays where the readable
expression holds three and allocates six. The tempting version folds the
gemm into a single buffer, which reassociates (a + b) - 2c into
(-2c + a) + b — one ulp on the Gram, 5e-15 relative, and it looks
harmless.

It is not harmless here. The window Gram is ill-conditioned by
construction (1.6e20 at d=8192, which is why truncation is mandatory),
and the truncated pseudo-inverse amplified that ulp to 2.8e-8 on the
operator at d=1024 alone. Every measured projection number in the repo
is quoted to four decimals against a specific operator, so this test
pins identity rather than tolerance: if it degrades to allclose, the
numbers have to be re-derived instead of carried forward.
"""
import importlib.util
import os

import numpy as np
import pytest

from holo.capture import BANDS, band_codebooks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(ROOT, "examples", "run_projection_pipeline.py")

D = 384          # the reassociation shows at any d; keep the suite fast


@pytest.fixture(scope="module")
def rp():
    spec = importlib.util.spec_from_file_location("_rp", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _readable_gram(fd, s):
    """The expression build_gram replaces, verbatim."""
    sq = (fd ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (fd @ fd.T)
    np.maximum(d2, 0.0, out=d2)
    return (2 * np.pi * s ** 2) ** 1.5 * np.exp(-0.5 * (s ** 2) * d2)


@pytest.mark.parametrize("band", [b[0] for b in BANDS])
def test_gram_is_bit_identical_to_the_expression_it_replaces(rp, band):
    name, _cap, cell = next(b for b in BANDS if b[0] == band)
    fd = band_codebooks(np.random.default_rng(42))[name][0][:D].astype(
        np.float64)
    s = (cell / 2) / 2
    assert np.array_equal(_readable_gram(fd, s), rp.build_gram(fd, s))


def test_operators_are_bit_identical_and_survive_a_mixed_sweep(rp):
    """Tikhonov writes lambda into the Gram's diagonal in place. A sweep
    that interleaves it with TSVD must therefore restore the diagonal,
    or the second setting silently solves a different problem — so the
    order below deliberately goes Tikhonov, TSVD, Tikhonov.
    """
    name, _cap, cell = BANDS[0]
    fd = band_codebooks(np.random.default_rng(42))[name][0][:D].astype(
        np.float64)
    s = (cell / 2) / 2
    G0 = _readable_gram(fd, s)
    solver = rp.BandSolver(rp.build_gram(fd, s))
    for kind, val in (("tikhonov", 1e-3), ("keep", 0.25), ("tikhonov", 1e-1)):
        if kind == "keep":
            ev, vec = np.linalg.eigh(G0)
            order = np.argsort(np.abs(ev))[::-1]
            ev, vec = ev[order], vec[:, order]
            k = max(1, round(val * len(ev)))
            want = (vec[:, :k] / ev[:k][None, :]) @ vec[:, :k].T
        else:
            lam = val * float(np.abs(G0).max())
            want = np.linalg.inv(G0 + lam * np.eye(len(G0)))
        got, _how = solver.operator((kind, val))
        assert np.array_equal(want, got), "%s=%g diverged" % (kind, val)


def test_chunk_size_falls_out_of_a_byte_budget(rp):
    # the hand-tuned 256 this replaced was measured at 4 channels, d=8192
    assert rp.cell_chunk(4, 8192) == 256
    # a wider capture takes smaller chunks rather than more memory
    assert rp.cell_chunk(8, 8192) < rp.cell_chunk(4, 8192)
    assert rp.cell_chunk(1, 8192) > rp.cell_chunk(4, 8192)


def test_sweep_parsing_keeps_the_old_forms_working(rp):
    assert rp.parse_settings(["s.spz", "0.25"]) == ("s.spz", [("keep", 0.25)])
    assert rp.parse_settings(["s.spz", "--tikhonov", "1e-6"]) == (
        "s.spz", [("tikhonov", 1e-6)])
    path, settings = rp.parse_settings(
        ["s.spz", "--sweep", "tikhonov=1e-6,1e-3", "--sweep", "keep=0.25"])
    assert path == "s.spz"
    assert settings == [("tikhonov", 1e-6), ("tikhonov", 1e-3), ("keep", 0.25)]


def test_divergence_is_detectable_in_the_norm_before_anything_decodes(rp):
    """Truncation is a knife edge and its failure is silent in the decode.

    Measured on saguaro's xfine band, solved-to-forward norm ratio against
    what the slice error then did:

        keep 0.10-0.40   ratio 1.1-1.3    works
        keep 0.55        ratio 5.3        BEST measured, +59.3%
        keep 0.70        ratio 97.4       -1417%, i.e. 37x worse than
                                          not projecting at all
        keep 1.00        ratio 5.2e11     -71,836,511%

    Nothing good exceeds 5.3 and nothing broken comes below 97, so the
    threshold sits in an 18x gap. This pins the separation rather than
    the exact numbers.
    """
    rng = np.random.default_rng(0)
    forward = {k: (rng.standard_normal((1, 64))
                   + 1j * rng.standard_normal((1, 64))).astype(np.complex64)
               for k in ("a", "b", "c")}
    healthy = {k: v * 1.2 for k, v in forward.items()}
    diverged = {k: v * 500.0 for k, v in forward.items()}

    assert rp.divergence_ratio(healthy, forward) < rp.DIVERGENCE_RATIO
    assert rp.divergence_ratio(diverged, forward) > rp.DIVERGENCE_RATIO
    # and the threshold sits inside the measured gap, not on its edge
    assert 5.3 < rp.DIVERGENCE_RATIO < 97.0


def test_divergence_ratio_is_none_when_there_is_nothing_to_compare(rp):
    assert rp.divergence_ratio({}, {}) is None
    assert rp.divergence_ratio({"a": np.ones((1, 4), np.complex64)}, {}) is None


def test_a_diverged_band_is_refused_not_warned_about(rp):
    """The gate, and the reason it is a gate.

    It used to print and carry on. On Red Rock at keep=0.55 every cell of
    the `fine` band sits at 1030x the forward bundle norm while the SLICE
    ERROR IMPROVES — that band holds 0.7% of the splats, so destroying it
    barely moves the metric. A caller reading those bundles for a render
    or a `what_is_at` query gets garbage, and the number they were shown
    said it was the best of four settings. A warning on stdout is not a
    defence against that.
    """
    rng = np.random.default_rng(0)
    forward = {k: (rng.standard_normal((1, 32))
                   + 1j * rng.standard_normal((1, 32))).astype(np.complex64)
               for k in ("a", "b", "c")}
    diverged = {k: v * 500.0 for k, v in forward.items()}

    with pytest.raises(rp.Diverged, match="past the cliff"):
        rp.check_divergence(diverged, forward, "fine", "keep=0.55")


def test_the_gate_names_the_way_out(rp):
    """A refusal the caller cannot act on is only half a gate."""
    rng = np.random.default_rng(1)
    forward = {"a": (rng.standard_normal((1, 32))
                     + 1j * rng.standard_normal((1, 32))).astype(np.complex64)}
    diverged = {"a": forward["a"] * 500.0}
    with pytest.raises(rp.Diverged) as exc:
        rp.check_divergence(diverged, forward, "fine", "keep=0.55")
    message = str(exc.value)
    assert "--allow-divergence" in message      # the deliberate-sweep escape
    assert "smaller keep" in message            # and the actual fix
    assert "0.7%" in message or "small share" in message   # why the metric lies


def test_a_deliberate_sweep_can_open_the_gate(rp, capsys):
    """Sweeping past the cliff is how the cliff was found; the gate must
    not make that impossible, only deliberate."""
    rng = np.random.default_rng(2)
    forward = {"a": (rng.standard_normal((1, 32))
                     + 1j * rng.standard_normal((1, 32))).astype(np.complex64)}
    diverged = {"a": forward["a"] * 500.0}
    ratio = rp.check_divergence(diverged, forward, "fine", "keep=0.70",
                                allow=True)
    assert ratio > rp.DIVERGENCE_RATIO
    assert "diverged" in capsys.readouterr().out.lower()


def test_a_clean_band_passes_silently(rp, capsys):
    rng = np.random.default_rng(3)
    forward = {"a": (rng.standard_normal((1, 32))
                     + 1j * rng.standard_normal((1, 32))).astype(np.complex64)}
    healthy = {"a": forward["a"] * 1.2}
    assert rp.check_divergence(healthy, forward, "fine", "keep=0.25") < 2
    assert capsys.readouterr().out == ""
