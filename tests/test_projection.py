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
import os

import numpy as np
import pytest

from examples import run_projection_pipeline as driver
from holo import projection as rp
from holo.capture import (
    BANDS,
    band_codebooks,
    decode_slice,
    encode_bands,
    exact_slice,
)
from holo.spectral import SplatScene

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 384


def _readable_gram(fd, s):
    """The expression build_gram replaces, verbatim."""
    sq = (fd ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (fd @ fd.T)
    np.maximum(d2, 0.0, out=d2)
    return (2 * np.pi * s ** 2) ** 1.5 * np.exp(-0.5 * (s ** 2) * d2)


@pytest.mark.parametrize("band", [b[0] for b in BANDS])
def test_gram_is_bit_identical_to_the_expression_it_replaces(band):
    name, _cap, cell = next(b for b in BANDS if b[0] == band)
    fd = band_codebooks(np.random.default_rng(42))[name][0][:D].astype(
        np.float64)
    s = (cell / 2) / 2
    assert np.array_equal(_readable_gram(fd, s), rp.build_gram(fd, s))


def test_operators_are_bit_identical_and_survive_a_mixed_sweep():
    """Tikhonov writes lambda into the Gram's diagonal in place. A sweep
    that interleaves it with TSVD must therefore restore the diagonal,
    or the second setting silently solves a different problem — so the
    order below deliberately alternates.

    It also pins that adding `eps` did not move `keep`. The two share
    one eigendecomposition and one operator expression and differ only
    in how many columns they take, which is what lets every keep=
    number already in docs/fit.md stand without re-deriving it.
    """
    name, _cap, cell = BANDS[0]
    fd = band_codebooks(np.random.default_rng(42))[name][0][:D].astype(
        np.float64)
    s = (cell / 2) / 2
    G0 = _readable_gram(fd, s)
    solver = rp.BandSolver(rp.build_gram(fd, s))
    for kind, val in (("tikhonov", 1e-3), ("keep", 0.25), ("eps", 1e-3),
                      ("tikhonov", 1e-1), ("keep", 0.55)):
        if kind in ("keep", "eps"):
            ev, vec = np.linalg.eigh(G0)
            order = np.argsort(np.abs(ev))[::-1]
            ev, vec = ev[order], vec[:, order]
            k = (max(1, round(val * len(ev))) if kind == "keep"
                 else max(1, int((np.abs(ev) > val * np.abs(ev[0])).sum())))
            want = (vec[:, :k] / ev[:k][None, :]) @ vec[:, :k].T
        else:
            lam = val * float(np.abs(G0).max())
            want = np.linalg.inv(G0 + lam * np.eye(len(G0)))
        got, _how = solver.operator((kind, val))
        assert np.array_equal(want, got), "%s=%g diverged" % (kind, val)


def test_chunk_size_falls_out_of_a_byte_budget():
    # the hand-tuned 256 this replaced was measured at 4 channels, d=8192
    assert rp.cell_chunk(4, 8192) == 256
    # a wider capture takes smaller chunks rather than more memory
    assert rp.cell_chunk(8, 8192) < rp.cell_chunk(4, 8192)
    assert rp.cell_chunk(1, 8192) > rp.cell_chunk(4, 8192)


def test_sweep_parsing_keeps_the_old_forms_working():
    assert driver.parse_settings(["s.spz", "0.25"]) == ("s.spz", [("keep", 0.25)])
    assert driver.parse_settings(["s.spz", "--tikhonov", "1e-6"]) == (
        "s.spz", [("tikhonov", 1e-6)])
    assert driver.parse_settings(["s.ply", "--sweep", "eps=1e-4,1e-5"]) == (
        "s.ply", [("eps", 1e-4), ("eps", 1e-5)])
    assert driver.label_of(("eps", 1e-4)) == "eps=1e-04"
    assert driver.label_of(("tikhonov", 1e-6)) == "tikhonov=1e-06"
    path, settings = driver.parse_settings(
        ["s.spz", "--sweep", "tikhonov=1e-6,1e-3", "--sweep", "keep=0.25"])
    assert path == "s.spz"
    assert settings == [("tikhonov", 1e-6), ("tikhonov", 1e-3), ("keep", 0.25)]


def test_divergence_is_detectable_in_the_norm_before_anything_decodes():
    """Truncation is a knife edge and its failure is silent in the decode.

    Measured on saguaro's xfine band, solved-to-forward norm ratio against
    what the slice error then did:

        keep 0.10-0.40   ratio 1.1-1.3    works
        keep 0.55        ratio 5.3        BEST measured, +59.3%
        keep 0.70        ratio 97.4       -1417%, i.e. 37x worse than
                                          not projecting at all
        keep 1.00        ratio 5.2e11     -71,836,511%

    The historical catastrophe gap missed minority-band degradation.
    The promoted gate instead separates the measured fine-band knee.
    """
    rng = np.random.default_rng(0)
    forward = {k: (rng.standard_normal((1, 64))
                   + 1j * rng.standard_normal((1, 64))).astype(np.complex64)
               for k in ("a", "b", "c")}
    healthy = {k: v * 1.09 for k, v in forward.items()}
    diverged = {k: v * 500.0 for k, v in forward.items()}

    assert rp.divergence_ratio(healthy, forward) < rp.DIVERGENCE_RATIO
    assert rp.divergence_ratio(diverged, forward) > rp.DIVERGENCE_RATIO
    # The tighter gate catches degradation before catastrophe.
    assert 1.15 < rp.DIVERGENCE_RATIO < 1.32


def test_divergence_ratio_is_none_when_there_is_nothing_to_compare():
    assert rp.divergence_ratio({}, {}) is None
    assert rp.divergence_ratio({"a": np.ones((1, 4), np.complex64)}, {}) is None


def test_a_diverged_band_is_refused_not_warned_about():
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


def test_the_gate_names_the_way_out():
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


def test_a_deliberate_sweep_can_open_the_gate(capsys):
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


def test_a_clean_band_passes_silently(capsys):
    rng = np.random.default_rng(3)
    forward = {"a": (rng.standard_normal((1, 32))
                     + 1j * rng.standard_normal((1, 32))).astype(np.complex64)}
    healthy = {"a": forward["a"] * 1.09}
    assert rp.check_divergence(healthy, forward, "fine", "keep=0.25") < 2
    assert capsys.readouterr().out == ""



# ---------------------------------------------------------------------------
# Per-band scoring — the referee Red Rock showed was missing
# ---------------------------------------------------------------------------

def _lopsided_scene(rng, big=400, small=6):
    """A scene shaped like a real capture: one band holds nearly all the
    splats and another holds a handful (Red Rock is 542,122 against
    3,854 — 0.7%). s = 0.003 lands in `xfine`, s = 0.006 in `fine`.
    """
    n = big + small
    mu = rng.uniform(0.35, 0.65, (n, 3)).astype(np.float32)
    s = np.concatenate([np.full(big, 0.003),
                        np.full(small, 0.006)]).astype(np.float32)
    cov = np.stack([np.eye(3) * si ** 2 for si in s]).astype(np.float32)
    return SplatScene(mu=mu, cov=cov, amp=np.ones((n, 1), np.float32)), s


@pytest.fixture(scope="module")
def lopsided():
    """Encoded scene, probe points, and both referees. d=1024: the
    partition and isolation properties under test do not depend on d.
    """
    rng = np.random.default_rng(3)
    scene, smax = _lopsided_scene(rng)
    books = band_codebooks(np.random.default_rng(2), dim=1024)
    bundles, members = encode_bands(scene, smax, books, dim=1024,
                                    verbose=False)
    pts = (scene.mu[rng.integers(0, scene.n, 400)]
           + 0.004 * rng.standard_normal((400, 3))).astype(np.float32)
    return {
        "books": books,
        "bundles": bundles,
        "pts": pts,
        "slices": [("probe", (pts, None))],
        "band_truth": {"probe": {b[0]: exact_slice(pts, scene, members,
                                                   bands=[b])
                                 for b in BANDS}},
        "truth": exact_slice(pts, scene, members),
    }


def _per_band(bundles, sc):
    return rp.band_errors(bundles, sc["books"], sc["slices"], sc["band_truth"])


def _aggregate(bundles, sc):
    return rp.rel_err(decode_slice(sc["pts"], bundles, sc["books"])[:, 0],
                      sc["truth"][:, 0])


def test_per_band_truth_partitions_the_aggregate_exactly(lopsided):
    """Why per-band scoring is free rather than four times the cost.

    The bands partition the splats, so four single-band referee passes
    touch the same splats one full pass does — and their sum IS the
    aggregate referee, to the bit. If that stops holding, per-band error
    has stopped being a decomposition of the number the repo reports.
    """
    per = lopsided["band_truth"]["probe"]
    assert np.array_equal(sum(per[b[0]] for b in BANDS), lopsided["truth"])


def test_a_destroyed_band_is_visible_per_band_and_diluted_in_the_aggregate(
        lopsided):
    """The hole Red Rock exposed, in miniature.

    Corrupting one band's bundles moves that band's own score by the
    corruption, leaves every other band's score BIT-identical, and moves
    the aggregate by strictly less — the aggregate is a blend weighted
    by where the field actually is.

    How much less is scene geometry, and this toy is not a model of Red
    Rock's ratio: there, at keep=0.55, `fine` ran at 1030x the forward
    norm while the aggregate slice error IMPROVED. What is pinned here
    is the property the code guarantees — detection, isolation, and
    dilution — which is why a per-band number has to be reported beside
    the aggregate rather than instead of it.
    """
    clean = _per_band(lopsided["bundles"], lopsided)
    assert clean["fine"][0] < 0.5 and clean["xfine"][0] < 0.5

    broken = {k: ({c: b * 1000.0 for c, b in v.items()} if k == "fine" else v)
              for k, v in lopsided["bundles"].items()}
    got = _per_band(broken, lopsided)

    assert got["fine"][0] > 100                    # detected, in proportion
    assert got["xfine"] == clean["xfine"]          # and it did not smear
    assert _aggregate(broken, lopsided) < got["fine"][0] / 2


def test_a_band_with_no_signal_scores_none_rather_than_zero():
    """A band with no splats near a slice has an ABSENT measurement, not
    a perfect one. Reporting 0.0 there would read as the best band in
    the table."""
    want = np.zeros(8, dtype=np.float32)
    assert rp.rel_err(np.ones(8, np.float32), want) is None
    assert rp.rel_err(want, np.ones(8, np.float32)) == 1.0


def test_eps_truncates_the_ill_conditioned_band_harder():
    """The reason to change the knob at all.

    One `keep` is one rank for every band, whatever their spectra do.
    One `eps` is one regularisation LEVEL, so the band whose Gram decays
    fastest keeps fewer of its eigenvalues — automatically, and without
    a per-band constant to tune. `fine` is that band, and `fine` is the
    one that ran at 1030x on Red Rock.

    Pinned as an ordering rather than a value: the ranks themselves move
    with d (at d=8192 the shipped keep=0.25 corresponds to 2.56e-04 on
    `xfine` and 1.77e-05 on `fine`), but which band needs more damping
    is a property of the band, not of d.
    """
    books = band_codebooks(np.random.default_rng(42))
    ranks = {}
    for name, _cap, cell in BANDS:
        fd = books[name][0][:D].astype(np.float64)
        ev, _ = rp.eigen(rp.build_gram(fd, (cell / 2) / 2))
        ranks[name] = {
            "eps": rp.BandSolver.rank(ev, "eps", 1e-2),
            "keep": rp.BandSolver.rank(ev, "keep", 0.25),
        }
    assert ranks["fine"]["eps"] < ranks["xfine"]["eps"]
    assert ranks["fine"]["eps"] < ranks["coarse"]["eps"]
    # ...where a rank fraction is by construction blind to all of that
    assert len({r["keep"] for r in ranks.values()}) == 1


def test_a_faint_band_is_absent_not_catastrophic(lopsided):
    """The saguaro `coarse` bug: dividing by nearly nothing.

    `rel_err` used to guard only an EXACTLY zero referee. A band with a
    few distant splats gives a nearly zero one, and the quotient looks
    like catastrophic failure while actually being an absence of signal.
    Saguaro's `coarse` band reported 4,359,160 on the top-down slice.
    That is not a band that failed; it is a band that is not there.

    Pinned both ways round, because a floor that swallowed real bands
    would be worse than the bug it fixes: the same call declines the
    faint band and scores the loud one normally.
    """
    sc = lopsided
    band_truth = {n: dict(per, fine=per["fine"] * 1e-6)   # a millionth
                  for n, per in sc["band_truth"].items()}

    got = rp.band_errors(sc["bundles"], sc["books"], sc["slices"], band_truth)
    assert got["fine"][0] is None              # declined, not 4e6
    assert got["xfine"][0] is not None         # and nothing else was taken
    assert got["xfine"][0] < 0.5

    # the same comparison with no floor is the number that misled: huge,
    # four significant figures, and meaningless
    one = [b for b in BANDS if b[0] == "fine"]
    unfloored = rp.rel_err(
        decode_slice(sc["pts"], sc["bundles"], sc["books"], bands=one)[:, 0],
        band_truth["probe"]["fine"][:, 0])
    assert unfloored > 1e3


def test_default_limit_refuses_a_diverging_band():
    forward = {0: np.ones((2, 32), np.complex64)}
    with pytest.raises(rp.Diverged, match=r"1.320.*limit 1.200"):
        rp.check_divergence({0: forward[0] * 1.32}, forward, "fine", "eps")
    ratio = rp.check_divergence({0: forward[0] * 1.09}, forward, "fine", "eps")
    np.testing.assert_allclose(ratio, 1.09, rtol=1e-6)


def test_rank_fractions_match_the_documented_spectrum():
    expected = {"xfine": 0.186, "fine": 0.103, "mid": 0.509, "coarse": 0.593}
    with np.load(os.path.join(ROOT, "out", "gram_spectrum_d8192.npz")) as spectra:
        for band, fraction in expected.items():
            spectrum = spectra[band]
            got = rp.BandSolver.rank(spectrum, "eps", 1e-3) / len(spectrum)
            # The published table rounds exact integer ranks to three places.
            assert round(got, 3) == fraction


def test_a_forward_failed_band_is_excluded_from_the_gate(monkeypatch):
    scene, scales, _ = driver.synthetic_scene()
    bands = [BANDS[2]]
    books = band_codebooks(np.random.default_rng(42), bands, dim=32)
    forward, members = encode_bands(scene, scales, books, bands,
                                     dim=32, verbose=False)
    calls = []

    def measured_errors(*args, **kwargs):
        calls.append(True)
        return {"mid": [1.4294, 1.0726]}

    def broken_solve(*args, **kwargs):
        return {k: v * 100 for k, v in forward["mid"].items()}

    def probe_planes(scene):
        return [("probe", (scene.mu, None))]

    monkeypatch.setattr(rp, "evidence_slices", probe_planes)
    monkeypatch.setattr(rp, "band_errors", measured_errors)
    monkeypatch.setattr(rp, "solve_band", broken_solve)
    solved = rp.project_cells(scene, members, books, bands)
    assert calls
    assert rp.divergence_ratio(solved["mid"], forward["mid"]) > 99
    # The same proposed bundles must be refused when forward is viable.
    with pytest.raises(rp.Diverged):
        rp.check_divergence(solved["mid"], forward["mid"], "mid", "eps",
                            forward_errors=[0.9, None])
    assert not rp.gate_eligible([None, None])
    assert not rp.gate_eligible([1.0, 0.5])
    assert rp.gate_eligible([0.5, None])


def test_project_cells_preserves_the_legacy_solve_and_empty_bands(monkeypatch):
    scene, scales, _ = driver.synthetic_scene()
    books = band_codebooks(np.random.default_rng(42), dim=32)
    _, members = encode_bands(scene, scales, books, dim=32, verbose=False)

    def probe_planes(scene):
        return [("probe", (scene.mu, None))]

    monkeypatch.setattr(rp, "evidence_slices", probe_planes)
    got = rp.project_cells(scene, members, books, setting=("keep", 0.25),
                           allow_divergence=True)
    for name, _cap, cell in BANDS:
        if not members[name]:
            assert got[name] == {}
            continue
        freqs, _, weights = books[name]
        fd = freqs.astype(np.float64)
        s = (cell / 2) / 2
        ev, vec = rp.eigen(_readable_gram(fd, s))
        rank = max(1, round(0.25 * len(ev)))
        operator = (vec[:, :rank] / ev[:rank][None, :]) @ vec[:, :rank].T
        expected = rp.solve_band(operator, scene, members[name],
                                 rp.BandGeom(cell, s, freqs, fd, weights),
                                 rp.cell_chunk(scene.channels, len(freqs)))
        assert got[name].keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(got[name][key], expected[key])
            assert got[name][key].dtype == np.complex64


@pytest.mark.parametrize("setting", [("typo", 0.1), ("eps", 0), ("keep", 2)])
def test_project_cells_rejects_invalid_settings(setting):
    scene, _, _ = driver.synthetic_scene()
    with pytest.raises(ValueError):
        rp.project_cells(scene, {}, {}, setting=setting)


def test_project_cells_beats_forward_on_a_sparse_synthetic_scene(monkeypatch):
    rng = np.random.default_rng(42)
    bands = [("fine", 0.008, 1 / 32)]
    mu = rng.uniform(0.011, 0.020, (3, 3)).astype(np.float32)
    scene = SplatScene(
        mu, np.broadcast_to(np.eye(3) * 0.006**2, (3, 3, 3)).astype(np.float32),
        np.ones((3, 1), np.float32))
    books = band_codebooks(np.random.default_rng(42), bands, dim=384)
    forward, members = encode_bands(scene, np.full(3, 0.006), books, bands,
                                     dim=384, verbose=False)
    pts = rng.uniform(0.006, 0.025, (256, 3)).astype(np.float32)

    def probe_planes(scene):
        # Bound CI cost; retain real exact-mixture and forward gate calculations.
        return [("probe", (pts, None))]

    monkeypatch.setattr(rp, "evidence_slices", probe_planes)
    solved = rp.project_cells(scene, members, books, bands)
    truth = exact_slice(pts, scene, members, bands)
    before = rp.rel_err(decode_slice(pts, forward, books, bands), truth)
    after = rp.rel_err(decode_slice(pts, solved, books, bands), truth)
    # Interior smooth splats: measured errors 0.10579 -> 0.01002 give
    # after/before = 0.0947. Relaxing that by more than a factor of five
    # gives the 0.5 margin, leaving room for numerical variation while
    # requiring a material gain. This fixture is not an accuracy guarantee.
    assert after < 0.5 * before, (before, after)
