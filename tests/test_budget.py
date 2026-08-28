"""The headroom guard, whose whole job is to be right about units.

`ru_maxrss` is bytes on macOS and kilobytes on Linux; a 1024x error
there reads as "0.004 GB peak" or "45 GB available" and would make the
guard useless in opposite directions on the two platforms. So the unit
is checked against an allocation of known size rather than asserted.
"""
import numpy as np
import pytest

from holo import budget


def test_peak_rss_is_in_gb_not_kb_or_bytes():
    hold = np.ones(48 << 20, dtype=np.uint8)      # 48 MB, touched
    hold[::4096] = 7
    peak = budget.peak_rss_gb()
    del hold
    # A python process holding 48 MB of numpy sits well above 20 MB and
    # nowhere near 64 GB. Both wrong units land outside this window:
    # bytes-as-KB would report ~1e-5, KB-as-bytes ~200.
    assert 0.02 < peak < 64.0


def test_available_is_plausible_or_honestly_unknown():
    avail = budget.available_gb()
    assert avail is None or 0.0 <= avail < 1e5


def test_heavy_processes_shape():
    heavy = budget.heavy_processes(threshold_gb=0.0)
    if heavy is None:
        pytest.skip("ps unreadable on this platform")
    assert heavy == sorted(heavy, reverse=True)
    for gb, pid, cmd in heavy:
        assert isinstance(gb, float) and isinstance(pid, int)
        assert isinstance(cmd, str) and len(cmd) <= budget._ARGV_CHARS
    # the guard must never veto a run because of its own parent
    assert all(pid not in budget._ancestors() for _, pid, _ in heavy)


def test_impossible_need_is_refused_and_says_why():
    if budget.available_gb() is None:
        pytest.skip("no memory probe on this platform")
    with pytest.raises(MemoryError) as exc:
        budget.require_headroom(1e6)
    msg = str(exc.value)
    assert "1000000.0 GB" in msg          # the need, echoed back
    assert "force=True" in msg            # and the way past it


def test_force_proceeds_past_the_refusal():
    # deliberately impossible, but forced: returns instead of raising,
    # so the bare call IS the assertion
    budget.require_headroom(1e6, force=True)


def test_short_argv_keeps_the_informative_half():
    cmd = ("/Applications/Xcode.app/Contents/Developer/Library/Frameworks/"
           "Python3.framework/Versions/3.9/Resources/Python.app/Contents/"
           "MacOS/Python -m examples.run_projection_pipeline data/x.spz 0.25")
    short = budget._short_argv(cmd)
    assert short.startswith("Python -m examples.run_projection_pipeline")
    assert len(short) <= budget._ARGV_CHARS


def test_budget_is_not_on_the_public_surface():
    import holo
    # a developer utility, deliberately not exported: it shells out to
    # `ps` and reads /proc, which is not library behaviour
    assert "budget" not in getattr(holo, "__all__", [])
    assert not hasattr(holo, "require_headroom")
