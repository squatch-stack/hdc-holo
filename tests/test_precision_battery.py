"""The battery's plumbing, which is the part a crash exercises.

The experiments themselves need a real capture and minutes of compute,
so they are not unit-tested here. What IS tested is everything that
decides whether an interrupted run can be picked up again — because the
reason this runner exists is that four measurement runs died in one
evening on a shared machine, and a resume that silently re-runs
everything would put the next one straight back into the OOM killer.
"""
import json

import numpy as np
import pytest

from bench import precision_battery as pb


def test_done_ids_survives_a_truncated_line(tmp_path):
    # a kill mid-write leaves a partial line; it must not poison resume
    path = tmp_path / "battery.jsonl"
    path.write_text('{"id": "A1"}\n{"id": "B1"}\n{"id": "C1", "resu\n')
    assert pb.done_ids(str(path)) == {"A1", "B1"}


def test_done_ids_on_a_missing_file_is_empty(tmp_path):
    assert pb.done_ids(str(tmp_path / "nope.jsonl")) == set()


def test_record_writes_one_line_with_the_reproducibility_fields(tmp_path):
    path = tmp_path / "out" / "battery.jsonl"
    exp = pb.Experiment("X1", "X", 1.0, 1, "summary", lambda: {})
    pb.record(exp, {"k": 1.0}, wall=2.5, peak=3.25, path=str(path))
    row = json.loads(path.read_text().strip())
    # wall and peak are not decoration: a number measured on a contended
    # box is not reproducible without them
    assert row["id"] == "X1" and row["result"] == {"k": 1.0}
    assert row["wall_s"] == 2.5 and row["peak_rss_gb"] == 3.25


def test_resume_skips_recorded_ids_and_runs_the_rest(tmp_path, monkeypatch):
    path = tmp_path / "battery.jsonl"
    path.write_text('{"id": "DONE"}\n')
    monkeypatch.setattr(pb, "RESULTS", str(path))
    def enough(*_args, **_kw):
        return 99.0
    monkeypatch.setattr(pb, "wait_for_headroom", enough)
    monkeypatch.setattr(pb.budget, "peak_rss_gb", lambda: 1.0)
    ran = []
    sel = [pb.Experiment("DONE", "X", 1.0, 1, "s", lambda: ran.append("DONE")
                         or {}),
           pb.Experiment("TODO", "X", 1.0, 1, "s", lambda: ran.append("TODO")
                         or {})]
    pb.run(sel, resume=True, dry_run=False)
    assert ran == ["TODO"]


def test_dry_run_touches_neither_headroom_nor_the_experiment(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("dry run must not run anything")
    monkeypatch.setattr(pb, "wait_for_headroom", boom)
    pb.run([pb.Experiment("X", "X", 1.0, 1, "s", boom)], resume=False,
           dry_run=True)


def test_wait_for_headroom_gives_up_rather_than_hanging(monkeypatch):
    calls = []

    def never(_need):
        calls.append(1)
        raise MemoryError("busy")
    monkeypatch.setattr(pb.budget, "require_headroom", never)
    monkeypatch.setattr(pb.time, "sleep", lambda _s: None)
    with pytest.raises(MemoryError):
        pb.wait_for_headroom(1.0, poll=1, limit=3)
    assert len(calls) > 1        # it waited, rather than failing at once


def test_complex32_is_lossy_but_faithful():
    rng = np.random.default_rng(0)
    v = (rng.standard_normal(512) + 1j * rng.standard_normal(512)
         ).astype(np.complex64)
    q = pb.as_complex32(v)
    err = np.linalg.norm(q - v) / np.linalg.norm(v)
    assert 0 < err < 1e-2        # fp16 mantissa, ~3 decimal digits


def test_bfp16_with_one_block_is_a_single_shared_scale():
    """block=len(v) is exactly pack_polar's `scale = m.max()` strategy,
    which is why the sweep varies block size rather than asking whether
    scaling helps at all."""
    rng = np.random.default_rng(1)
    v = (rng.standard_normal(256) + 1j * rng.standard_normal(256)
         ).astype(np.complex64)
    whole = pb.as_bfp16(v, len(v))
    fine = pb.as_bfp16(v, 32)
    assert len(whole) == len(v) and len(fine) == len(v)
    # smaller blocks track local magnitude, so they cannot be worse
    assert (np.linalg.norm(fine - v) <= np.linalg.norm(whole - v) * 1.05)


def test_bfp16_handles_a_length_that_is_not_a_multiple_of_the_block():
    v = np.ones(70, dtype=np.complex64)
    assert len(pb.as_bfp16(v, 64)) == 70


def test_results_path_is_resolved_at_call_time(tmp_path, monkeypatch):
    """Regression: RESULTS was bound as a DEFAULT ARGUMENT.

    `def record(..., path=RESULTS)` freezes the module global at import,
    so every later override of it is silently ignored — which is not a
    theoretical bug: it wrote this test's own fixture rows into the real
    out/precision/battery.jsonl, where they sat looking like results.
    Both record() and done_ids() must read RESULTS when called.
    """
    real = tmp_path / "real.jsonl"
    monkeypatch.setattr(pb, "RESULTS", str(real))
    exp = pb.Experiment("Z9", "Z", 1.0, 1, "s", lambda: {})
    pb.record(exp, {"v": 1}, wall=1.0, peak=1.0)
    assert real.exists() and pb.done_ids() == {"Z9"}

    moved = tmp_path / "moved.jsonl"
    monkeypatch.setattr(pb, "RESULTS", str(moved))
    assert pb.done_ids() == set()        # follows the override, not the import


def test_the_real_results_file_rejects_unregistered_ids():
    """Test fixtures have twice landed in out/precision/battery.jsonl
    looking like measurements — once via the default-argument bug above,
    once from deliberately breaking the code to prove that test caught
    it. Only registered experiment ids may write to the real path."""
    bogus = pb.Experiment("BOGUS", "X", 1.0, 1, "s", lambda: {})
    with pytest.raises(ValueError, match="unregistered"):
        pb.record(bogus, {}, wall=1.0, peak=1.0)


def test_registered_ids_are_unique_and_the_registry_is_reachable():
    ids = [e.id for e in pb.REGISTRY]
    assert ids and len(ids) == len(set(ids))
    assert {"A1", "A2", "B1", "B2", "C1", "C2", "D1"} <= set(ids)
