"""Run records, and the one property that makes them worth writing.

A process killed by the OOM killer runs no `atexit` handler and no
`except` block, so it cannot write its own epitaph. That is why a run
writes TWO records — one at the start and one at the end — and why a
start with no end MEANS killed. Four runs were lost in one evening (two
OOM kills, a Metal command-buffer fault, a silent SIGKILL) and left
nothing behind; these tests pin the behaviour that would have caught
every one of them.
"""
import json
import os
import signal
import subprocess
import sys

import pytest

from holo import runlog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows(path):
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def test_a_completed_run_writes_a_start_and_an_end(tmp_path):
    path = str(tmp_path / "runs.jsonl")
    with runlog.record("unit", path=path) as run:
        run.stage("first", 1.5)
        run.result(err=0.217)
    rows = _rows(path)
    assert [r["event"] for r in rows] == ["started", "ended"]
    assert rows[0]["run"] == rows[1]["run"]
    assert rows[1]["status"] == "ok"
    assert rows[1]["results"] == {"err": 0.217}
    assert rows[1]["stages"][0]["name"] == "first"


def test_the_start_record_carries_what_reproduction_needs(tmp_path):
    """A number whose commit, dirtiness and backend are unknown is not
    reproducible, and a wall-clock without knowing what else held the
    machine is not comparable — that is why `contending` is captured."""
    path = str(tmp_path / "runs.jsonl")
    with runlog.record("unit", path=path):
        pass
    start = _rows(path)[0]
    for field in ("sha", "dirty", "backend", "argv", "machine", "at"):
        assert field in start, field


def test_an_exception_still_leaves_an_epitaph(tmp_path):
    path = str(tmp_path / "runs.jsonl")
    with pytest.raises(ZeroDivisionError), runlog.record("boom", path=path):
        raise ZeroDivisionError("nope")
    end = _rows(path)[1]
    assert end["status"] == "error" and "ZeroDivisionError" in end["error"]


def test_a_refused_run_is_distinguishable_from_a_vanished_one(tmp_path):
    """"It never started" and "it started and disappeared" are different
    diagnoses, so the headroom check runs AFTER the start record."""
    path = str(tmp_path / "runs.jsonl")
    with pytest.raises(MemoryError), runlog.record("greedy", need_gb=1e6,
                                                   path=path):
        raise AssertionError("body must not run")
    rows = _rows(path)
    assert [r["event"] for r in rows] == ["started", "ended"]
    assert rows[1]["status"] == "refused"


def test_a_sigkilled_run_is_reported_as_killed(tmp_path):
    """The whole design. SIGKILL is exactly what the OOM killer sends,
    and nothing in-process can catch it — so the evidence has to already
    be on disk before the run dies."""
    path = str(tmp_path / "runs.jsonl")
    code = (
        "import os, signal, sys\n"
        "sys.path.insert(0, %r)\n"
        "from holo import runlog\n"
        "with runlog.record('doomed', path=%r) as run:\n"
        "    run.stage('got this far', 1.0)\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n" % (ROOT, path)
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, check=False)
    assert proc.returncode == -signal.SIGKILL

    rows = _rows(path)
    assert [r["event"] for r in rows] == ["started"]      # no end record
    killed = [(s, e) for s, e in runlog.runs(str(tmp_path)) if e is None]
    assert len(killed) == 1 and killed[0][0]["label"] == "doomed"


def test_a_line_torn_by_a_kill_mid_write_does_not_poison_the_reader(tmp_path):
    path = tmp_path / "runs.jsonl"
    with runlog.record("unit", path=str(path)):
        pass
    path.write_text(path.read_text() + '{"run": "x", "event": "star')
    assert len(runlog.runs(str(tmp_path))) == 1           # the torn line is skipped


def test_summarize_flags_killed_runs(tmp_path, capsys):
    path = str(tmp_path / "runs.jsonl")
    with runlog.record("finished", path=path):
        pass
    runlog._append({"run": "ghost", "event": "started", "label": "vanished",
                    "at": "2026-01-01T00:00:00+00:00"}, path)
    runlog.summarize(str(tmp_path))
    out = capsys.readouterr().out
    assert "KILLED" in out and "vanished" in out


def test_runlog_is_not_on_the_public_surface():
    """Developer tooling, like holo.budget: importable as a submodule but
    never re-exported, because it shells out to git and ps.

    (`holo.record` is unrelated — it is the RecordSpace module, and the
    first version of this test asserted its absence by mistake.)"""
    import holo
    assert "runlog" not in getattr(holo, "__all__", [])
    for name in runlog.__all__:
        assert name not in getattr(holo, "__all__", []), name


# ---------------------------------------------------------------------------
# Where the records land — the reason the first ten runs left nothing
# ---------------------------------------------------------------------------

def _repo(path, name="main"):
    """A git repo with one commit, so it can carry a worktree."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "seed").write_text(name, encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "seed"], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.email=t@example.com",
                    "-c", "user.name=t", "commit", "-qm", "seed"], check=True)
    return path


@pytest.fixture
def fresh_run_dir(monkeypatch):
    """run_dir() is cached and reads the environment; reset both."""
    monkeypatch.delenv("HDC_RUN_DIR", raising=False)
    runlog.run_dir.cache_clear()
    yield
    runlog.run_dir.cache_clear()


def test_a_worktree_writes_its_records_to_the_shared_checkout(
        tmp_path, monkeypatch, fresh_run_dir):
    """The bug this fixes, and it destroyed evidence rather than raising.

    Every lane works in its own git worktree, so a package-relative
    RUN_DIR wrote the record INTO the worktree — and `git worktree
    remove` deleted it with the lane. A missing file is indistinguishable
    from a run nobody launched, which is why it went unnoticed until the
    shared checkout was found to have no `out/runs` at all.
    """
    main = _repo(tmp_path / "main")
    wt = tmp_path / "wt"
    subprocess.run(["git", "-C", str(main), "worktree", "add", "-q",
                    str(wt), "-b", "lane"], check=True)
    monkeypatch.setattr(runlog, "_repo_root", lambda: str(wt))

    assert runlog.run_dir() == os.path.join(str(main), "out", "runs")


def test_an_ordinary_clone_still_writes_beside_the_package(
        tmp_path, monkeypatch, fresh_run_dir):
    """Outside a worktree `--git-common-dir` is just `.git`, so this
    resolves where it always did. The fix must not move anyone else."""
    repo = _repo(tmp_path / "solo")
    monkeypatch.setattr(runlog, "_repo_root", lambda: str(repo))

    assert runlog.run_dir() == os.path.join(str(repo), "out", "runs")


def test_a_tree_git_cannot_read_falls_back_to_the_package(
        tmp_path, monkeypatch, fresh_run_dir):
    """No git, no worktree, no crash — telemetry must not be the thing
    that stops a run from starting."""
    plain = tmp_path / "nogit"
    plain.mkdir()
    monkeypatch.setattr(runlog, "_repo_root", lambda: str(plain))

    assert runlog.run_dir() == os.path.join(str(plain), "out", "runs")


def test_an_explicit_override_wins_over_both(tmp_path, monkeypatch,
                                             fresh_run_dir):
    monkeypatch.setenv("HDC_RUN_DIR", str(tmp_path / "elsewhere"))
    runlog.run_dir.cache_clear()

    assert runlog.run_dir() == str(tmp_path / "elsewhere")
