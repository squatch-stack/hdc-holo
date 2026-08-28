"""A record of what a heavy run did — including the ones that died.

Deliberately NOT part of the SDK surface and absent from
`holo/__init__.py`: like `holo.budget`, this is developer tooling.

THE DESIGN POINT. A process killed by the OOM killer cannot write its
own epitaph — SIGKILL runs no `atexit` handler and no `except` block. So
this writes TWO records, one when a run starts and one when it ends, and
a run with a start and no end is one that was killed. That inversion is
the whole idea; a better end-of-run summary would have recorded nothing
at all for the four runs lost in one evening (two OOM kills, one Metal
command-buffer fault as the innocent victim of another process, and one
silent SIGKILL).

What the started record carries is chosen from what was actually missing
when those runs were reconstructed afterwards:

  * the commit and whether the tree was dirty — a number whose code is
    unknown is not reproducible;
  * the backend, because HDC_BACKEND=numpy and MLX/Metal give different
    results and nothing recorded which ran;
  * WHAT ELSE WAS RUNNING. The same command took 769 s and 327 s on the
    same machine depending on whether a splat trainer had the box, and
    because that went unrecorded a PR body had to say "wall-clock across
    versions is not comparable" instead of showing why.

Usage:

    from holo import runlog

    with runlog.record("projection-pipeline", need_gb=5.5) as run:
        run.stage("xfine", 89.0)
        run.result(top_down=0.2170, side=0.1367)

Reading them back:

    python -m holo.runlog              # recent runs, killed ones flagged
    python -m holo.runlog --killed     # only the ones with no end record
"""
import contextlib
import functools
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from . import budget

__all__ = ["record", "run_dir", "runs", "summarize"]


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@functools.lru_cache(maxsize=1)
def run_dir():
    """Where run records go: `out/runs` in the SHARED checkout.

    Gitignored — out/ otherwise holds committed evidence figures, and a
    heavy run should not dirty the tree just by being observed.

    Not package-relative, which is what this used to be, because of
    where these runs actually happen. Every lane here works in its own
    git worktree, so a package-relative path writes the record INTO the
    worktree and `git worktree remove` then deletes it along with the
    lane. Every record written between this module landing and
    2026-08-28 went that way, and nobody noticed, because a missing file
    looks exactly like a run that was never launched.

    `git rev-parse --git-common-dir` names the main repository's .git
    from inside any worktree, so its parent is the shared checkout; in
    an ordinary clone it names `.git` and this resolves to the same
    place it always did. `HDC_RUN_DIR` overrides both, and the
    package-relative path remains the fallback for a tree git cannot
    read.

    Cached: resolved on first write, not at import, because finding it
    costs a `git` call and importing a module should not shell out.
    `run_dir.cache_clear()` re-reads the environment.
    """
    return os.environ.get("HDC_RUN_DIR") or os.path.join(
        _shared_root(), "out", "runs")


def _shared_root():
    common = _git("rev-parse", "--git-common-dir")
    if not common:
        return _repo_root()
    # a relative answer is relative to the -C git ran under, not to cwd
    if not os.path.isabs(common):
        common = os.path.join(_repo_root(), common)
    return os.path.dirname(os.path.normpath(common))


def _git(*args):
    try:
        out = subprocess.run(("git", "-C", _repo_root(), *args),
                             capture_output=True, text=True, timeout=10,
                             check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _provenance():
    """Commit, dirtiness and backend — the three things that decide
    whether a number can be reproduced at all."""
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    try:
        from . import accel
        backend = accel.backend_name()
    except Exception:
        backend = "unknown"
    return {
        "sha": sha[:12] if sha else None,
        "dirty": bool(status) if status is not None else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "backend": backend,
        "python": sys.version.split()[0],
    }


def _machine():
    out = {"platform": sys.platform, "cpus": os.cpu_count()}
    avail = budget.available_gb()
    if avail is not None:
        out["available_gb"] = round(avail, 1)
    return out


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path(when=None):
    directory = run_dir()
    os.makedirs(directory, exist_ok=True)
    day = (when or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    return os.path.join(directory, "%s.jsonl" % day)


def _append(row, path=None):
    path = path or _path()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())      # the next thing may be a SIGKILL


class Run:
    """Handle for the run in progress; `record()` yields one."""

    def __init__(self, run_id, label, path):
        self.id = run_id
        self.label = label
        self.path = path
        self.stages = []
        self.results = {}
        self.notes = []

    def stage(self, name, seconds, **extra):
        """Time one phase. Recorded in order, so a killed run's last
        stage says how far it got."""
        row = {"name": name, "seconds": round(float(seconds), 3)}
        row.update(extra)
        self.stages.append(row)
        return row

    def result(self, **values):
        self.results.update(values)

    def note(self, text):
        self.notes.append(str(text))


@contextlib.contextmanager
def record(label, need_gb=0.0, force=False, path=None):
    """Guard on headroom, write a start record, and leave an epitaph.

    The start record is written BEFORE the headroom check so a refused
    run is visible too — "it never started" and "it started and
    vanished" are different diagnoses and both are worth having.
    """
    run_id = "%s-%d" % (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
                        os.getpid())
    path = path or _path()
    run = Run(run_id, label, path)
    started = {"run": run_id, "event": "started", "label": label,
               "at": _now(), "argv": sys.argv, "declared_gb": need_gb,
               "cwd": os.getcwd()}
    started.update(_provenance())
    started["machine"] = _machine()
    heavy = budget.heavy_processes()
    if heavy:
        started["contending"] = [{"gb": round(gb, 1), "pid": pid, "cmd": cmd}
                                 for gb, pid, cmd in heavy[:5]]
    _append(started, path)

    t0 = time.time()
    status, err = "ok", None
    try:
        if need_gb:
            budget.require_headroom(need_gb, force=force)
        yield run
    except MemoryError as exc:
        status, err = "refused", str(exc).split("\n")[0]
        raise
    except BaseException as exc:
        status, err = "error", "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        ended = {"run": run_id, "event": "ended", "label": label,
                 "at": _now(), "wall_s": round(time.time() - t0, 2),
                 "peak_rss_gb": round(budget.peak_rss_gb(), 2),
                 "status": status}
        if err:
            ended["error"] = err[:400]
        if run.stages:
            ended["stages"] = run.stages
        if run.results:
            ended["results"] = run.results
        if run.notes:
            ended["notes"] = run.notes
        _append(ended, path)
        print("  run %s: %s in %.0fs, peak %.2f GB  (%s)"
              % (run_id, status, ended["wall_s"], ended["peak_rss_gb"],
                 os.path.relpath(path, _repo_root())), flush=True)


# ---------------------------------------------------------------------------
# Reading them back
# ---------------------------------------------------------------------------

def runs(directory=None, days=None):
    """[(started, ended-or-None)] newest last. A None end is a run that
    was KILLED: nothing else can remove the end record, because it is
    written in a finally block that even an exception reaches."""
    directory = directory or run_dir()
    if not os.path.isdir(directory):
        return []
    files = sorted(f for f in os.listdir(directory) if f.endswith(".jsonl"))
    if days:
        files = files[-days:]
    starts, ends = {}, {}
    order = []
    for name in files:
        with open(os.path.join(directory, name), encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue          # a line torn by a kill mid-write
                rid = row.get("run")
                if row.get("event") == "started":
                    starts[rid] = row
                    order.append(rid)
                elif row.get("event") == "ended":
                    ends[rid] = row
    return [(starts[r], ends.get(r)) for r in order if r in starts]


def summarize(directory=None, killed_only=False, limit=25):
    rows = runs(directory)
    if killed_only:
        rows = [(s, e) for s, e in rows if e is None]
    rows = rows[-limit:]
    if not rows:
        print("no runs recorded in %s" % (directory or run_dir()))
        return rows
    print("%-24s %-22s %-9s %8s %8s  %s"
          % ("run", "label", "status", "wall_s", "peak_gb", "sha"))
    for start, end in rows:
        status = "KILLED" if end is None else end.get("status", "?")
        wall = "" if end is None else "%.0f" % end.get("wall_s", 0)
        peak = "" if end is None else "%.2f" % end.get("peak_rss_gb", 0)
        sha = (start.get("sha") or "-") + ("+" if start.get("dirty") else "")
        print("%-24s %-22s %-9s %8s %8s  %s"
              % (start["run"], start.get("label", "")[:22], status, wall,
                 peak, sha))
        if end is None and start.get("contending"):
            worst = start["contending"][0]
            print("%-24s   ^ %.1f GB was held by %s"
                  % ("", worst["gb"], worst["cmd"][:60]))
    return rows


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="recorded runs; killed ones "
                                             "are those with no end record")
    ap.add_argument("--killed", action="store_true")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args(argv)
    summarize(args.dir, killed_only=args.killed, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
