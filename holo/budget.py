"""Memory headroom checks for the heavy runs.

Deliberately NOT part of the SDK surface and absent from
`holo/__init__.py`: this is a developer utility, not a technique.

It exists because CONTRIBUTING.md already carried the rule — check `ps`
for running >4 GB jobs before launching a heavy encode — and a rule you
have to remember is a rule you eventually forget. Two concurrent
real-scene runs OOM-killed each other once; a later session launched a
lambda sweep as several parallel processes (each rebuilding the SAME
537 MB Gram) next to a 15 GB splat trainer and did it twice more. The
sweep was also the wrong shape: one process reusing the Gram is cheaper
in both memory and time.

Stdlib only. Every probe returns None rather than guessing when it
cannot read the platform, and `require_headroom` says so out loud — a
guard that silently passes is worse than no guard, because it is
mistaken for a check.
"""
import contextlib
import os
import resource
import subprocess
import sys

__all__ = ["available_gb", "heavy_processes", "heavy_run", "peak_rss_gb",
           "report_peak", "require_headroom"]

GB = 1 << 30

#: Left for the OS and everything not visible to `ps` as a big job.
RESERVE_GB = 4.0

#: A job at or above this is "heavy" — the figure CONTRIBUTING.md names.
HEAVY_GB = 4.0

#: Argv is truncated to this in reports; a full one can run to kilobytes.
_ARGV_CHARS = 90


def peak_rss_gb():
    """This process's peak resident size, in GB.

    `ru_maxrss` is BYTES on the BSDs (macOS included) and KILOBYTES on
    Linux. Getting that wrong is a silent 1024x, which is why
    tests/test_budget.py checks the value against a known allocation
    rather than trusting the unit.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / GB if sys.platform == "darwin" else raw * 1024 / GB


def available_gb():
    """Memory a new job could actually claim, in GB, or None if this
    platform is not one we know how to read."""
    if sys.platform == "darwin":
        return _available_darwin()
    if sys.platform.startswith("linux"):
        return _available_linux()
    return None


def _available_darwin():
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    page, pages = 4096, {}
    for line in out.splitlines():
        if line.startswith("Mach Virtual Memory Statistics"):
            page = int(line.rsplit("of ", 1)[1].split()[0])
        elif ":" in line:
            k, _, v = line.partition(":")
            v = v.strip().rstrip(".")
            if v.isdigit():
                pages[k.strip()] = int(v)
    # inactive and speculative pages are reclaimable on demand, so a job
    # can have them; wired and active pages it cannot.
    have = sum(pages.get(k, 0) for k in
               ("Pages free", "Pages inactive", "Pages speculative"))
    return have * page / GB if pages else None


def _available_linux():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / GB
    except OSError:
        pass
    return None


def _ancestors():
    """This process and every parent of it, so a heavy parent does not
    veto its own child."""
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,ppid="], capture_output=True,
                             text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return {os.getpid()}
    parent = {}
    for line in out.splitlines():
        bits = line.split()
        if len(bits) == 2 and bits[0].isdigit() and bits[1].isdigit():
            parent[int(bits[0])] = int(bits[1])
    seen, pid = set(), os.getpid()
    while pid and pid not in seen:
        seen.add(pid)
        pid = parent.get(pid, 0)
    return seen


def heavy_processes(threshold_gb=HEAVY_GB):
    """[(gb, pid, command)] for every OTHER process at or above the
    threshold, largest first, or None if `ps` is unreadable.

    Not filtered to python: the job that shared the machine last time
    was a compiled splat trainer holding 15 GB. Reported by ARGV rather
    than `comm`, because this venv's interpreter is a symlink and `comm`
    resolves it to an Xcode framework path that names no job at all.
    """
    try:
        out = subprocess.run(["ps", "-Ao", "rss=,pid=,command="],
                             capture_output=True, text=True, timeout=10,
                             check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    mine, found = _ancestors(), []
    for line in out.splitlines():
        bits = line.split(None, 2)
        if len(bits) != 3 or not bits[0].isdigit() or not bits[1].isdigit():
            continue
        gb, pid = int(bits[0]) * 1024 / GB, int(bits[1])
        if gb >= threshold_gb and pid not in mine:
            found.append((gb, pid, _short_argv(bits[2])))
    return sorted(found, reverse=True)


def _short_argv(cmd):
    """`/very/long/path/to/Python -m examples.run_x data/y` reads as
    `Python -m examples.run_x data/y`. The interpreter path is the least
    informative 60 characters on the line, and truncating from the left
    would keep exactly those."""
    exe, _, args = cmd.partition(" ")
    short = (exe.rsplit("/", 1)[-1] + (" " + args if args else "")).strip()
    return short[:_ARGV_CHARS - 1] + "\u2026" if len(short) > _ARGV_CHARS else short


def require_headroom(need_gb, force=False, reserve_gb=RESERVE_GB):
    """Refuse to start when `need_gb` will not fit beside what is
    already running. Returns the available GB (or None when unknown).

    Raises MemoryError with the arithmetic and the offending jobs, so
    the caller can wait for one rather than discover the OOM killer.
    `force=True` prints the same reasoning and proceeds anyway.
    """
    avail = available_gb()
    heavy = heavy_processes()
    if avail is None:
        print("  memory: cannot read this platform's free memory — "
              "NOT checked (%s). Watch it yourself." % sys.platform)
        return None
    detail = ""
    if heavy:
        detail = "\n  already running:\n" + "\n".join(
            "    %6.1f GB  pid %-7d %s" % h for h in heavy[:5])
    room = avail - reserve_gb
    line = ("  memory: need ~%.1f GB, %.1f GB available, %.1f GB reserved "
            "for the OS -> %.1f GB usable%s"
            % (need_gb, avail, reserve_gb, room, detail))
    if need_gb <= room:
        print(line)
        return avail
    if force:
        print(line + "\n  FORCED past the headroom check; expect the OOM "
                     "killer if this was wrong.")
        return avail
    raise MemoryError(
        "not enough memory to start.\n" + line + "\n"
        "  wait for a job above to finish, or pass force=True "
        "(examples take --force-memory) if you know better. "
        "CONTRIBUTING.md: two concurrent real-scene runs have OOM-killed "
        "each other.")


def report_peak(label="", declared_gb=None):
    """One line, for the end of a heavy run.

    Pass what the run declared to `require_headroom` and the estimate
    corrects itself in public: a guard whose numbers nobody ever checks
    drifts into either refusing valid runs or protecting nothing.
    """
    gb = peak_rss_gb()
    note = ""
    if declared_gb:
        if gb > declared_gb:
            note = ("  OVER the %.1f GB declared — raise it, the guard is "
                    "under-protecting" % declared_gb)
        elif gb < 0.4 * declared_gb:
            note = ("  well under the %.1f GB declared — lower it, the guard "
                    "is refusing runs that would fit" % declared_gb)
    print("  peak RSS%s: %.2f GB%s"
          % (" (%s)" % label if label else "", gb, note))
    return gb


@contextlib.contextmanager
def heavy_run(need_gb, label="", force=False):
    """Guard on the way in, report on the way out.

    Wrapped around the ENTRYPOINT rather than a function body, because
    peak RSS is a property of the process and the guard belongs where
    the process decides to start. The report fires on the way out of a
    crash too — a run that died holding 40 GB is exactly the one whose
    number you want.
    """
    require_headroom(need_gb, force=force)
    try:
        yield
    finally:
        report_peak(label, need_gb)
