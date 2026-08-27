"""Lint ratchet: today's violations are a debt, not a gate.

A repo that adopts a linter late has two bad options — block on 291
existing violations (nobody can commit) or report and never block
(nothing changes). The ratchet is the third: `quality/baseline.json`
records the count of every (file, rule) pair as of adoption, and CI
fails only when a pair's count RISES or a new pair appears. Existing
debt is visible and frozen; new debt is impossible. Paying debt down
is a separate, deliberate act (`ruff check --fix`, then
`holo-quality baseline`), and the baseline can only shrink.

Deliberately keyed by (file, rule) rather than by line: line numbers
churn on every edit, so a line-keyed baseline would either go stale
instantly or need fuzzy matching. Counts per file are stable under
edits that do not add violations, which is exactly the property a
gate needs.
"""

import json
import os
import subprocess
import sys

__all__ = ["BASELINE_PATH", "collect", "compare", "load_baseline",
           "save_baseline"]

BASELINE_PATH = os.path.join("quality", "baseline.json")


def collect(root, ruff=None):
    """{(relpath, code): count} for the whole tree, via ruff's JSON."""
    exe = ruff or os.path.join(os.path.dirname(sys.executable), "ruff")
    if not os.path.exists(exe):
        exe = "ruff"
    try:
        proc = subprocess.run(
            [exe, "check", ".", "--output-format", "json", "--no-cache"],
            cwd=root, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ruff not found — pip install 'hdc-holo[quality]' (the "
            "checker's own dependency, kept out of the core)") from e
    if proc.returncode not in (0, 1):      # 1 = violations found
        raise RuntimeError("ruff failed: %s" % (proc.stderr.strip()[:400]))
    counts = {}
    for v in json.loads(proc.stdout or "[]"):
        rel = os.path.relpath(v["filename"], root)
        key = "%s::%s" % (rel, v["code"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def load_baseline(root):
    path = os.path.join(root, BASELINE_PATH)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("violations", {})


def save_baseline(root, counts, total=None):
    path = os.path.join(root, BASELINE_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "_comment": "Lint debt frozen at adoption. CI fails only on "
                        "increases. Regenerate with `holo-quality "
                        "baseline` — and only ever to record a DECREASE.",
            "total": total if total is not None else sum(counts.values()),
            "violations": dict(sorted(counts.items())),
        }, f, indent=1)
        f.write("\n")


def compare(current, baseline):
    """(regressions, improvements) as [(key, was, now)] lists."""
    regressions, improvements = [], []
    for key, now in sorted(current.items()):
        was = baseline.get(key, 0)
        if now > was:
            regressions.append((key, was, now))
    for key, was in sorted(baseline.items()):
        now = current.get(key, 0)
        if now < was:
            improvements.append((key, was, now))
    return regressions, improvements
