"""holo-quality: structure rules + the lint ratchet.

  holo-quality check       structure + lint ratchet (CI and the hook)
  holo-quality structure   project-layout rules only
  holo-quality lint        ratchet only
  holo-quality baseline    re-record the debt (only ever to shrink it)
  holo-quality graph build      index the code into a Kuzu database
  holo-quality graph checks     run the canned structure questions
  holo-quality graph query "MATCH (f:Function) ... RETURN ..."

`check` exits 1 on FAIL findings or lint regressions, 0 otherwise;
exit 2 means the tooling itself could not run.
"""

import argparse
import os
import sys

from . import ratchet, structure


def _find_root(start):
    d = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(d, "pyproject.toml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _run_structure(root):
    findings = structure.check_structure(root)
    for level, code, path, msg in findings:
        print("%-4s %-18s %-28s %s" % (level, code, path, msg))
    return [f for f in findings if f[0] == "FAIL"]


def _run_lint(root, quiet=False):
    """(regressions, improvements) or None when the baseline is absent."""
    current = ratchet.collect(root)
    baseline = ratchet.load_baseline(root)
    if baseline is None:
        print("no baseline yet — run: holo-quality baseline",
              file=sys.stderr)
        return None
    regressions, improvements = ratchet.compare(current, baseline)
    for key, was, now in regressions:
        path, _, code = key.rpartition("::")
        print("FAIL %-18s %-28s %d -> %d violations"
              % ("lint-regression", "%s (%s)" % (path, code), was, now))
    if improvements and not quiet:
        fixed = sum(was - now for _, was, now in improvements)
        print("note: %d violation(s) fixed since the baseline in %d "
              "place(s) — `holo-quality baseline` to lock the gain in"
              % (fixed, len(improvements)))
    if not quiet:
        print("lint debt: %d (baseline %d)"
              % (sum(current.values()), sum(baseline.values())))
    return regressions, improvements


def _cmd_baseline(root):
    counts = ratchet.collect(root)
    ratchet.save_baseline(root, counts)
    print("baseline: %d violations across %d (file, rule) pairs -> %s"
          % (sum(counts.values()), len(counts), ratchet.BASELINE_PATH))
    return 0


def _cmd_structure(root):
    return 1 if _run_structure(root) else 0


def _cmd_lint(root):
    result = _run_lint(root)
    if result is None:
        return 2
    return 1 if result[0] else 0


def _cmd_check(root):
    fails = _run_structure(root)
    result = _run_lint(root)
    if result is None:
        return 2
    return 1 if (fails or result[0]) else 0


def _print_table(columns, rows):
    if not rows:
        print("  (no rows)")
        return
    widths = [max(len(str(c)), *(len(str(r[i])) for r in rows))
              for i, c in enumerate(columns)]
    print("  " + "  ".join(str(c).ljust(w) for c, w in zip(columns, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("  " + "  ".join(str(v).ljust(w)
                               for v, w in zip(row, widths)))


def _graph_build(root, _cypher):
    from . import graph
    counts = graph.build(root)
    print("graph built -> %s" % graph.DB_PATH)
    print("  %(modules)d modules, %(functions)d functions, %(classes)d "
          "classes, %(imports)d imports, %(calls)d calls" % counts)
    return 0


def _graph_query(root, cypher):
    from . import graph
    if not cypher:
        print('usage: holo-quality graph query "MATCH ..."',
              file=sys.stderr)
        return 2
    _print_table(*graph.query(graph.connect(root), cypher))
    return 0


def _graph_checks(root, only):
    from . import graph
    conn = graph.connect(root)
    for name, description, cypher in graph.CHECKS:
        if only and only != name:
            continue
        print("\n== %s — %s" % (name, description))
        _print_table(*graph.query(conn, cypher))
    return 0


GRAPH_ACTIONS = {"build": _graph_build, "query": _graph_query,
                 "checks": _graph_checks}


COMMANDS = {"baseline": _cmd_baseline, "structure": _cmd_structure,
            "lint": _cmd_lint, "check": _cmd_check}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="holo-quality", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    for name in COMMANDS:
        sub.add_parser(name).add_argument("--root", default=".")
    g = sub.add_parser("graph")
    g.add_argument("action", choices=sorted(GRAPH_ACTIONS))
    g.add_argument("cypher", nargs="?",
                   help="Cypher for `query`; a check name for `checks`")
    g.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 2
    root = _find_root(args.root)
    if root is None:
        print("no pyproject.toml found upward of %s" % args.root,
              file=sys.stderr)
        return 2
    try:
        if args.cmd == "graph":
            return GRAPH_ACTIONS[args.action](root, args.cypher)
        return COMMANDS[args.cmd](root)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
