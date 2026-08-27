"""holo-facts: the claims registry CLI.

Usage:
  holo-facts check [--strict] [--json] [--root DIR]
  holo-facts new
  holo-facts index | search | mcp        (phase 2/3 — not built yet)

`check` warns by default (pre-commit mode) and exits 1 on FAIL findings
only under --strict (the CI mode). Exit 2 = registry malformed or
internal error.
"""

import argparse
import json
import os
import sys

from . import check as checkmod

_TEMPLATE = {
    "id": "module.claim_name", "statement": "… {value} …", "value": None,
    "units": "", "kind": "measurement", "status": "current",
    "as_of": {"date": "", "version": ""},
    "source": {"doc": "SDK.md#02-findings-running-log", "generator": ""},
    "evidence": [], "patterns": [], "accepted": [], "context_any": [],
    "cites": [], "allow_historical_in": [], "check": {}, "tolerance": None,
    "lane": "", "notes": "",
}


def _find_root(start):
    d = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(d, "claims", "config.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def main(argv=None):
    ap = argparse.ArgumentParser(prog="holo-facts", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    p_check = sub.add_parser("check", help="run the stale-claim checker")
    p_check.add_argument("--strict", action="store_true",
                         help="exit 1 on FAIL findings (CI mode)")
    p_check.add_argument("--json", action="store_true")
    p_check.add_argument("--fuzzy", action="store_true",
                         help="(phase 2) add fuzzy paraphrase warns")
    p_check.add_argument("--root", default=".")
    sub.add_parser("new", help="print a registry line template")
    for name in ("index", "search", "mcp"):
        sub.add_parser(name)
    args = ap.parse_args(argv)

    if args.cmd == "new":
        print(json.dumps(_TEMPLATE))
        return 0
    if args.cmd in ("index", "search", "mcp"):
        print("holo-facts %s is not built yet (phase %s of the facts plan)"
              % (args.cmd, "3" if args.cmd == "mcp" else "2"),
              file=sys.stderr)
        return 2
    if args.cmd != "check":
        ap.print_help()
        return 2

    root = _find_root(args.root)
    if root is None:
        print("no claims/config.json found upward of %s" % args.root,
              file=sys.stderr)
        return 2
    if args.fuzzy:
        print("note: --fuzzy is phase 2; running exact checks only",
              file=sys.stderr)
    try:
        result = checkmod.run(root, strict=args.strict)
    except Exception as e:
        print("internal error: %s" % e, file=sys.stderr)
        return 2

    if args.json:
        print(result.to_json())
    else:
        for f in result.findings:
            print(f.render())
        print("%d FAIL, %d WARN" % (len(result.fails), len(result.warns)))
    if args.strict and result.fails:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
