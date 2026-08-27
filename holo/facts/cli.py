"""holo-facts: the claims registry CLI.

Usage:
  holo-facts check [--strict] [--fuzzy] [--json] [--root DIR]
  holo-facts index                 build the fuzzy chunk index
  holo-facts search "query" [-k 8] rank chunks by trigram cosine
  holo-facts calibrate             score histograms -> threshold advice
  holo-facts new                   print a registry line template
  holo-facts mcp                   (phase 3 — not built yet)

`check` warns by default (pre-commit mode) and exits 1 on FAIL findings
only under --strict (the CI mode); `--fuzzy` adds WARN-only paraphrase
probes of superseded claims against the index. Exit 2 = registry
malformed / index missing / internal error.
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
                         help="add WARN-only fuzzy paraphrase probes")
    p_check.add_argument("--root", default=".")
    sub.add_parser("new", help="print a registry line template")
    p_index = sub.add_parser("index", help="build the fuzzy chunk index")
    p_index.add_argument("--root", default=".")
    p_search = sub.add_parser("search", help="rank chunks for a query")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=8)
    p_search.add_argument("--root", default=".")
    p_cal = sub.add_parser("calibrate",
                           help="score histograms -> threshold advice")
    p_cal.add_argument("--root", default=".")
    sub.add_parser("mcp")
    args = ap.parse_args(argv)

    if args.cmd == "new":
        print(json.dumps(_TEMPLATE))
        return 0
    if args.cmd == "mcp":
        root = _find_root(".")
        if root is None:
            print("no claims/config.json found upward of cwd",
                  file=sys.stderr)
            return 2
        from . import mcp_server
        try:
            mcp_server.serve(root)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 2
        return 0
    if args.cmd not in ("check", "index", "search", "calibrate"):
        ap.print_help()
        return 2

    root = _find_root(args.root)
    if root is None:
        print("no claims/config.json found upward of %s" % args.root,
              file=sys.stderr)
        return 2

    if args.cmd == "index":
        from . import index as indexmod
        try:
            n = indexmod.build_index(root, checkmod.load_config(root))
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 2
        print("indexed %d chunks -> %s" % (n, indexmod.INDEX_DIR))
        return 0

    if args.cmd == "search":
        from . import index as indexmod
        config = checkmod.load_config(root)
        threshold = config.get("fuzzy_threshold", 0.18)
        try:
            hits = indexmod.search(root, args.query, k=args.k)
        except (RuntimeError, FileNotFoundError) as e:
            print("index unavailable (%s) — run: holo-facts index" % e,
                  file=sys.stderr)
            return 2
        for score, c, stale in hits:
            tag = " (stale — reindex)" if stale else ""
            tag += " (abstain)" if score < threshold else ""
            print("%6.3f  %s:%d-%d  [%s]%s"
                  % (score, c["file"], c["lines"][0], c["lines"][1],
                     (c["heading"] or "-")[:48], tag))
        return 0

    if args.cmd == "calibrate":
        from . import calibrate as calmod
        return calmod.main(root)

    try:
        result = checkmod.run(root)
    except Exception as e:
        print("internal error: %s" % e, file=sys.stderr)
        return 2
    if args.fuzzy:
        from . import index as indexmod
        from .registry import load_registry
        import os as _os
        try:
            claims = load_registry(
                _os.path.join(root, "claims", "registry.jsonl"))
            result.findings.extend(indexmod.fuzzy_findings(
                root, claims, checkmod.load_config(root)))
        except (RuntimeError, FileNotFoundError) as e:
            print("fuzzy layer unavailable (%s) — run: holo-facts index"
                  % e, file=sys.stderr)
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
