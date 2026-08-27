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


def _build_parser():
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
    return ap


def _cmd_new(args, root):
    print(json.dumps(_TEMPLATE))
    return 0


def _cmd_mcp(args, root):
    from . import mcp_server
    try:
        mcp_server.serve(root)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


def _cmd_index(args, root):
    from . import index as indexmod
    try:
        n = indexmod.build_index(root, checkmod.load_config(root))
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2
    print("indexed %d chunks -> %s" % (n, indexmod.INDEX_DIR))
    return 0


def _cmd_search(args, root):
    from . import index as indexmod
    threshold = checkmod.load_config(root).get("fuzzy_threshold", 0.18)
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


def _cmd_calibrate(args, root):
    from . import calibrate as calmod
    return calmod.main(root)


def _add_fuzzy_findings(result, root):
    """WARN-only paraphrase probes; returns an exit code or None."""
    from . import index as indexmod
    from .registry import load_registry
    try:
        claims = load_registry(
            os.path.join(root, "claims", "registry.jsonl"))
        result.findings.extend(indexmod.fuzzy_findings(
            root, claims, checkmod.load_config(root)))
    except (RuntimeError, FileNotFoundError) as e:
        print("fuzzy layer unavailable (%s) — run: holo-facts index"
              % e, file=sys.stderr)
        return 2
    return None


def _report(result, as_json):
    if as_json:
        print(result.to_json())
        return
    for f in result.findings:
        print(f.render())
    print("%d FAIL, %d WARN" % (len(result.fails), len(result.warns)))


def _cmd_check(args, root):
    try:
        result = checkmod.run(root)
    except Exception as e:
        print("internal error: %s" % e, file=sys.stderr)
        return 2
    if args.fuzzy:
        failed = _add_fuzzy_findings(result, root)
        if failed is not None:
            return failed
    _report(result, args.json)
    return 1 if (args.strict and result.fails) else 0


COMMANDS = {"new": _cmd_new, "mcp": _cmd_mcp, "index": _cmd_index,
            "search": _cmd_search, "calibrate": _cmd_calibrate,
            "check": _cmd_check}

#: `new` needs no repo at all; `mcp` resolves from its own working
#: directory because the client, not the caller, chooses where it runs
_ROOTLESS = {"new"}
_ROOT_FROM_CWD = {"mcp"}


def main(argv=None):
    ap = _build_parser()
    args = ap.parse_args(argv)
    if args.cmd not in COMMANDS:
        ap.print_help()
        return 2
    if args.cmd in _ROOTLESS:
        return COMMANDS[args.cmd](args, None)

    start = "." if args.cmd in _ROOT_FROM_CWD else args.root
    root = _find_root(start)
    if root is None:
        print("no claims/config.json found upward of %s" % start,
              file=sys.stderr)
        return 2
    return COMMANDS[args.cmd](args, root)


if __name__ == "__main__":
    sys.exit(main())
