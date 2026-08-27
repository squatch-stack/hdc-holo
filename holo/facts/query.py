"""Query logic behind the facts MCP server — plain functions, no MCP.

The server (`mcp_server.py`) is a thin stdio shim over these three, so
the logic tests under the stdlib-only CI contract and any future
surface (CLI, HTTP, a second agent protocol) reuses it unchanged.

`search_claims` ranks registry records by keyword overlap, unioned
with fuzzy chunk hits mapped back through each claim's cite files —
the Context7 resolve-then-query shape. `get_claim` returns the full
record, its supersession chain, a LIVE derivation run, and the cite
sites with line numbers. `search_kb` runs the same matrix search over
a sibling knowledge-base checkout (HOLO_KB_PATH or config kb_path) and
reports honestly when none is configured.
"""

import os

from .check import DERIVATIONS, _match_values, load_config
from .normalize import canon, front_matter, normalize_file
from .registry import base_id, load_registry

__all__ = ["search_claims", "get_claim", "search_kb"]


def _registry(root):
    return load_registry(os.path.join(root, "claims", "registry.jsonl"))


def _render(claim, score=None):
    out = {"id": claim.id,
           "statement": claim.statement.replace("{value}",
                                                str(claim.value)),
           "value": claim.value, "units": claim.units,
           "kind": claim.kind, "status": claim.status,
           "as_of": claim.as_of, "source": claim.source,
           "cites": claim.cites, "notes": claim.notes}
    if score is not None:
        out["score"] = round(score, 3)
    return out


def search_claims(root, query, status="current", limit=8):
    claims = _registry(root)
    if status != "any":
        claims = [c for c in claims if c.status == status]
    tokens = [t for t in query.lower().split() if len(t) > 2]

    scores = {}
    for c in claims:
        hay = " ".join([c.id, c.statement, c.notes, c.units,
                        str(c.value), " ".join(c.cites)]).lower()
        scores[c.id] = float(sum(1 for t in tokens if t in hay))

    fuzzy_note = None
    try:
        from . import index as indexmod
        threshold = load_config(root).get("fuzzy_threshold", 0.18)
        for hit_score, chunk, _ in indexmod.search(root, query, k=6,
                                                   verify=False):
            if hit_score < threshold:
                continue
            for c in claims:
                if chunk["file"] in c.cites:
                    scores[c.id] = scores.get(c.id, 0) + hit_score
    except Exception as e:   # index or dispatch absent: registry-only
        fuzzy_note = "fuzzy union unavailable: %s" % e

    by_id = {c.id: c for c in claims}
    ranked = sorted((s, cid) for cid, s in scores.items() if s > 0)
    out = [_render(by_id[cid], s) for s, cid in reversed(ranked)][:limit]
    result = {"query": query, "results": out}
    if fuzzy_note:
        result["note"] = fuzzy_note
    return result


def get_claim(root, claim_id):
    claims = _registry(root)
    by_id = {c.id: c for c in claims}
    claim = by_id.get(claim_id)
    if claim is None:
        near = [c.id for c in claims
                if base_id(c.id) == base_id(claim_id)]
        return {"error": "unknown claim id %r" % claim_id,
                "did_you_mean": near}

    chain, seen = [], set()
    cur = claim
    while cur and cur.id not in seen:          # newest to oldest
        seen.add(cur.id)
        nxt = by_id.get(cur.superseded_by or "")
        if nxt and nxt.id not in seen:
            cur = nxt
            continue
        break
    while cur and cur.id not in {c["id"] for c in chain}:
        chain.append({"id": cur.id, "value": cur.value,
                      "status": cur.status, "as_of": cur.as_of})
        cur = by_id.get(cur.supersedes or base_id(cur.id) + "@_none_")

    record = _render(claim)
    record["chain"] = chain
    record["evidence"] = claim.evidence

    fn = DERIVATIONS.get(claim.check.get("fn")) if claim.check else None
    if fn:
        try:
            derived = fn(root)
            record["derivation"] = {
                "fn": claim.check["fn"], "derived": derived,
                "matches": canon(derived) in
                {canon(v) for v in claim.accepted_values()}}
        except Exception as e:
            record["derivation"] = {"fn": claim.check.get("fn"),
                                    "error": str(e)}

    sites = []
    for cite in claim.cites:
        path = os.path.join(root, cite)
        if not os.path.exists(path):
            continue
        for par in normalize_file(path):
            par.file = cite
            if _match_values(claim, par.text):
                sites.append({"file": cite, "line": par.line_start})
    record["cite_sites"] = sites
    return record


def search_kb(root, query, limit=8):
    kb = os.environ.get("HOLO_KB_PATH") or \
        load_config(root).get("kb_path")
    if not kb:
        return {"configured": False,
                "note": "no knowledge base configured — set HOLO_KB_PATH "
                        "or claims/config.json kb_path to a math-kb "
                        "checkout"}
    kb = os.path.expanduser(kb)
    if not os.path.isabs(kb):
        kb = os.path.normpath(os.path.join(root, kb))
    if not os.path.isdir(kb):
        return {"configured": False,
                "note": "kb_path %r does not exist" % kb}
    try:
        from . import index as indexmod
        from .chunk import chunk_file
        hits = indexmod.search(kb, query, k=limit)
    except FileNotFoundError:
        return {"configured": True, "indexed": False,
                "note": "KB has no index — run `holo-facts index` in %r"
                        % kb}
    except RuntimeError as e:
        return {"configured": True, "indexed": False, "note": str(e)}
    results = []
    for score, chunk, stale in hits:
        path = os.path.join(kb, chunk["file"])
        snippet = ""
        if os.path.exists(path):
            match = [c for c in chunk_file(path, chunk["file"])
                     if c.sha == chunk["sha256"]]
            if match:
                snippet = match[0].text[:240]
        fm = front_matter(path)
        results.append({"repo": os.path.basename(kb),
                        "file": chunk["file"],
                        "lines": chunk["lines"],
                        "heading": chunk["heading"],
                        "topic": fm.get("topic", ""),
                        "arxiv_ids": fm.get("arxiv", []),
                        "score": round(score, 3),
                        "stale": stale, "snippet": snippet})
    return {"configured": True, "query": query, "results": results}
