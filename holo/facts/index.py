"""The fuzzy layer: a trigram-profile MATRIX over doc chunks.

One L2-normalized complex64 hypervector row per chunk
(`holo.dispatch.FastNGramProfiler`, d=2048); ranking is
`Re(mat.conj() @ q)`. This is deliberately a matrix, never a bundle:
with K = N (every chunk is its own answer) a bundle's O(K) readout
advantage vanishes while its crosstalk floor `sqrt(N/(2d))` — ~0.45
at the corpus' ~800 chunks — sits above any usable threshold. Our own
capacity law forbids the romantic design; docs/facts.md carries the
arithmetic.

Persistence is the SDK's own tagged storage: each row through
`pack_polar` (HG-8 — the codec measured faithful on wide-dynamic-range
bundles; never HP, which discards magnitudes) concatenated into
`claims/index/profiles.hg8`, with `index-meta.json` holding file /
heading / line-span / sha256-of-normalized-text / blob offsets — and
NO plaintext. Retrieval re-reads the working tree; a sha mismatch
marks the hit stale and suggests a reindex.

Scores are lexical (trigram cosine): robust to typos, morphology, and
markup residue; blind to pure paraphrase that shares no character
trigrams; ignorant of word order beyond the trigram horizon. That is
why fuzzy findings are WARN-only — exact value matching owns the gate.
"""

import json
import os
import subprocess
import time

import numpy as np

from .chunk import chunk_surfaces, chunk_file

__all__ = ["build_index", "load_index", "search", "profiler",
           "fuzzy_findings", "INDEX_DIR"]

INDEX_DIR = os.path.join("claims", "index")
DIM, SEED = 2048, 0


def profiler():
    """The shared profiler (lazy: holo.dispatch is the dispatch lane's
    module and may be absent in older checkouts)."""
    try:
        from holo import dispatch
    except ImportError as e:
        raise RuntimeError(
            "the fuzzy layer needs holo.dispatch (FastNGramProfiler); "
            "not present in this checkout: %s" % e)
    from holo import FHRR
    space = FHRR(dim=DIM, seed=SEED)
    return dispatch.FastNGramProfiler(space, n=3)


def _git_head(root):
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def build_index(root, config):
    from holo.storage import pack_polar
    prof = profiler()
    chunks = chunk_surfaces(root, [s for s in _surface_list(root, config)])
    rows, meta_chunks, blob = [], [], bytearray()
    for c in chunks:
        v = prof.unit_profile(c.text)
        packed = pack_polar(v, bits=8)
        meta_chunks.append({"file": c.file, "heading": c.heading,
                            "lines": [c.line_start, c.line_end],
                            "sha256": c.sha,
                            "blob": [len(blob), len(packed)]})
        blob.extend(packed)
        rows.append(v)
    outdir = os.path.join(root, INDEX_DIR)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "profiles.hg8"), "wb") as f:
        f.write(bytes(blob))
    meta = {"dim": DIM, "seed": SEED, "codec": "HG-8",
            "built_at": time.time(), "git_head": _git_head(root),
            "chunks": meta_chunks}
    with open(os.path.join(outdir, "index-meta.json"), "w") as f:
        json.dump(meta, f)
    return len(chunks)


def _surface_list(root, config):
    import glob
    seen, out = set(), []
    for pat in config["surfaces"]:
        for path in sorted(glob.glob(os.path.join(root, pat))):
            rel = os.path.relpath(path, root)
            if rel not in seen and os.path.isfile(path):
                seen.add(rel)
                out.append(rel)
    return out


def load_index(root):
    """(meta, (N, d) complex64 matrix) — HG-8 rows re-normalized after
    dequantization so scores stay cosines."""
    from holo.storage import unpack
    outdir = os.path.join(root, INDEX_DIR)
    with open(os.path.join(outdir, "index-meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(outdir, "profiles.hg8"), "rb") as f:
        blob = f.read()
    rows = []
    for c in meta["chunks"]:
        off, length = c["blob"]
        v = unpack(blob[off:off + length]).astype(np.complex64)
        norm = np.linalg.norm(v)
        rows.append(v / norm if norm > 0 else v)
    mat = np.stack(rows) if rows else np.zeros((0, meta["dim"]),
                                               np.complex64)
    return meta, mat


def search(root, query, k=8, threshold=None, verify=True):
    """Ranked [(score, chunk_meta, stale?)]; scores below threshold are
    still returned, flagged abstain by the caller."""
    prof = profiler()
    meta, mat = load_index(root)
    q = prof.unit_profile(query)
    scores = np.real(mat.conj() @ q)
    order = np.argsort(scores)[::-1][:k]
    out = []
    fresh = {}
    for i in order:
        c = meta["chunks"][int(i)]
        stale = False
        if verify:
            if c["file"] not in fresh:
                path = os.path.join(root, c["file"])
                fresh[c["file"]] = {ch.sha for ch in
                                    chunk_file(path, c["file"])} \
                    if os.path.exists(path) else set()
            stale = c["sha256"] not in fresh[c["file"]]
        out.append((float(scores[i]), c, stale))
    return out


def fuzzy_findings(root, claims, config, cap=15):
    """WARN-tier probes: superseded/retracted claim statements against
    the chunk matrix. A high-scoring chunk with no exact pattern match
    is a candidate PARAPHRASED stale restatement — verify by hand."""
    import re
    from .check import Finding, _match_values
    threshold = config.get("fuzzy_threshold", 0.18)
    prof = profiler()
    meta, mat = load_index(root)
    findings = []
    stale_claims = [c for c in claims
                    if c.status in ("superseded", "retracted")
                    and c.statement]
    texts = None
    for claim in stale_claims:
        probe = claim.statement.replace("{value}", str(claim.value))
        q = prof.unit_profile(probe)
        scores = np.real(mat.conj() @ q)
        for i in np.argsort(scores)[::-1][:3]:
            if scores[i] < threshold:
                break
            c = meta["chunks"][int(i)]
            if texts is None:
                texts = {}
            key = (c["file"], tuple(c["lines"]))
            if key not in texts:
                path = os.path.join(root, c["file"])
                match = [ch for ch in chunk_file(path, c["file"])
                         if ch.sha == c["sha256"]] \
                    if os.path.exists(path) else []
                texts[key] = match[0].text if match else None
            text = texts[key]
            if text is None:
                continue  # stale index row; reindex will resolve
            if _match_values(claim, text):
                continue  # exact tier already owns this site
            findings.append(Finding(
                "WARN", "fuzzy-paraphrase", claim.id, c["file"],
                c["lines"][0],
                "possible paraphrased restatement (score %.2f) — verify"
                % scores[i]))
    return findings[:cap]
