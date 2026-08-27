"""Markdown chunking for the fuzzy index — one normalization, shared
with the checker, so fuzzy hits and exact findings cite identical text.

Paragraphs from `normalize` merge under their heading until a chunk
reaches MIN_CHARS (~a short paragraph), capped at MAX_CHARS so no
chunk dominates its profile. Every chunk records its file, heading,
line span, and the sha256 of its normalized text — retrieval re-reads
the working tree and a sha mismatch means the index is stale.
"""

import hashlib
from dataclasses import dataclass

from .normalize import normalize_file

__all__ = [
    "MAX_CHARS",
    "MIN_CHARS",
    "Chunk",
    "chunk_file",
    "chunk_surfaces",
]

MIN_CHARS = 200
MAX_CHARS = 1200


@dataclass
class Chunk:
    file: str
    heading: str
    line_start: int
    line_end: int
    text: str

    @property
    def sha(self):
        return hashlib.sha256(self.text.encode()).hexdigest()


def chunk_file(path, rel=None):
    rel = rel or path
    chunks, heading = [], ""
    buf, start, end = [], None, None

    def flush():
        nonlocal buf, start, end
        if buf:
            chunks.append(Chunk(rel, heading, start, end, " ".join(buf)))
        buf, start, end = [], None, None

    for par in normalize_file(path):
        text = par.text.strip()
        if text.startswith("#"):
            flush()
            heading = text.lstrip("# ").strip()
            continue
        if start is None:
            start = par.line_start
        end = par.line_end
        buf.append(text)
        size = sum(len(b) for b in buf)
        if size >= MIN_CHARS:
            if size > MAX_CHARS and len(buf) > 1:
                last = buf.pop()
                le = end
                flush()
                buf, start, end = [last], par.line_start, le
            else:
                flush()
    flush()
    return chunks


def chunk_surfaces(root, surfaces):
    """Chunks for every markdown surface, in stable path order."""
    import os
    out = []
    for rel in surfaces:
        if rel.endswith(".md"):
            out.extend(chunk_file(os.path.join(root, rel), rel))
    return out
