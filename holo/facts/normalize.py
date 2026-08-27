"""Surface normalization: markdown, mermaid, and docstrings to matchable text.

The stale-claim checker and the fuzzy chunker share ONE normalization so
findings and search hits cite identical text. The pipeline exists because
of two measured bug classes: prose in this repo is hard-wrapped at ~70
columns (numbers split across lines: "13\nmin"), and a real historical
staleness bug lived in a mermaid node label ("3 scale bands" after the
code moved to 4). Markdown emphasis, links, tables, and <br/> are
stripped; fenced code survives verbatim (commands carry claims too);
Python surfaces contribute only their docstrings (via ast, with real
line numbers).

Pragmas ride as HTML comments, invisible in rendered markdown:
  <!-- claims: allow tests.count@0.1.0 -->   legitimizes a superseded value
  <!-- claims: ignore -->                    suppresses unregistered-number warns
A pragma applies to the paragraph it sits in (or the one that follows a
standalone pragma line).
"""

import ast
import re
from dataclasses import dataclass, field

__all__ = [
    "Paragraph",
    "canon",
    "figure_refs",
    "front_matter",
    "normalize_file",
    "normalize_markdown",
    "normalize_plain",
    "normalize_python",
]

_PRAGMA = re.compile(r"<!--\s*claims:\s*([^>]*?)\s*-->")
_MERMAID_QUOTED = re.compile(r'"([^"]+)"')
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_FIG_REF = re.compile(r"(?:results|out|assets)/[\w.\-]+\.(?:png|gif|svg|jpe?g)")

_UNICODE_MAP = str.maketrans({
    "–": "-", "—": "-", "−": "-",   # dashes/minus
    "×": "x", "→": ">", " ": " ",   # times, arrow, nbsp
    "“": '"', "”": '"', "‘": "'", "’": "'",
})

_WORD_NUMS = {"zero": "0", "one": "1", "two": "2", "three": "3",
              "four": "4", "five": "5", "six": "6", "seven": "7",
              "eight": "8", "nine": "9", "ten": "10"}


@dataclass
class Paragraph:
    file: str
    line_start: int
    line_end: int
    kind: str            # prose | verbatim | mermaid | table | docstring
    text: str
    pragmas: set = field(default_factory=set)


def canon(value):
    """Canonical form for value comparison: lowercase, unicode folded,
    number words to digits, whitespace/tildes collapsed."""
    v = str(value).translate(_UNICODE_MAP).lower().strip()
    v = _WORD_NUMS.get(v, v)
    v = v.replace("~", "").replace(" ", "")
    return v


def _strip_markup(line):
    line = _PRAGMA.sub(" ", line)
    line = _MD_LINK.sub(r"\1", line)
    line = _HTML_TAG.sub(" ", line)
    line = line.replace("`", "").replace("*", "")
    line = line.translate(_UNICODE_MAP)
    return line.strip()


def _pragmas_in(text):
    return {m.group(1).strip() for m in _PRAGMA.finditer(text)}


def normalize_markdown(text, path):
    """Markdown -> paragraphs. Fences split out (mermaid gets label
    extraction), table rows stand alone, wrapped prose lines rejoin."""
    lines = text.split("\n")
    out = []
    pending_pragmas = set()
    buf, buf_start = [], None
    i = 0

    def flush(end_line, kind="prose"):
        nonlocal buf, buf_start, pending_pragmas
        if buf:
            raw = "\n".join(buf)
            pragmas = _pragmas_in(raw) | pending_pragmas
            pending_pragmas = set()
            joined = " ".join(_strip_markup(l) for l in buf if _strip_markup(l))
            if joined:
                out.append(Paragraph(path, buf_start, end_line, kind,
                                     joined, pragmas))
        buf, buf_start = [], None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        fence = stripped.startswith("```")
        if fence:
            flush(i)
            lang = stripped[3:].strip().lower()
            block, start = [], i + 2  # 1-indexed first content line
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            end = i  # closing fence line (1-indexed = i+1; content ends at i)
            if lang == "mermaid":
                labels = []
                for bl in block:
                    labels.extend(_MERMAID_QUOTED.findall(bl))
                    if bl.strip().lower().startswith("note "):
                        labels.append(bl.split(":", 1)[-1])
                textm = " ".join(_strip_markup(l) for l in labels if l.strip())
                if textm:
                    out.append(Paragraph(path, start, end, "mermaid", textm))
            else:
                textv = "\n".join(block).translate(_UNICODE_MAP)
                if textv.strip():
                    out.append(Paragraph(path, start, end, "verbatim", textv))
            i += 1
            continue
        if stripped.startswith("|"):
            flush(i)
            row = _strip_markup(stripped.replace("|", " "))
            if row and set(row) - set("- :"):
                out.append(Paragraph(path, i + 1, i + 1, "table", row,
                                     _pragmas_in(stripped)))
            i += 1
            continue
        if not stripped:
            flush(i)
            i += 1
            continue
        if _PRAGMA.fullmatch(stripped):
            pending_pragmas |= _pragmas_in(stripped)
            i += 1
            continue
        # a new top-level list item starts its own paragraph — bullet
        # runs have no blank lines between items, and merging them makes
        # marker proximity, line attribution, and chunking all coarser
        # than the prose actually is
        if re.match(r"(?:[-*]|\d+\.)\s", stripped) and \
                not line.startswith((" ", "\t")) and buf:
            flush(i)
        if buf_start is None:
            buf_start = i + 1
        buf.append(line)
        i += 1
    flush(len(lines))
    return out


def normalize_python(text, path):
    """Python -> one paragraph stream per docstring (module/class/def)."""
    out = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    nodes = [tree] + [n for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                        ast.ClassDef))]
    for node in nodes:
        doc = ast.get_docstring(node, clean=True)
        if not doc:
            continue
        lineno = getattr(node, "lineno", 1)
        for block in doc.split("\n\n"):
            joined = " ".join(_strip_markup(l) for l in block.split("\n")
                              if _strip_markup(l))
            if joined:
                out.append(Paragraph(path, lineno, lineno, "docstring",
                                     joined, _pragmas_in(block)))
    return out


def normalize_plain(text, path):
    """Metadata files (toml/cff/json): one verbatim paragraph per line."""
    out = []
    for i, line in enumerate(text.split("\n")):
        s = line.translate(_UNICODE_MAP).strip()
        if s:
            out.append(Paragraph(path, i + 1, i + 1, "verbatim", s))
    return out


def normalize_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if path.endswith(".py"):
        return normalize_python(text, path)
    if path.endswith((".md", ".markdown")):
        return normalize_markdown(text, path)
    return normalize_plain(text, path)


def figure_refs(text):
    """Figure paths referenced anywhere in raw text."""
    return sorted(set(_FIG_REF.findall(text)))


def front_matter(path):
    """Restricted flat front-matter: `key: value` and `- item` lists
    between leading --- fences. Deliberately not YAML."""
    out, key = {}, None
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
    except OSError:
        return out
    if not lines or lines[0].strip() != "---":
        return out
    for line in lines[1:]:
        s = line.strip()
        if s == "---":
            break
        if s.startswith("- ") and key:
            out.setdefault(key, [])
            if isinstance(out[key], list):
                out[key].append(s[2:].strip())
        elif ":" in s:
            key, _, val = s.partition(":")
            key, val = key.strip(), val.strip()
            out[key] = val if val else []
    return out
