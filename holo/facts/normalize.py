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


def _mermaid_labels(block):
    """Node/edge label text — where a stale claim can hide in a diagram."""
    labels = []
    for line in block:
        labels.extend(_MERMAID_QUOTED.findall(line))
        if line.strip().lower().startswith("note "):
            labels.append(line.split(":", 1)[-1])
    return labels


def _read_fence(lines, i, path):
    """Consume a fenced block: (paragraph or None, index past the fence)."""
    lang = lines[i].strip()[3:].strip().lower()
    block, first = [], i + 2          # 1-indexed first content line
    i += 1
    while i < len(lines) and not lines[i].strip().startswith("```"):
        block.append(lines[i])
        i += 1
    last = i                           # content ends before the close fence
    if lang == "mermaid":
        text = " ".join(_strip_markup(ln)
                        for ln in _mermaid_labels(block) if ln.strip())
        par = Paragraph(path, first, last, "mermaid", text) if text else None
    else:
        text = "\n".join(block).translate(_UNICODE_MAP)
        par = (Paragraph(path, first, last, "verbatim", text)
               if text.strip() else None)
    return par, i + 1


def _table_row(stripped, i, path):
    """A `|`-delimited row, unless it is the ---|--- separator."""
    row = _strip_markup(stripped.replace("|", " "))
    if row and set(row) - set("- :"):
        return Paragraph(path, i + 1, i + 1, "table", row,
                         _pragmas_in(stripped))
    return None


def _starts_list_item(line, stripped):
    """A new top-level bullet starts its own paragraph — bullet runs have
    no blank lines between items, and merging them makes marker
    proximity, line attribution, and chunking all coarser than the prose
    actually is."""
    return (bool(re.match(r"(?:[-*]|\d+\.)\s", stripped))
            and not line.startswith((" ", "\t")))


class _Prose:
    """The wrapped-prose accumulator: lines join until something ends
    the paragraph (blank line, fence, table row, or a new list item)."""

    def __init__(self, path, out):
        self.path = path
        self.out = out
        self.buf = []
        self.start = None
        self.pending_pragmas = set()

    def add(self, line, lineno):
        if self.start is None:
            self.start = lineno + 1
        self.buf.append(line)

    def flush(self, end_line):
        if self.buf:
            raw = "\n".join(self.buf)
            pragmas = _pragmas_in(raw) | self.pending_pragmas
            self.pending_pragmas = set()
            joined = " ".join(_strip_markup(ln) for ln in self.buf
                              if _strip_markup(ln))
            if joined:
                self.out.append(Paragraph(self.path, self.start, end_line,
                                          "prose", joined, pragmas))
        self.buf, self.start = [], None


def normalize_markdown(text, path):
    """Markdown -> paragraphs. Fences split out (mermaid gets label
    extraction), table rows stand alone, wrapped prose lines rejoin."""
    lines = text.split("\n")
    out = []
    prose = _Prose(path, out)
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            prose.flush(i)
            par, i = _read_fence(lines, i, path)
            if par:
                out.append(par)
            continue
        if stripped.startswith("|"):
            prose.flush(i)
            par = _table_row(stripped, i, path)
            if par:
                out.append(par)
        elif not stripped:
            prose.flush(i)
        elif _PRAGMA.fullmatch(stripped):
            prose.pending_pragmas |= _pragmas_in(stripped)
        else:
            if _starts_list_item(line, stripped) and prose.buf:
                prose.flush(i)
            prose.add(line, i)
        i += 1
    prose.flush(len(lines))
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
            joined = " ".join(_strip_markup(ln) for ln in block.split("\n")
                              if _strip_markup(ln))
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


def _front_matter_lines(path):
    """The lines between the leading --- fences, or [] when absent."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
    except OSError:
        return []
    if not lines or lines[0].strip() != "---":
        return []
    out = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        out.append(line)
    return out


def front_matter(path):
    """Restricted flat front-matter: `key: value` and `- item` lists
    between leading --- fences. Deliberately not YAML — this parses the
    subset the knowledge-base pages use and nothing more."""
    out, key = {}, None
    for line in _front_matter_lines(path):
        s = line.strip()
        if s.startswith("- ") and key:
            out.setdefault(key, [])
            if isinstance(out[key], list):
                out[key].append(s[2:].strip())
        elif ":" in s:
            key, _, val = s.partition(":")
            key, val = key.strip(), val.strip()
            out[key] = val if val else []
    return out
