"""The LaTeX build of the paper (paper/md2tex.py -> paper/main.tex).

main.tex is generated and committed, which is the same arrangement the
claims registry has with the prose: convenient to read, and worthless
unless something proves it still matches its source. The first test is
that proof. The rest are the structural checks a compiler would make,
because no TeX toolchain is available in CI and a broken .tex would
otherwise only be discovered at submission.
"""

import importlib.util
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
TEX = os.path.join(PAPER, "main.tex")


def _md2tex():
    spec = importlib.util.spec_from_file_location(
        "md2tex", os.path.join(PAPER, "md2tex.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _body():
    """main.tex with comments stripped, for structural accounting."""
    with open(TEX) as f:
        return re.sub(r"(?<!\\)%.*", "", f.read())


def test_main_tex_matches_its_markdown_source():
    """The drift gate: editing draft.md without regenerating main.tex
    leaves the two telling different stories, and the .tex is what gets
    submitted."""
    m = _md2tex()
    regenerated = m.convert(os.path.join(PAPER, "draft.md"),
                            os.path.join(PAPER, "references.bib"))
    with open(TEX) as f:
        committed = f.read()
    assert regenerated == committed, (
        "paper/main.tex is stale — run `python paper/md2tex.py`")


def test_environments_and_braces_balance():
    body = _body()
    stack = []
    for m in re.finditer(r"\\(begin|end)\{(\w+\*?)\}", body):
        if m.group(1) == "begin":
            stack.append(m.group(2))
        else:
            assert stack and stack[-1] == m.group(2), \
                "\\end{%s} does not close the open environment" % m.group(2)
            stack.pop()
    assert not stack, "unclosed environments: %s" % stack
    bare = re.sub(r"\\[{}]", "", body)
    assert bare.count("{") == bare.count("}"), "unbalanced braces"
    assert len(re.findall(r"(?<!\\)\$", bare)) % 2 == 0, \
        "odd number of inline-math delimiters"


def test_every_citation_resolves_and_every_reference_is_cited():
    body = _body()
    items = set(re.findall(r"\\bibitem\{([^}]+)\}", body))
    cited = {k for group in re.findall(r"\\cite\{([^}]+)\}", body)
             for k in group.split(",")}
    assert not cited - items, "cited but absent: %s" % sorted(cited - items)
    # thebibliography prints every entry whether cited or not, so an
    # uncited one is a reference list the paper does not support
    assert not items - cited, "listed but never cited: %s" % sorted(items - cited)


def test_cross_references_resolve():
    body = _body()
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
    assert not refs - labels, "dangling: %s" % sorted(refs - labels)


def test_every_graphic_resolves_through_graphicspath():
    body = _body()
    spec = re.search(r"\\graphicspath\{(.*?)\}\s*\n", body).group(1)
    dirs = re.findall(r"\{([^}]*)\}", spec)
    names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
    assert len(names) >= 10
    for g in names:
        assert any(os.path.isfile(os.path.join(PAPER, d, g + ext))
                   for d in dirs for ext in (".png", ".pdf", "")), \
            "figure not found through \\graphicspath: %s" % g


def test_bib_entry_years_agree_with_their_keys():
    """A venue-typed entry must carry the VENUE year. Mip-Splatting is
    the case that motivated this: its arXiv preprint is 2023 and the
    CVPR paper is 2024, and taking the year from the arXiv API put
    `year = {2023}` under a key and a booktitle that both said 2024."""
    with open(os.path.join(PAPER, "references.bib")) as f:
        src = f.read()
    body = "\n".join(ln for ln in src.split("\n")
                     if not ln.lstrip().startswith("%"))
    for typ, key, fields in re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}",
                                       body, re.S):
        if typ != "inproceedings":
            continue
        in_key = re.search(r"(\d{4})", key)
        in_field = re.search(r"year\s*=\s*\{(\d{4})\}", fields)
        if in_key and in_field:
            assert in_key.group(1) == in_field.group(1), (
                "%s: key says %s, year field says %s"
                % (key, in_key.group(1), in_field.group(1)))


def test_plain_text_abstract_matches_its_source():
    m = _md2tex()
    with open(os.path.join(PAPER, "abstract.txt")) as f:
        committed = f.read()
    assert m.abstract_text(os.path.join(PAPER, "draft.md")) == committed, (
        "paper/abstract.txt is stale — run `python paper/md2tex.py`")


def test_plain_text_abstract_meets_arxivs_constraints():
    """arXiv's abstract field is plain text with a hard length cap: it
    renders no markup and no non-ASCII, and pasting the paper's own
    abstract there produces mojibake in the first thing a reader sees."""
    with open(os.path.join(PAPER, "abstract.txt")) as f:
        text = f.read()
    assert text.isascii(), "non-ASCII: %s" % sorted(
        {c for c in text if ord(c) > 127})
    assert len(text.strip()) <= 1920, "abstract exceeds arXiv's 1920 chars"
    for markup in ("`", "**", "](", "\\"):
        assert markup not in text, "markup survived: %r" % markup
