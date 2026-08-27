"""Convert paper/draft.md to paper/main.tex.

Tailored to this one document, not a general converter: the value here
is the per-construct judgment (which inline spans are mathematics and
which are literals, which proper nouns take a citation), and a general
markdown-to-LaTeX pass would get exactly those wrong.
"""
import os
import re
import sys

# Inline `spans`: mathematics, decided one at a time.
MATH = {
    "σ": r"\sigma",
    "Σ⁻¹": r"\Sigma^{-1}",
    "e^{iWp}": r"e^{iWp}",
    "N(0, Σ⁻¹)": r"\mathcal{N}(0, \Sigma^{-1})",
    "σ ~ √(N·R / 2d)": r"\sigma \sim \sqrt{N R / 2d}",
    "p": r"p",
    "w ~ N(0, Σ⁻¹)": r"w \sim \mathcal{N}(0, \Sigma^{-1})",
    "exp(−½(p−q)ᵀΣ⁻¹(p−q))":
        r"\exp\!\left(-\tfrac{1}{2}(p-q)^{\mathsf{T}}\Sigma^{-1}(p-q)\right)",
    "d": r"d",
    "0 ± 1/√(2d)": r"0 \pm 1/\sqrt{2d}",
    "d^-0.50": r"d^{-0.50}",
    "σ²I": r"\sigma^2 I",
}
# Inline `spans`: literals.
TT = {".tex", "claims/registry.jsonl", "references.bib",
      "what_is_at(p)", "where_is(label)", "(dim, seed)",
      "translate_bundle", "footprint_blur", "sh_flip_x180"}

# Proper nouns that carry a citation, cited once per section.
CITES = [
    # Anchored on phrases the prose actually uses. Several works are
    # described rather than named — the anti-aliasing pair, the shift
    # property, the model-merging paper — so the anchor has to be the
    # description, not a proper noun that is not in the text.
    ("Plate's Holographic Reduced Representations",
     "plate1995hrr,plate2003hrr"),
    ("Kanerva's Sparse Distributed Memory",
     "kanerva1988sdm,kanerva2009hdc"),
    ("Frady et al.", "frady2021vfa"),
    ("Komer and Eliasmith", "komer2019fractional"),
    ("Rahimi and Recht", "rahimi2007random"),
    ("Kleyko et al.'s surveys",
     "kleyko2021survey1,kleyko2021survey2"),
    ("Quantized-phase FHRR", "snyder2026qfhrr"),
    ("quantized-phase FHRR", "snyder2026qfhrr"),
    ("sampling rate", "yu2024mipsplatting"),
    ("pixel window analytically", "liang2024analytic"),
    ("shift property", "voelker2021ssp"),
    ("model merging", "gillespie2026crdtmerge"),
    ("compression literature", "ali2025compression"),
    ("Fourier-extension", "adcock2012fourier"),
    ("3D Gaussian splatting", "kerbl20233dgs"),
    ("VSA-OGM", "snyder2024vsaogm"),
    ("GVKF", "song2024gvkf"),
    ("CryoSplat", "chen2025cryosplat"),
    ("R2-Gaussian", "zha2024r2gaussian"),
    ("HyperSpace", "snyder2026hyperspace"),
]

UNI = [
    ("—", "---"), ("–", "--"), ("−", "$-$"), ("±", "$\\pm$"),
    ("×", "$\\times$"), ("·", "$\\cdot$"), ("√", "$\\sqrt{\\ }$"),
    ("σ", "$\\sigma$"), ("Σ", "$\\Sigma$"), ("²", "$^2$"),
    ("¹", "$^1$"), ("⁷", "$^7$"), ("⁻", "$^-$"), ("½", "$\\tfrac{1}{2}$"),
    ("ᵀ", "$^{\\mathsf{T}}$"), ("§", "\\S"), ("’", "'"), ("‘", "'"),
    ("“", "``"), ("”", "''"),
]


def esc(t):
    """Escape LaTeX specials in ordinary prose."""
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        t = t.replace(a, b)
    for a, b in UNI:
        t = t.replace(a, b)
    return t


def inline(t):
    """Prose -> LaTeX, protecting code spans from escaping."""
    out, i = [], 0
    for m in re.finditer(r"`([^`]+)`", t):
        out.append(esc(t[i:m.start()]))
        c = m.group(1)
        if c in MATH:
            out.append("$" + MATH[c] + "$")
        else:
            out.append(r"\texttt{%s}" % c.replace("_", r"\_")
                       .replace("{", r"\{").replace("}", r"\}"))
        i = m.end()
    out.append(esc(t[i:]))
    s = "".join(out)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\\emph{\1}", s)
    s = re.sub(r"\[([^\]]+)\]\((?:https?://[^)]+)\)", r"\1", s)   # links
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)                # rel links
    # straight quotes render as two right-quotes in LaTeX
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    # prose figure references become real cross-references, so the
    # numbering cannot drift from the float order
    return re.sub(r"\bFigure (\d+)\b", r"Figure~\\ref{fig:\1}", s)


def wrap(t, width=72):
    """Wrap prose so the generated file diffs like the markdown does.
    Never breaks inside a math span or a LaTeX command argument."""
    words, lines, cur = t.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _read(path):
    with open(path) as f:
        return f.read()


def _bib_field(fields, name):
    m = re.search(r"^\s*%s\s*=\s*\{(.*?)\},?\s*$" % name,
                  fields, re.S | re.M)
    return " ".join(m.group(1).split()) if m else None


def _bibitem(key, fields):
    """One \\bibitem, formatted from a parsed .bib entry."""
    def dot(x):
        # NOT esc()'d: a .bib field is LaTeX source already. Escaping it
        # turns Leimk{\"u}hler into visible backslashes and breaks the
        # math in R$^2$-Gaussian. (Checked: no field here contains a
        # bare &, %, _ or # that would need escaping.)
        return x if x.endswith(".") else x + "."

    f = _bib_field
    venue = (f(fields, "journal") or f(fields, "booktitle")
             or f(fields, "publisher") or "")
    eprint = f(fields, "eprint")
    if venue.startswith("arXiv preprint"):
        venue, eprint = "arXiv:%s" % eprint, None
    bits = [dot(f(fields, "author")),
            "\\newblock %s" % dot(f(fields, "title"))]
    tail = "\\newblock \\emph{%s}" % esc(venue) if venue else ""
    vol, num, pg = (f(fields, "volume"), f(fields, "number"),
                    f(fields, "pages"))
    if vol:
        tail += " \\textbf{%s}" % esc(vol)
        tail += "(%s)" % num if num else ""
        tail += ":%s" % pg if pg else ""
    year = f(fields, "year")
    tail += ", %s" % year if year else ""
    if tail:
        bits.append(tail + ".")
    if eprint:
        bits.append("\\newblock arXiv:%s." % eprint)
    doi = f(fields, "doi")
    if doi:
        bits.append("\\newblock \\texttt{doi:%s}." % esc(doi))
    return "\\bibitem{%s}\n%s" % (key, "\n".join(wrap(b) for b in bits))


def bibliography(bib_path):
    """thebibliography from references.bib.

    Inline rather than a \\bibliography{} call on purpose: arXiv's
    processor may or may not run a bibliography pass depending on what
    is uploaded, and no toolchain here can generate a .bbl to hand it.
    An inline environment needs no bibliography pass at all, so the
    document compiles the same way everywhere.
    """
    body = "\n".join(ln for ln in _read(bib_path).split("\n")
                     if not ln.lstrip().startswith("%"))
    entries = re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", body, re.S)
    items = [_bibitem(key, fields) for _typ, key, fields in entries]
    return ("\\begin{thebibliography}{%d}\n\n" % len(items)
            + "\n\n".join(items) + "\n\n\\end{thebibliography}\n")


def table(lines):
    """One markdown pipe table -> booktabs tabular."""
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")]
            for ln in lines if not re.match(r"^\|[\s:|-]+\|$", ln.strip())]
    align_src = [ln for ln in lines if re.match(r"^\|[\s:|-]+\|$", ln.strip())]
    spec = "l" * len(rows[0])
    if align_src:
        cols = align_src[0].strip().strip("|").split("|")
        spec = "".join("r" if c.strip().endswith(":") else "l" for c in cols)
    head = " & ".join(inline(c) for c in rows[0]) + r" \\"
    body = "\n".join(" & ".join(inline(c) for c in r) + r" \\"
                     for r in rows[1:])
    return ("\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{%s}\n"
            "\\toprule\n%s\n\\midrule\n%s\n\\bottomrule\n"
            "\\end{tabular}\n\\end{table}\n" % (spec, head, body))


PREAMBLE = r"""% Generated from paper/draft.md by paper/md2tex.py — edit the
% markdown and regenerate; do not hand-edit this file.
%
% Plain article class with a deliberately small package set: arXiv does
% not have your style files, and every package is a way for a remote
% compile to fail. Bibliography is inline (see md2tex.bibliography), so
% no BibTeX/Biber pass is required anywhere.
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}

% Figures live outside paper/ in the repo, but an arXiv bundle is flat.
% Naming them without a directory and listing both locations here means
% the same source compiles in-repo AND in a flattened submission.
\graphicspath{{../results/}{../out/}{./}}

\title{Holographic Scene Representation:\\Gaussian Splats as Hypervectors}
\author{Squatch Stack}
\date{}

\begin{document}
\maketitle
"""


# arXiv's submission form takes the abstract as PLAIN TEXT: it renders
# no backticks, no markdown emphasis, and no non-ASCII glyphs. Pasting
# the paper's abstract there produces mojibake in the one piece of the
# paper every reader sees first. These are the ASCII readings of the
# three expressions the abstract uses.
ASCII_MATH = {
    "e^{iWp}": "exp(i W p)",
    "N(0, Σ⁻¹)": "N(0, Sigma^-1)",
    "σ ~ √(N·R / 2d)": "sigma ~ sqrt(N R / 2d)",
}
ASCII_UNI = [("—", " -- "), ("–", "-"), ("−", "-"), ("±", "+/-"),
             ("×", "x"), ("·", " "), ("√", "sqrt"), ("σ", "sigma"),
             ("Σ", "Sigma"), ("²", "^2"), ("¹", "^1"), ("⁻", "^-"),
             ("½", "1/2"), ("ᵀ", "^T"), ("§", "Section "),
             ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')]


def abstract_text(src):
    """The abstract as ASCII plain text, for arXiv's submission form."""
    text = _read(src)
    body = text.split("## Abstract", 1)[1].split("\n## ", 1)[0].strip()
    for code, plain in ASCII_MATH.items():
        body = body.replace("`%s`" % code, plain)
    body = re.sub(r"`([^`]+)`", r"\1", body)
    body = re.sub(r"\*\*([^*]+)\*\*", r"\1", body)
    body = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", body)
    for a, b in ASCII_UNI:
        body = body.replace(a, b)
    body = re.sub(r"[ \t]+", " ", " ".join(body.split("\n")))
    return wrap(body.strip(), 78) + "\n"


def _figure(lines, i, m, cited_here):
    """One markdown figure plus its `**Figure N.**` caption paragraph."""
    num = m.group(1)
    stem = os.path.splitext(os.path.basename(m.group(2)))[0]
    i += 1
    while i < len(lines) and not lines[i].startswith("**Figure "):
        i += 1
    cap = []
    while i < len(lines) and lines[i].strip():
        cap.append(lines[i])
        i += 1
    capt = re.sub(r"^\*\*Figure \d+\.\*\*\s*", "", " ".join(cap).strip())
    blk = ("\n\\begin{figure}[htbp]\n\\centering\n"
           "\\includegraphics[width=\\linewidth]{%s}\n"
           "\\caption{%s}\n\\label{fig:%s}\n\\end{figure}\n"
           % (stem, wrap(cite(inline(capt), cited_here)), num))
    return i, blk


def _table_block(lines, i):
    blk = []
    while i < len(lines) and lines[i].startswith("|"):
        blk.append(lines[i])
        i += 1
    return i, "\n" + table(blk)


def convert(src, bib):
    text = _read(src)
    text = text.split("\n## References")[0]
    # drop the markdown front matter: title line and draft status block,
    # which describe the draft rather than belonging to the paper
    text = text[text.index("## Abstract"):]

    out, i = [PREAMBLE], 0
    lines = text.split("\n")
    cited_here = set()
    while i < len(lines):
        ln = lines[i]

        if ln.startswith("## Abstract"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("## "):
                buf.append(lines[i])
                i += 1
            out.append("\\begin{abstract}\n%s\n\\end{abstract}\n"
                       % wrap(inline(" ".join(buf).split() and
                                     " ".join(buf).strip())))
            continue

        if ln.startswith("## "):
            cited_here = set()
            t = re.sub(r"^##\s+\d+\.\s*", "", ln)
            out.append("\n\\section{%s}\n" % cite(inline(t), cited_here))
            i += 1
            continue

        if ln.startswith("### "):
            t = re.sub(r"^###\s+[\d.]+\s*", "", ln)
            out.append("\n\\subsection{%s}\n" % cite(inline(t), cited_here))
            i += 1
            continue

        m = re.match(r"^!\[Figure (\d+)\]\(([^)]+)\)", ln)
        if m:
            i, blk = _figure(lines, i, m, cited_here)
            out.append(blk)
            continue

        if ln.startswith("|"):
            i, blk = _table_block(lines, i)
            out.append(blk)
            continue

        if not ln.strip():
            i += 1
            continue

        para = []
        while i < len(lines) and lines[i].strip() \
                and not lines[i].startswith(("#", "|", "![")):
            para.append(lines[i])
            i += 1
        out.append("\n" + wrap(cite(inline(" ".join(para)), cited_here))
                   + "\n")

    out.append("\n" + bibliography(bib))
    out.append("\n\\end{document}\n")
    return "".join(out)


def cite(s, seen):
    """First mention of a cited work in each section gets a \\cite."""
    for name, key in CITES:
        if name in seen:
            continue
        pat = re.escape(name)
        m = re.search(pat, s)
        if m:
            at = _outside_command(s, m.end())
            s = s[:at] + "~\\cite{%s}" % key + s[at:]
            seen.add(name)
    return s


def _outside_command(s, at):
    """Push an insertion point out of any \\textbf{...} or \\emph{...} it
    would land inside. A citation typeset in bold as part of a run-in
    heading is legal but wrong-looking, and prose gets edited, so the
    converter should not depend on nobody ever naming a cited work
    inside emphasis."""
    depth, i = 0, at
    while i < len(s):
        if s[i] == "{" and s[i - 1] != "\\":
            depth += 1
        elif s[i] == "}" and s[i - 1] != "\\":
            if depth == 0:
                head = s[:at]
                open_at = head.rfind("{")
                if open_at > 0 and re.search(r"\\(textbf|emph)\{$",
                                            head[:open_at + 1]):
                    return i + 1
                return at
            depth -= 1
        i += 1
    return at


def main(root):
    """Regenerate paper/main.tex from paper/draft.md under `root`."""
    here = os.path.join(root, "paper")
    tex = convert(os.path.join(here, "draft.md"),
                  os.path.join(here, "references.bib"))
    with open(os.path.join(here, "main.tex"), "w") as f:
        f.write(tex)
    with open(os.path.join(here, "abstract.txt"), "w") as f:
        f.write(abstract_text(os.path.join(here, "draft.md")))
    return tex


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(root)
    print("wrote", os.path.join(root, "paper", "main.tex"))
