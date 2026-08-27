"""Assemble a flat arXiv submission directory from paper/.

arXiv unpacks a submission into ONE directory, so the repo's
`../results/x.png` paths cannot survive it. main.tex is written with
bare graphic names and a \\graphicspath listing both repo locations
plus `./`, which means the identical file compiles in-tree AND flat —
this script just gathers what it references.

The file list is READ OUT OF main.tex rather than maintained here, so
a figure added to the paper cannot be forgotten at submission time.

    python paper/make_arxiv_bundle.py [outdir]

Then upload the directory's contents (or a zip of them) to arXiv. No
BibTeX pass is needed: the bibliography is inline.
"""

import os
import re
import shutil
import sys

PAPER = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PAPER)


def graphics(tex):
    spec = re.search(r"\\graphicspath\{(.*?)\}\s*\n", tex).group(1)
    dirs = re.findall(r"\{([^}]*)\}", spec)
    found = []
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                           tex):
        for d in dirs:
            for ext in (".png", ".pdf", ".jpg", ""):
                p = os.path.join(PAPER, d, name + ext)
                if os.path.isfile(p):
                    found.append(p)
                    break
            else:
                continue
            break
        else:
            raise SystemExit("figure not found: %s" % name)
    return found


def main(out):
    with open(os.path.join(PAPER, "main.tex")) as f:
        tex = f.read()
    os.makedirs(out, exist_ok=True)
    shutil.copy2(os.path.join(PAPER, "main.tex"), out)
    figs = graphics(tex)
    for p in figs:
        shutil.copy2(p, out)
    total = sum(os.path.getsize(os.path.join(out, f))
                for f in os.listdir(out))
    print("%s: main.tex + %d figures, %.1f MB"
          % (out, len(figs), total / 1e6))
    print("arXiv's limit is 50 MB; no BibTeX pass required.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.join(ROOT, "out", "arxiv"))
