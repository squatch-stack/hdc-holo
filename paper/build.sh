#!/bin/sh
# Typeset the paper: draft.md -> main.tex (md2tex.py) -> main.pdf (tectonic).
#
#     paper/build.sh
#
# md2tex.py is the canonical converter — it decides span by span which
# backticked text is mathematics and which proper nouns carry citations,
# which a general Markdown-to-LaTeX pass would get wrong — and CI asserts
# that the committed main.tex and abstract.txt match the draft. tectonic
# then compiles main.tex, fetching the LaTeX packages the document uses on
# first run; it comes from Homebrew (`brew install tectonic`).
set -e
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

python3 md2tex.py
tectonic --keep-logs main.tex
echo "wrote $(pwd)/main.pdf ($(du -h main.pdf | cut -f1))"
