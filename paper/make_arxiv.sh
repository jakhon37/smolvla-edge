#!/usr/bin/env bash
# Builds dist/smolvla-edge-arxiv.tar.gz (and the same as dist/smolvla-edge-source.zip), the files arXiv needs: main.tex, main.bbl and the figures main.tex uses.
# arXiv compiles with pdfLaTeX and does not run BibTeX, so main.bbl must come from a build of this main.tex.
# refs.bib comes along too, so that the source also builds where BibTeX does run (tectonic always runs it).
#
#   bash paper/make_arxiv.sh     # builds with tectonic if it is installed; otherwise run latexmk -pdf main.tex first
set -euo pipefail
cd "$(dirname "$0")"
if command -v tectonic >/dev/null; then
  tectonic --keep-intermediates main.tex
fi
if [ ! main.bbl -nt main.tex ] || [ ! main.bbl -nt refs.bib ]; then
  echo "main.bbl is missing or older than main.tex or refs.bib: build the paper first" >&2
  exit 1
fi
rm -rf dist
mkdir -p dist/src/figures
cp main.tex main.bbl refs.bib dist/src/
grep -o '\\includegraphics\(\[[^]]*\]\)\?{[^}]*}' main.tex | sed 's/.*{\(.*\)}/\1/' | while read -r f; do
  cp "figures/$f.pdf" dist/src/figures/
done
tar -C dist/src -czf dist/smolvla-edge-arxiv.tar.gz .
(cd dist/src && python3 -m zipfile -c ../smolvla-edge-source.zip main.tex main.bbl refs.bib figures)  # Preprints.org: zip only
tar -tzvf dist/smolvla-edge-arxiv.tar.gz
python3 -m zipfile -l dist/smolvla-edge-source.zip
