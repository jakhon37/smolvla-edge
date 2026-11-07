# Report

`main.tex` is a technical report on this repo's results, in IEEE's two-column conference format.

## Build

```bash
cd paper
latexmk -pdf main.tex        # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Or with [tectonic](https://tectonic-typesetting.github.io), a single binary that fetches the TeX packages it needs: `tectonic main.tex`. The draft was checked that way: tectonic runs XeTeX, where `main.tex` sets TeX Gyre Termes in place of the Times that pdfLaTeX and arXiv use, so the line breaks match.

On Overleaf, upload `main.tex`, `refs.bib` and the `figures/` folder; IEEEtran is built in.

## Figures

The PDFs in `figures/` are drawn from `results/*.json` by the same script as the README's figures, sized for a column (3.5 in) or the page width (7.16 in):

```bash
~/venvs/smolvla/bin/python tools/make_figures.py --paper paper/figures
```

Every number in the text and tables comes from `results/` and the guides in `docs/`.

## Submitting

[`ARXIV.md`](ARXIV.md) has the steps, the form fields and `make_arxiv.sh`, which packs `main.tex`, `main.bbl`, `refs.bib` and the figures for arXiv.
