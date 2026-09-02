# PRISM-EGT TMLR manuscript

`paper.tex` is an anonymous protocol draft using the official TMLR LaTeX style
retrieved from <https://github.com/JmlrOrg/tmlr-style-file> on 2026-09-02.
The unmodified style files are redistributed under the adjacent
`TMLR_STYLE_LICENSE`.

Build from this directory with a TeX distribution that provides `pdflatex` and
`bibtex`:

```text
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

The source was compiled with MiKTeX 25.12 and visually inspected page by page
on 2026-09-02. The local review copy is written to
`output/pdf/PRISM-EGT-TMLR-draft.pdf`; generated PDF and auxiliary files are
ignored because CI rebuilds the manuscript from source. Scientific result
gates remain separate from successful typesetting.

The manuscript must remain anonymous for review. Only use TMLR's `accepted`
option after acceptance or `preprint` option for a de-anonymized preprint. Read
`SUBMISSION_CHECKLIST.md` before changing the paper status.
