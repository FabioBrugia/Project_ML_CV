# Paper

LaTeX source for the project report.

## File

- `paper.tex` — IEEEtran 2-column conference paper, ~8 pages.

## Compile

Need TeX Live / MikTeX. Standard chain:

```
pdflatex paper.tex
pdflatex paper.tex      # 2nd pass for refs
```

No `.bib` file: bibliography is inline (`\begin{thebibliography}`).

## Figures

Currently 3 placeholders (`\fbox`) for:
- `figures/phase2_sweep.pdf` — Phase 2 sweep, VAF vs hidden size.
- `figures/pareto.pdf` — Pareto VAF vs FLOPs.
- `figures/sim_overlay.pdf` — Free-run overlay y_true vs NNARX-FROLS and LSTM.

Generate from the notebook, save as PDF under `paper/figures/`, then uncomment the `\includegraph
ics` lines and delete the `\fbox` stubs.

## Numbers source

Table 1 (Phase 1) values come from `../phase1_metrics.json`. If you rerun Phase 1, regenerate the table from that file.
