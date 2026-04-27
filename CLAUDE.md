# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Coursework for *Machine Learning for Vision and Multimedia* (Polito, 2025/26), project §2.5 — **System Identification of a Nonlinear Dynamical System**. The task is SISO modeling `y[k] = f(u[·], y[·])` where `u` is the wrist handle angle and `y` is the top-SNR ICA component of the EEG response (Vlaar et al., IEEE TNSRE 2018). The brief is in [specs.md](specs.md) and the original PDF.

The deliverable runs three phases inside a single notebook: Phase 1 trains NNARX / simple RNN / LSTM / GRU; Phase 2 sweeps architecture sizes; Phase 3 compares simulation NRMSE/VAF, training time, parameter count, and FLOPs/sample.

## Working notebook vs. legacy notebook

There are two notebooks at the root and they are **not** different versions of the same file:

- [project-si-mlicv.ipynb](project-si-mlicv.ipynb) — the active, lean working notebook. Edit this one.
- [Project_SI_MLiCV.ipynb](Project_SI_MLiCV.ipynb) — large (~2 MB), carries embedded outputs from earlier runs; treat as a reference snapshot, not the source of truth.

When asked to make changes, default to `project-si-mlicv.ipynb` unless the user names the other one.

## Environment

A local venv lives at `.venv/` (Python 3.14, PyTorch 2.11, NumPy 2.3, SciPy 1.17, scikit-learn 1.8, matplotlib, ipykernel, jupytext). Activate with `source .venv/bin/activate` or invoke directly via `.venv/bin/python` / `.venv/bin/jupyter`.

Run the notebook headlessly when verifying changes:

```
.venv/bin/jupyter nbconvert --to notebook --execute project-si-mlicv.ipynb --output /tmp/out.ipynb
```

Phase 1 + Phase 2 sweep can take a while on CPU; CUDA is auto-detected (`DEVICE` in cell 2).

## Architecture

Everything in the notebook is driven from a single top-level `CFG` dict (cell 4). Phase 2 sweeps work by `dict(CFG)`-copying and overriding fields — preserve this convention; do not introduce a parallel config object.

Pipeline stages, in notebook order:

1. **Loaders** (`load_small`, `load_medium`) return `[S, M, N]` arrays. `load_medium` is a faithful Python port of [Benchmark_EEG_medium/Load_Plot_EEG_2.m](Benchmark_EEG_medium/Load_Plot_EEG_2.m): average over periods, downsample by 8 (2048 → 256 Hz), per-subject zero-mean and scale-by-mean-of-per-realization-std, then `circshift(u, 5, axis=-1)` to delay the input. If you change preprocessing, keep it consistent with the MATLAB reference so results stay comparable to the paper.
2. **Splits** (`make_split`, `make_loso_split`) both flatten to `[num_sequences, N]` so all downstream code is split-agnostic. `"within"` splits realizations per subject; `"loso"` holds out one subject entirely. The optional outer LOSO CV loop (last cell, gated by `RUN_LOSO_CV`) iterates `loso_test_subject` over all `S` subjects.
3. **Models**:
   - `NNARX` (cell 14) is a plain MLP over a regressor `[u[k..k-nu+1], y[k-1..k-ny]]`. Trained teacher-forced; **simulated** at test time via `NNARX.simulate`, which uses the first `max(nu, ny)` true-`y` samples as warm-up and feeds predictions back. Always evaluate via `eval_narx`, not the teacher-forced loss.
   - `SeqModel` (cell 16) is a thin wrapper around `nn.RNN` / `nn.LSTM` / `nn.GRU` + linear head. Closed-loop by construction — a forward pass on a test sequence already *is* the simulation, no teacher forcing at eval time.
4. **Metrics** — `nrmse`, `vaf`, `count_params`, and a hand-rolled `flops_per_sample` (cell 22). Recurrent FLOPs are computed analytically because `ptflops` doesn't handle the cells well; if you change cell math, update both training and the FLOPs formula together.
5. **Runners** `run_nnarx` / `run_seq` (cell 24) wrap train + simulate + plot + metrics into one call and append a row to the `RESULTS` dict. Phase 2 (cell 34) and Phase 3 (cell 36) consume `RESULTS` / `PHASE2`. Results JSON is written to `results/` (created on demand).

## Data

`Benchmark_EEG_small/Benchmark_EEG_small.mat` (~270 KB, already preprocessed) and `Benchmark_EEG_medium/Benchmark_EEG_medium.mat` (~460 MB, raw `[M, P, N_raw]` per subject) — both selected by `CFG["dataset"]`. `find_mat_file` also looks under `/kaggle/input/...` so the same notebook runs unmodified on Kaggle. Neither `.mat` file should be committed if not already tracked; check `git status` before adding anything from these directories.

## Conventions worth preserving

- All data tensors stay shaped `[S, M, N]` until the splitter flattens them. Keep that invariant — many cells index by `axis=-1` assuming time is last.
- One config dict, one `RESULTS` dict, plotting helpers (`plot_sim`) shared across model families. Resist factoring these into separate modules unless the user asks; the project is graded as a single notebook.
