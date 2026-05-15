---
name: Benchmark EEG literature results
description: Published results on the Vlaar 2018 wrist-EEG benchmark — Volterra, NARMAX, NARMAX-HNN, JADE-STACK — with key hyperparameters and comparison to our experimental results as of 2026-05-06
type: reference
---
## Original Paper

Vlaar et al., "Modeling the Nonlinear Cortical Response in EEG Evoked by Wrist Joint Manipulation," IEEE TNSRE vol.26 no.1, Jan 2018. DOI: 10.1109/TNSRE.2017.2751650. PMID: 28920904.

Companion: Vlaar et al. 2017, IEEE TNSRE vol.25 no.5. DOI: 10.1109/TNSRE.2016.2579118.

Dataset DOI: 10.4121/uuid:176d8f78-d9fd-491e-90e7-9370e249b701 (4TU).
Benchmark website: https://www.nonlinearbenchmark.org/benchmarks/cortical-responses

## Dataset Specs (confirmed)

- 10 subjects, 7 realizations each, 2048 Hz raw → 256 Hz downsampled
- 30s usable segments, input delay = 5 samples (~19.5 ms)
- Input: wrist handle angle; Output: top-SNR ICA component
- Normalization: zero-mean, scale by mean-of-per-realization-std, per subject

## Published Results

### Free-run / open-loop simulation (most comparable to our eval)

| Paper | Model | Free-run VAF | Params |
|---|---|---|---|
| Vlaar 2018 | Volterra 2nd-order (subject-specific) | **~46%** | ~46 |
| Gu 2021 | Volterra common-struct 20 terms | 23.3% | 20 |

### One-step-ahead (OSA) prediction

| Paper | Model | OSA VAF | Params |
|---|---|---|---|
| Santos 2023 (Sensors) | JADE-STACK ensemble | **94.5% +/- 1.5%** | many |
| Gu 2021 (IEEE TBME) | Subject-specific NARMAX | 94.3% +/- 1.6% | 20-25 |
| Gu 2021 | Common-struct NARMAX | 93.9% +/- 1.5% | 20-25 |
| Gu 2018 (Frontiers) | NARMAX-HNN | 92.3% +/- 1.6% | 19 |

### 3-step-ahead prediction (~12 ms at 256 Hz)

| Paper | Model | 3-step VAF |
|---|---|---|
| Gu 2018 | NARMAX-HNN | **69.4% +/- 11.9%** |
| Santos 2023 | JADE-STACK | 67.5% +/- 7.4% |
| Gu 2021 | Subject-specific NARMAX | 54.8% +/- 14.1% |
| Gu 2021 | Common-struct NARMAX | 47.1% +/- 13.3% |

## Key Literature Hyperparameters

| Parameter | Gu 2021 NARMAX | Gu 2018 HNN |
|---|---|---|
| Input lags (nu) | **20** (~78 ms) | d1=4, d2=8, n1=n2=2 |
| Output lags (ny) | **5** (~20 ms) | 5 |
| Poly degree | 2 | sigmoid |
| Split | 6 train / 1 test | 6 train / 1 test |

## Our Results (2026-05-06, full free-run simulation)

> Table predates the plateau-curriculum + PolyNARMAX-OSA patch (2026-05-15). Re-run the notebook on the full LOSO loop to refresh; the post-patch PolyNARMAX row will gain an `OSA VAF` column.


| Model | NRMSE | VAF | Params |
|---|---|---|---|
| NNARX (ensemble=3, dilated 0-64) | 0.858 | 26.4% | 5,785 |
| NNARX-FROLS | 0.877 | 23.1% | 5,655 |
| RNN h=64 | 0.961 | 8.1% | 4,353 |
| LSTM h=64 | 0.999 | 0.2% | 17,217 |
| GRU h=64 | 0.999 | 0.3% | 12,929 |

**Gap**: Our best (NNARX 26% VAF) is well below Volterra baseline (46%). Recurrent models essentially non-functional in simulation mode.

## Identified Improvement Priorities

Status as of 2026-05-15 (post-commit `6eb7df3` + plateau-curriculum patch).

1. **Recurrent models** -- `done`. Truncated BPTT (chunk = `rec_kstep_schedule` final stage, currently 256), teacher forcing with scheduled sampling via `rec_y_noise_std`, gradient clipping (`rec_grad_clip=1.0`) all wired in `train_seq` (cell 20). Free-run val NRMSE still ~0.99 -- model is well-conditioned but the recurrent family does not fit this signal at the tested capacities.
2. **NNARX tap structure** -- `done`. `narx_u_taps = tuple(range(0, 20))`, `narx_y_taps = tuple(range(1, 6))` in `CFG` (cell 4) -- dense match to Gu 2021.
3. **Curriculum convergence gate** -- `done`. `narx_kstep_advance = "plateau"` (and the recurrent twin `rec_kstep_advance`) advance to the next `k` only after `narx_kstep_min_epochs[stage]` have elapsed AND val sim NRMSE has plateaued for `narx_kstep_plateau_checks` consecutive checks (relative tol `narx_kstep_plateau_tol`). Fixed-epoch schedule is still available via `kstep_advance="fixed"` for back-compat.
4. **PolyNARMAX stability** -- `done`. `run_poly_narmax` (cell 36) now reports both OSA (`steps_ahead=1`, comparable to Gu 2021 / Santos 2023 ~94 % VAF) and a divergence-guarded free-run (`steps_ahead=None`, per-sequence clipped at `diverge_factor=50 * std(y_seq)`). Result dict gains `nrmse_osa` / `vaf_osa` / `free_run_diverged` alongside the existing `nrmse_sim` / `vaf_sim` keys, so downstream Phase-3 tables are unchanged. `run_rnn` / `run_lstm` / `run_gru` likewise compute OSA via the existing `eval_seq` (teacher-forced) and emit the same `nrmse_osa` / `vaf_osa` fields, so RNN-family results are directly comparable to PolyNARMAX OSA and to Gu 2021.
5. **LOSO CV** -- `done`. Outer loop iterates `loso_test_subject` across all `S` subjects, gated by `RUN_LOSO_CV` (cell 43); aggregated mean +/- std written to `results/loso_folds.json`.
6. **3-step-ahead VAF** -- `done`. Literature reports 3-step VAF as a separate axis (Gu 2018 NARMAX-HNN 69.4 +/- 11.9 %, Santos 2023 JADE-STACK 67.5 +/- 7.4 %, Gu 2021 NARMAX 47-55 %). All six runners (`run_nnarx`, `run_nnarx_frols`, `run_rnn`, `run_lstm`, `run_gru`, `run_poly_narmax`) now also emit `nrmse_3step` / `vaf_3step` via `eval_3step_narx` (NNARX, cell 22) / `kstep_seq` (recurrent, cell 22) / `poly.predict(steps_ahead=3)` with the same divergence guard as free-run (PolyNARMAX). `k` defaults to 3; horizon-correct definition: condition on true `y` up to `t-k`, true `u` up to `t`, feed predictions back for the intermediate steps.

## Citing Papers (full references)

- Gu et al. 2018: "A Novel Approach for Modeling Neural Responses to Joint Perturbations Using the NARMAX Method and a Hierarchical Neural Network," Frontiers Comput Neurosci 12:96. DOI: 10.3389/fncom.2018.00096
- Gu et al. 2021: "Nonlinear Modeling of Cortical Responses to Mechanical Wrist Perturbations Using the NARMAX Method," IEEE TBME 68(3):948-958. DOI: 10.1109/TBME.2020.3013545
- Santos et al. 2023: "Decoding Electroencephalography Signal Response by Stacking Ensemble Learning and Adaptive Differential Evolution," Sensors 23(16):7049. PMID: 37631586
- Gu et al. 2025: "Decoding the cortical responses to mechanical wrist perturbations: A two-step shared structure NARX method," Artif Intell Med, 2025
- Bakels et al. 2025: "Accurate linear modeling of EEG-based cortical activity during a passive motor task with input," arXiv:2510.02596

