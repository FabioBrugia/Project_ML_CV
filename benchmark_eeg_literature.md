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

| Model | NRMSE | VAF | Params |
|---|---|---|---|
| NNARX (ensemble=3, dilated 0-64) | 0.858 | 26.4% | 5,785 |
| NNARX-FROLS | 0.877 | 23.1% | 5,655 |
| RNN h=64 | 0.961 | 8.1% | 4,353 |
| LSTM h=64 | 0.999 | 0.2% | 17,217 |
| GRU h=64 | 0.999 | 0.3% | 12,929 |

**Gap**: Our best (NNARX 26% VAF) is well below Volterra baseline (46%). Recurrent models essentially non-functional in simulation mode.

## Identified Improvement Priorities

1. **Recurrent models broken**: need truncated BPTT (256-512 steps), teacher forcing + scheduled sampling, gradient clipping
2. **NNARX tap structure**: literature uses dense nu=20 ny=5; our dilated taps skip important 0-20 range lags
3. **Curriculum too aggressive**: k=1 phase too short (15 epochs), model not converged before jumping to k=5
4. **PolyNARMAX divergent**: unstable poles in free-run, needs stability check/clipping
5. **Need LOSO CV**: literature reports mean +/- std across all 10 subjects

## Citing Papers (full references)

- Gu et al. 2018: "A Novel Approach for Modeling Neural Responses to Joint Perturbations Using the NARMAX Method and a Hierarchical Neural Network," Frontiers Comput Neurosci 12:96. DOI: 10.3389/fncom.2018.00096
- Gu et al. 2021: "Nonlinear Modeling of Cortical Responses to Mechanical Wrist Perturbations Using the NARMAX Method," IEEE TBME 68(3):948-958. DOI: 10.1109/TBME.2020.3013545
- Santos et al. 2023: "Decoding Electroencephalography Signal Response by Stacking Ensemble Learning and Adaptive Differential Evolution," Sensors 23(16):7049. PMID: 37631586
- Gu et al. 2025: "Decoding the cortical responses to mechanical wrist perturbations: A two-step shared structure NARX method," Artif Intell Med, 2025
- Bakels et al. 2025: "Accurate linear modeling of EEG-based cortical activity during a passive motor task with input," arXiv:2510.02596
su

