# Project 2.5 — System Identification of Nonlinear Dynamical Systems

Polito, Machine Learning for Vision and Multimedia 2025/26.
Dataset: Cortical Responses Evoked by Wrist Joint Manipulation (Vlaar et al. 2018).

Everything lives in one notebook: [`Project_SI_MLiCV.ipynb`](Project_SI_MLiCV.ipynb).

## Run on Kaggle

1. Create a new **Kaggle Dataset** from `Benchmark_EEG_small/Benchmark_EEG_small.mat`
   (any name — the notebook scans `/kaggle/input/*/` for the file).
2. Create a new **Kaggle Notebook**, upload `Project_SI_MLiCV.ipynb`, attach the dataset,
   enable **GPU T4 x2** (optional but recommended for the LSTM/GRU sweeps).
3. Run all cells. No extra `pip install` needed — PyTorch, SciPy, Matplotlib
   are preinstalled on Kaggle.

## Run locally

```bash
python -m venv .venv
source .venv/Scripts/activate            # Git Bash on Windows
# or: .venv\Scripts\Activate.ps1          # PowerShell
pip install torch scipy numpy matplotlib jupyter
jupyter lab Project_SI_MLiCV.ipynb
```

The notebook finds `Benchmark_EEG_small/Benchmark_EEG_small.mat` in this repo automatically.

## Notebook structure

| Section | What |
| --- | --- |
| 1–5 | Imports, config dict, data loading (auto local/Kaggle path), split, metrics |
| 6–7 | Model definitions — NNARX (MLP) and `SeqModel` (RNN/LSTM/GRU) |
| 8–10 | Training loops (teacher-forced) and simulation-mode evaluation; param + FLOPs counts |
| 11 | Phase 1 — trains NNARX, RNN, LSTM, GRU end-to-end and plots test simulations |
| 12 | Phase 2 — small LSTM hidden×layers sweep |
| 13 | Phase 3 — comparison table (params, FLOPs, train time, sim NRMSE, VAF) |

All hyperparameters are in the `CFG` dict at the top of the notebook.
