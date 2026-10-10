# Reproducing the corrected-drift benchmark

Work from `merton_comparative_study`. The commands import the package `src`, so a launch from the parent folder fails.

## Environment

```
pip install -r requirements.txt
pip install yfinance
```

The logged study used CUDA when `torch.cuda.is_available()` was true. CPU runs use the same seeds and the same pass bands below. They are not bit-identical.

## Frozen inputs

- `src/experiment/v2/market.json` is the only coefficient file the new study reads.
- `m_log = 0.20164940530246397` is the annualized mean log return of AAPL adjusted closes from 23 May 2023 through 22 May 2026 (753 sessions, 752 log returns, 252-day year). It is not the GBM drift.
- `sigma = 0.2565727586666484` is `sqrt(252)` times the sample standard deviation of those log returns (`ddof=1`).
- `mu = m_log + sigma^2/2 = 0.23456419554737107`.
- `r = 0.0368` is a declared constant, not the Treasury bill index.
- `gamma = 5`, `T = 1`, `W0 = 1`, no consumption in E1–E8.
- Evaluation mesh: 10,000 points, seed 12345, wealth log-uniform on `[0.5, 2]`, time uniform on `[0, 1)`.
- Traded utility: 20,000 paths, 252 steps, seed 7.
- Network budget unless a command overrides it: 2,000 Adam steps, learning rate `10^{-3}`, 25,000 Latin-hypercube collocation points and 8,000 terminal points redrawn every epoch. Interior law: `t` uniform on `[0, T]`, `w` uniform on `[0.5, 2]`. Terminal law: `t = T`, same wealth interval, target `w^(1-gamma)/(1-gamma)`.
- Deep BSDE training paths: 1,024 paths, 64 steps. `Z` is evaluated before the wealth update. The loss is `E[(Y_T - U(W_T))^2] - E[U(W_T)]`. This is the forward terminal-loss scheme. It is not the martingale objective in `src/bsde/`.

Check the identity before training:

```
python -m src.experiment.v2.check_market
```

The check fails if `mu` is not `m + sigma^2/2`, if `pi*` disagrees by more than `10^{-12}`, or if a fresh download moves `m` or `sigma` by more than `10^{-6}`.

## Commands

Smoke test, a few epochs, no paper rows:

```
python -m src.experiment.v2.run_benchmark --exp smoke
```

Logged experiments, one command each. They append to `results/v2/ledger.jsonl` and write `results/v2/runs/<exp>_s<seed>/`.

| ID | Command | What it records |
| --- | --- | --- |
| E1 | `python -m src.experiment.v2.run_benchmark --exp E1` | FDM `500 x 252` on the corrected drift |
| E2 | `python -m src.experiment.v2.run_benchmark --exp E2` | FDM refinement, nine grids |
| E3 | `python -m src.experiment.v2.run_benchmark --exp E3` | Vanilla PINN, seeds 0–4 |
| E4 | `python -m src.experiment.v2.run_benchmark --exp E4` | CRRA ansatz value net, seeds 0–4 |
| E5 | `python -m src.experiment.v2.run_benchmark --exp E5` | Forward deep BSDE, seeds 0–4 |
| E6 | `python -m src.experiment.v2.run_benchmark --exp E6` | FDM at `gamma` in `{3, 7}` and `sigma` scaled by `0.8` and `1.2` |
| E7 | `python -m src.experiment.v2.run_benchmark --exp E7` | OU factor in the drift, grid plus PINN and deep BSDE, seed 0, 800 epochs |
| E8 | `python -m src.experiment.v2.run_benchmark --exp E8` | Same networks, seed 1 |
| E9 | `python -m src.experiment.v2.run_benchmark --exp E9` | Leverage cap `pi in [0, 0.4]`, seed 0, scored on the constrained grid |
| E10 | `python -m src.experiment.v2.run_benchmark --exp E10` | Same cap, seed 1 |

E1 stops if the policy MAE on the shared mesh exceeds `2e-3`, or if `|V(0,1) - closed form|` exceeds `5e-3`.

Figures and CSV tables, without training:

```
python -m src.experiment.v2.make_figures
```

Outputs land in `results/v2/figures/` together with `manifest.json`. Copy the files the manuscript cites into `paper/figures/` before compiling.

## PDF

From `paper/`:

```
pdflatex merton_gbm_benchmark
bibtex merton_gbm_benchmark
pdflatex merton_gbm_benchmark
pdflatex merton_gbm_benchmark
```

## What a matching run should show

These are orders of magnitude, not bit-level tolerances.

- E1 policy MAE against `pi* = 0.600836870221195` is about `10^{-4}`. `V(0,1)` is about `-0.17014`.
- The CRRA ansatz policy MAE is roundoff. The ansatz forces `V_w / (w V_ww) = 1/gamma`. Judge that method by the value error.
- The vanilla PINN policy error stays large relative to the finite-difference policy if the residual training does not identify the weight. The paper reports that error rather than clipping it away.
- The factor grid with `phi = 0` and `rho = 0` matches `exp(A T)` within `5e-2` in the max norm, and the logged run is much tighter than that gate.
- The cap `[0, 0.4]` binds, because the unconstrained weight is about `0.60`. The constrained reference is the grid.
