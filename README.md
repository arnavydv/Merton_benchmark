# Merton benchmark

This folder compares four ways to solve one investment problem. The investor splits money between a safe account and Apple stock. The safe rate is a fixed number. The stock numbers come from Apple prices.

The write-up is the PDF:

[paper/merton_gbm_benchmark.pdf](paper/merton_gbm_benchmark.pdf)

## How to use it

Open a terminal in this folder. The folder must be the one that contains `run_one.py`. Then install the libraries and run one job.

```text
pip install -r requirements.txt
python run_one.py --list
python run_one.py --exp E1
```

`E1` is the fast grid check. It should finish in a few seconds. The other jobs are listed below. A job that is already finished is skipped, so running it again does not train a second time.

You do not have to retrain anything to read the results. They are already saved.

| What you want | What to open or run |
| --- | --- |
| Read the paper | `paper/merton_gbm_benchmark.pdf` |
| See the saved scores | `results/v2/ledger.jsonl` |
| Rebuild the plots from those scores | `python -m src.experiment.v2.make_figures` |
| Check that the Apple numbers still match | `python -m src.experiment.v2.check_market` |
| A short test of the code | `python run_one.py --exp smoke` |

`smoke` trains for only a few steps. It is a test. It is not a result in the paper.

Rebuild the PDF from the `paper` folder if you change the text:

```text
pdflatex merton_gbm_benchmark
bibtex merton_gbm_benchmark
pdflatex merton_gbm_benchmark
pdflatex merton_gbm_benchmark
```

## Words used below

- **Wealth** is the investor's money, called `W`.
- **Drift** `mu` is the expected growth rate in the stock price equation. It is not the average of daily log changes.
- **Mean log return** `m` is that average. For this Apple sample, `m = 0.201649`. The drift is `mu = m + sigma^2 / 2 = 0.234564`.
- **Volatility** `sigma = 0.256573` is how much the daily log changes spread out, scaled to one year.
- **pi\*** is the best fraction of wealth to put in the stock when the formula is known. Here it is `0.600837`.
- **Grid** means a finite-difference table. The computer fills a sheet of wealth and time values.
- **Value network** is a neural net that learns the investor's value `V`.
- **Deep BSDE** is a neural net that learns the stock fraction by walking many random price paths forward.

The price window is Apple adjusted closes from 23 May 2023 through 22 May 2026. That is 753 sessions and 752 daily log changes. The safe rate `r = 0.0368` is typed in by hand. Risk aversion `gamma = 5`. The horizon is one year. Starting wealth is 1.

## The ten jobs

Run each one with `python run_one.py --exp NAME`.

| Name | What it does | How heavy it is |
| --- | --- | --- |
| E1 | Grid with 500 wealth steps and 252 time steps. Checks that the stock fraction is close to pi\*. | Seconds |
| E2 | Same grid at nine coarser and finer sizes. | Seconds |
| E3 | Value network that may depend on time and wealth. Seeds 0, 1, 2, 3, 4. 2000 steps. | Long, about half an hour per seed on a GPU |
| E4 | Value network that depends on time only. Seeds 0 to 4. The stock fraction is then automatic. Judge it by the value error. | Medium |
| E5 | Deep BSDE. Seeds 0 to 4. 2000 steps, 1024 paths, 64 time steps. | Long |
| E6 | Grid again at two other risk aversions and two other volatilities. | Seconds |
| E7 | Stock drift also depends on a second random factor. Seed 0. 800 steps. | Medium |
| E8 | Same factor model, seed 1. | Medium |
| E9 | Stock fraction is not allowed to go above 0.40. Seed 0. | Long |
| E10 | Same cap, seed 1. | Long |

Add `--epochs 30` only when you want a short trial, for example `python run_one.py --exp E3 --epochs 30`. The paper uses the full budgets above.

## What the saved numbers say

These are the main errors on a shared set of 10,000 test points. Smaller is closer to the known answer.

| Method | Error in the stock fraction |
| --- | --- |
| Grid E1 | 0.000252 |
| Time-only value network E4 | about 0.00000003 (this is forced by the formula, not a real discovery) |
| Time-and-wealth value network E3 | 0.00162 on average over 5 seeds |
| Deep BSDE E5 | 0.0215 on average over 5 seeds |

The deep BSDE is less exact on the stock fraction. Its starting value `Y0` is still within about 0.00005 of the true value at time 0 and wealth 1.

In the factor model, the grid stock fraction at time 0 and factor 0 is 0.499, not 0.601. The cap of 0.40 is below 0.601, so it changes the answer. The network matches that capped grid.

## Repository structure

```text
.
|-- README.md                 this file
|-- BENCHMARK.md              longer notes on seeds, budgets, and pass checks
|-- LICENSE
|-- requirements.txt          Python libraries
|-- run_one.py                the command you type to run one job
|-- .gitignore                files Git should skip
|-- paper/                    the manuscript
|-- results/v2/               saved runs and plots
`-- src/                      the code
```

### `run_one.py`

This is the front door. It prints what the job is, then calls the trainer. It also prints the score files when the job ends.

### `BENCHMARK.md`

Same jobs as above, with the exact seeds, the mesh, the path count, and the rule that stops E1 if the grid is too far from pi\*.

### `requirements.txt`

`torch`, `numpy`, `scipy`, `matplotlib`, `pandas`, `psutil`, `tabulate`, and `yfinance`. `yfinance` is used only by the market check, to download Apple prices again.

### `paper/`

| File | What it is |
| --- | --- |
| `merton_gbm_benchmark.pdf` | The paper. Start here. |
| `merton_gbm_benchmark.tex` | The LaTeX source of that paper. |
| `refs_gbm.bib` | The reference list: Merton, Raissi, Sirignano, E, Han, Jentzen, Kim and Omberg. |
| `figures/` | Plots and tables copied into the PDF. |

Plots in `paper/figures/`:

| File | What it shows |
| --- | --- |
| `surface_closed.png` | True value and true stock fraction as 3D surfaces. |
| `surface_fdm.png` | The same surfaces from the grid. |
| `surface_pinn_error.png` | 3D picture of the value-network error. |
| `surface_pinn_error_clip.txt` | The color limits used on that error picture. |
| `surface_factor_policy.png` | Stock fraction when the extra factor is turned on. |
| `policy_slices.png` | Stock fraction against wealth at two times. |
| `policy_pinn_slice.png` | Value-network stock fraction near time 0. |
| `error_heatmap_fdm.png` | Grid value error on the time-wealth plane. |
| `error_heatmap_pinn.png` | Value-network error on the same plane. |
| `training_loss.png` | Loss during training for seed 0. |
| `table_market.csv` | `m`, `sigma`, `mu`, `pi*`, and the true value. |
| `table_main.csv` | Main errors and their spread across seeds. |
| `table_fdm_convergence.csv` | Grid errors at each size in E2. |
| `table_sensitivity.csv` | E6 rows. |
| `table_factor.csv` | E7 and E8. |
| `table_constraint.csv` | E9 and E10. |
| `manifest.json` | Short checksum list for those files. |

### `results/v2/`

This is the lab notebook. The paper numbers come from here.

| Path | What it is |
| --- | --- |
| `ledger.jsonl` | One JSON line per finished job. This is the master log. |
| `market_check.json` | The last comparison of the frozen Apple numbers with a new download. |
| `figures/` | Plots made by `make_figures` before they are copied into `paper/figures`. |
| `runs/` | One folder per job and seed. |

A run folder is named `E3_s0` for experiment E3, seed 0. Inside a run you may see:

| File | What it is |
| --- | --- |
| `row.json` | The score for that run: errors, time taken, settings. |
| `best_model.pt` | Saved neural-network weights. |
| `loss_history.npy` | Loss at each training step. |
| `prediction.npz` | Predictions on the 10,000 test points. |
| `surface.npz` | A regular grid used to draw the 3D pictures. |
| `fdm_prediction.npz` or `fdm_grid.npz` | The grid solution, used as the reference in E9, E10, E7, and E8. |
| `pinn.pt`, `bsde.pt` | Weights for the factor-model networks. |

E1 through E10 are present. E3, E4, and E5 have seeds 0 to 4. E7 is seed 0. E8 and E10 are seed 1. E9 is seed 0. E2 and E6 use the grid size as the folder id, for example `E2_s500252` means 500 wealth steps and 252 time steps.

### `src/experiment/v2/`

This is the code the paper uses.

| File | What it is |
| --- | --- |
| `market.json` | Frozen Apple numbers. Every new run reads this file. |
| `check_market.py` | Recomputes `m`, `sigma`, and `mu`, and stops if they do not match. |
| `run_benchmark.py` | Trains and scores E1 to E10. `run_one.py` calls this. |
| `crra_pinn.py` | The two value-network shapes. |
| `deep_bsde.py` | The forward deep BSDE step. |
| `factor.py` | The extra-factor grid. |
| `make_figures.py` | Builds the plots and CSV tables from the ledger. |

### `src/fdm/`

The wealth-time grid. `fdm_main.py` holds the solver. `config.json` is an old settings file and is not the file the new study reads. The new study reads `src/experiment/v2/market.json`. `training.py` is an older entry point.

### `src/pinn/`

Pieces shared by the value networks: where the training points are drawn (`collocation_points.py`), the loss (`loss.py`), and an older network (`neural_network.py`). `config.json` here is also an old settings file. The new runs set their own point counts in `run_benchmark.py`: 25,000 interior points and 8,000 terminal points.

Interior points are spread over time in `[0, 1]` and wealth in `[0.5, 2]`. Terminal points sit at time `T` and use the utility `W` to the power `1 - gamma`, divided by `1 - gamma`.

### `src/bsde/`

Older path code: Brownian motion, the wealth update, networks, and a different training loss. The paper does not train this older loss. The paper's deep BSDE is `src/experiment/v2/deep_bsde.py`.

### `src/experiment/` without `v2`

Shared helpers: the closed-form stock fraction and value (`closed_form.py`), error summaries (`metrics.py`), and older runners used before the corrected-drift study. The commands in this README do not need those older runners.

### `src/common/`

An empty package marker so Python can import the project.

## A note on the old settings files

`src/fdm/config.json`, `src/pinn/config.json`, and `src/bsde/config.json` still store `0.201649` in the drift slot. That number is `m`, not `mu`. Leave those files as history. Do not train from them if you want results that match the paper.
