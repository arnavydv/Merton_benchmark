# Corrected-drift Merton benchmark

The manuscript is `paper/merton_gbm_benchmark.pdf`.
The source is `paper/merton_gbm_benchmark.tex` with `paper/refs_gbm.bib`.

Apple adjusted closes, 23 May 2023 through 22 May 2026:

- mean log return `m = 0.20164940530246397` is not the GBM drift
- volatility `sigma = 0.2565727586666484`
- drift `mu = m + sigma^2/2 = 0.23456419554737107`
- `pi* = 0.600836870221195` at `r = 0.0368`, `gamma = 5`

Run one experiment from this directory:

```
python run_one.py --list
python run_one.py --exp E1
```

`E1` through `E10` are already logged in `results/v2/ledger.jsonl`.
Figures are rebuilt with `python -m src.experiment.v2.make_figures`.
Details are in `BENCHMARK.md`.
