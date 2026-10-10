"""Run one corrected-drift experiment and print what it is doing.

Work from this directory (merton_comparative_study):

    python run_one.py --list
    python run_one.py --exp E1
    python run_one.py --exp E2

Each command finishes before the next one starts. A finished (exp, seed)
with the full epoch budget is skipped, so repeating a command is safe.
Outputs:

    results/v2/ledger.jsonl          one JSON row per finished run
    results/v2/runs/<exp>_s<seed>/   weights, loss, prediction, row.json

After a few experiments you can rebuild tables and figures without training:

    python -m src.experiment.v2.make_figures

What the coefficients mean
--------------------------
m     annualized mean of daily log returns. This is NOT the GBM drift.
sigma sqrt(252) * sample standard deviation of those log returns.
mu    m + sigma^2/2. This is the drift in dS/S = mu dt + sigma dB.
pi*   (mu - r) / (gamma * sigma^2), about 0.600837 on the frozen file.

The frozen file is src/experiment/v2/market.json.
r = 0.0368 is a declared constant. gamma = 5, T = 1, W0 = 1.

Experiments
-----------
E1   Finite-difference grid, 500 wealth nodes by 252 time steps.
     Pass band: policy MAE <= 2e-3 and |V(0,1) - closed form| <= 5e-3.
E2   Same grid at Nx in {128, 256, 500} and Nt in {64, 128, 252}.
E3   Power-scaled value PINN, seeds 0,1,2,3,4, 2000 Adam epochs.
     V = exp(g(t, log w)) * w^(1-gamma)/(1-gamma). g may depend on
     wealth, so the weight is not forced to pi*. The CRRA ansatz is
     the special case g = g(t) only.
     25000 interior Latin-hypercube points and 8000 terminal points
     are redrawn every epoch. Time is uniform on [0, 1], wealth is
     uniform on [0.5, 2], and the terminal target is U(w) at t = T.
E4   CRRA ansatz V = exp(g(t)) * w^(1-gamma)/(1-gamma), seeds 0-4.
     Policy error is roundoff: the ansatz forces V_w/(w V_ww) = 1/gamma.
     Read the value error.
E5   Forward deep BSDE, seeds 0-4, 2000 epochs, 1024 paths, 64 steps.
     Z is taken at the wealth before the Brownian step.
     Loss is E[(Y_T - U)^2] - E[U]. This is the Han-Jentzen-E terminal
     loss plus a utility term so the policy can move. It is not the
     martingale objective in src/bsde/.
E6   Finite differences at gamma in {3, 7} and at sigma scaled by 0.8 and 1.2.
E7   One OU factor inside the drift, correlated with the stock, seed 0,
     800 epochs. Reference is the (t, Y) grid, not the constant pi*.
E8   The same factor networks, seed 1.
E9   Leverage cap pi in [0, 0.4], which binds because pi* is about 0.60.
     Seed 0. The vanilla PINN is scored against the constrained grid.
E10  The same cap, seed 1.

Shared evaluation for the one-dimensional runs: 10000 mesh points, seed
12345; traded utility on 20000 paths, 252 steps, seed 7.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LESSONS = {
    "E1": "Solve the log-wealth HJB on a 500 by 252 grid using mu = m + sigma^2/2. The script aborts if the policy is not close to pi*.",
    "E2": "Repeat the grid at nine resolutions. Policy error should fall when Nx grows and should barely move when only Nt grows, because pi* is constant in time.",
    "E3": "Train V = exp(g(t, log w)) * U-shape. g can depend on wealth, so this is not the CRRA ansatz. Five seeds. Policy comes from the first-order condition.",
    "E4": "Train only the time factor g(t) in the CRRA shape. A tiny policy MAE here does not mean the method discovered pi*. The value MAE does.",
    "E5": "Train Y0, Z, and a policy network by matching U(W_T) and rewarding terminal utility. Compare policy MAE and the gap |Y0 - V(0,1)| with the grid.",
    "E6": "Four extra grids away from the Apple point. Each row has its own pi*.",
    "E7": "Add a mean-reverting factor to the drift with correlation 0.45. The hedge term depends on f_y/f. Seed 0, 800 epochs, scored on the factor grid.",
    "E8": "Same factor problem, seed 1, so the factor gap is not a single draw.",
    "E9": "Cap the weight at 0.4. Unconstrained pi* is about 0.60, so the cap binds. Score the network against the constrained grid.",
    "E10": "Same cap, seed 1.",
}


def main():
    parser = argparse.ArgumentParser(description="Run one Merton benchmark experiment.")
    parser.add_argument("--exp", default=None, help="E1 ... E10, or smoke")
    parser.add_argument("--list", action="store_true", help="Print the menu and exit")
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch budget")
    args = parser.parse_args()
    if args.list or not args.exp:
        print(__doc__)
        return
    exp = args.exp
    print("=" * 72)
    print(exp)
    print(LESSONS.get(exp, "Smoke test: a few epochs of every one-dimensional solver, no paper claim."))
    print("Ledger: results/v2/ledger.jsonl")
    print("=" * 72)
    from src.experiment.v2.run_benchmark import main as _unused  # noqa: F401
    import sys
    sys.argv = ["run_benchmark", "--exp", exp]
    if args.epochs is not None:
        sys.argv += ["--epochs", str(args.epochs)]
    from src.experiment.v2 import run_benchmark
    run_benchmark.main()
    root = Path("results/v2/runs")
    rows = sorted(root.glob(f"{exp}_s*/row.json"))
    print("-" * 72)
    print(f"{exp} wrote {len(rows)} row file(s).")
    for path in rows:
        row = json.loads(path.read_text(encoding="utf-8"))
        pi = (row.get("pi") or row.get("pi_vs_grid") or row.get("pinn_pi") or {}).get("mae")
        print(f"  {path.parent.name}: status={row.get('status')} epochs={row.get('epochs')} pi_mae={pi}")
    print("Next: python run_one.py --list")
    print("Figures from whatever is already logged: python -m src.experiment.v2.make_figures")


if __name__ == "__main__":
    main()
