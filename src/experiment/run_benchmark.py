"""Run a logged FDM / PINN / BSDE benchmark.

From the merton_comparative_study directory:

    python -m src.experiment.run_benchmark --suite smoke
    python -m src.experiment.run_benchmark --suite benchmark

smoke checks the pipeline. benchmark is the suite to cite: one FDM
reference grid, an FDM refinement ladder, and three seeds of PINN and
BSDE under the budgets in SUITES. Every finished run is appended to
results/experiments/ledger.jsonl.
"""
from __future__ import annotations

import argparse
import json
import platform
import traceback
from pathlib import Path

import numpy as np

from src.experiment.closed_form import MARKET, growth_A, pi_star
from src.experiment.eval_mesh import load_or_create
from src.experiment.ledger import Ledger, new_run_id, utc_now
from src.experiment.runners import run_bsde, run_fdm, run_pinn

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "experiments"

# Budgets are the experimental condition. They are intentionally smaller
# than a one-off 5000-epoch training.py run so three seeds finish, and
# they are written into every ledger row.
SUITES = {
    "smoke": {
        "fdm": [{"Nx": 128, "Nt": 64}],
        "pinn": {"epochs": 30, "n_collocation": 512, "n_terminal": 256, "lr": 1e-3, "seeds": [0]},
        "bsde": {"epochs": 15, "num_steps": 16, "num_paths": 64, "lr": 1e-3, "seeds": [0]},
    },
    "benchmark": {
        "fdm": [
            {"Nx": 64, "Nt": 64},
            {"Nx": 128, "Nt": 128},
            {"Nx": 256, "Nt": 252},
            {"Nx": 500, "Nt": 252},
        ],
        "pinn": {
            "epochs": 1500,
            "n_collocation": 4000,
            "n_terminal": 1500,
            "lr": 1e-3,
            "seeds": [0, 1, 2],
        },
        "bsde": {
            "epochs": 800,
            "num_steps": 64,
            "num_paths": 512,
            "lr": 1e-3,
            "seeds": [0, 1, 2],
        },
    },
}


def _base_record(suite: str, method: str, seed, budget: dict, run_id: str) -> dict:
    return {
        "run_id": run_id,
        "suite": suite,
        "method": method,
        "seed": seed,
        "status": "started",
        "started_at": utc_now(),
        "market": dict(MARKET),
        "pi_star_closed": pi_star(),
        "A_closed": growth_A(MARKET["gamma"], MARKET["mu"], MARKET["sigma"], MARKET["rate"]),
        "budget": budget,
        "eval": {
            "n": 10000,
            "seed": 12345,
            "w_min": 0.5,
            "w_max": 2.0,
            "t_max": 0.95,
            "wealth_sampling": "log-uniform",
        },
        "notes": {
            "fdm": "Fully implicit theta=1 log-wealth grid. Policy is the discrete FOC, clipped to [-1, 5].",
            "pinn": "Unsupervised value network. Policy is the FOC of V, reported raw and clipped to [-1, 5]. No pi* penalty.",
            "bsde": "Policy network plus Z. Loss is -E[Y0] + Var(Y0). No pi* penalty. Value is scored only at (t,w)=(0,1).",
        }[method],
        "host": {"python": platform.python_version(), "system": platform.platform()},
    }


def _jobs(suite_name: str):
    spec = SUITES[suite_name]
    jobs = []
    for budget in spec["fdm"]:
        jobs.append(("fdm", None, dict(budget)))
    for seed in spec["pinn"]["seeds"]:
        budget = {k: v for k, v in spec["pinn"].items() if k != "seeds"}
        jobs.append(("pinn", seed, budget))
    for seed in spec["bsde"]["seeds"]:
        budget = {k: v for k, v in spec["bsde"].items() if k != "seeds"}
        jobs.append(("bsde", seed, budget))
    return jobs


def run_job(method: str, seed, budget: dict, points, out_dir: Path) -> dict:
    if method == "fdm":
        return run_fdm(points, budget, out_dir)
    if method == "pinn":
        return run_pinn(points, budget, seed, out_dir)
    if method == "bsde":
        return run_bsde(points, budget, seed, out_dir)
    raise ValueError(method)


def main():
    parser = argparse.ArgumentParser(description="Log an FDM / PINN / BSDE benchmark.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ledger = Ledger(args.out)
    points = load_or_create(args.out)
    jobs = _jobs(args.suite)
    print(f"Suite {args.suite}: {len(jobs)} runs -> {args.out}", flush=True)
    for method, seed, budget in jobs:
        run_id = new_run_id(method, seed)
        record = _base_record(args.suite, method, seed, budget, run_id)
        out_dir = args.out / "runs" / run_id
        print(f"START {run_id} {budget}", flush=True)
        try:
            result = run_job(method, seed, budget, points, out_dir)
            record.update(result)
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
            print(record["traceback"], flush=True)
        path = ledger.append(record)
        pi_mae = (record.get("metrics") or {}).get("pi", {}).get("mae")
        print(f"DONE {run_id} status={record['status']} pi_mae={pi_mae} -> {path}", flush=True)
    print(f"Summary: {args.out / 'summary.csv'}", flush=True)
    write_aggregate(ledger, args.out)


def write_aggregate(ledger: Ledger, out: Path) -> None:
    """Mean and sample std of policy MAE across ok runs of one suite."""
    grouped: dict[tuple, list] = {}
    for row in ledger.read():
        if row.get("status") != "ok":
            continue
        key = (row.get("suite"), row.get("method"))
        mae = (row.get("metrics") or {}).get("pi", {}).get("mae")
        if mae is None:
            continue
        grouped.setdefault(key, []).append(row)
    report = []
    for (suite, method), rows in sorted(grouped.items()):
        maes = np.array([(r["metrics"]["pi"]["mae"]) for r in rows], dtype=float)
        clipped = np.array([
            (r["metrics"].get("pi_clipped") or {}).get("mae", np.nan) for r in rows
        ], dtype=float)
        clocks = np.array([r.get("wall_clock_s", np.nan) for r in rows], dtype=float)
        report.append({
            "suite": suite,
            "method": method,
            "n": int(maes.size),
            "pi_mae_mean": float(maes.mean()),
            "pi_mae_std": float(maes.std(ddof=1)) if maes.size > 1 else 0.0,
            "pi_clipped_mae_mean": float(np.nanmean(clipped)),
            "wall_clock_s_mean": float(np.nanmean(clocks)),
        })
    path = out / "aggregate.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Aggregate: {path}", flush=True)


if __name__ == "__main__":
    main()
