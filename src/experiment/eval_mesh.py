"""One frozen evaluation set shared by FDM, PINN, and BSDE.

Wealth is log-uniform on [w_min, w_max], which is the PINN training window.
Time stops at 0.95 T so the score is not the terminal condition.
The draw is seed 12345 and is stored next to the ledger.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.experiment.closed_form import MARKET

EVAL_SEED = 12345
N_POINTS = 10000
W_MIN = 0.5
W_MAX = 2.0
T_MAX_FRAC = 0.95


def make_points(n: int = N_POINTS, seed: int = EVAL_SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, T_MAX_FRAC * MARKET["T"], size=n)
    log_w = rng.uniform(np.log(W_MIN), np.log(W_MAX), size=n)
    w = np.exp(log_w)
    return {"t": t.astype(np.float64), "w": w.astype(np.float64)}


def load_or_create(directory: Path) -> dict[str, np.ndarray]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "eval_points.npz"
    if path.exists():
        data = np.load(path)
        return {"t": data["t"], "w": data["w"]}
    points = make_points()
    np.savez(path, **points, seed=np.array([EVAL_SEED]), n=np.array([N_POINTS]))
    return points
