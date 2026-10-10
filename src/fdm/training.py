"""Run the FDM solve and save the grid.

There is one solve, so that solve is the saved model:
  saved_models/best_model.npz
"""
import json
from pathlib import Path

import numpy as np

try:
    from fdm_main import MertonFDMSolver, MertonParams
except ImportError:
    from src.fdm.fdm_main import MertonFDMSolver, MertonParams


def train():
    here = Path(__file__).resolve().parent
    with open(here / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    params = MertonParams(
        gamma=cfg["gamma"],
        mu=cfg["mu"],
        sigma=cfg["sigma"],
        r=cfg["rate"],
        T=cfg["T"],
    )
    solver = MertonFDMSolver(
        params,
        Nx=int(cfg["Nx"]),
        Nt=int(cfg["Nt"]),
        x_min=cfg["x_min"],
        x_max=cfg["x_max"],
        theta=cfg["theta"],
        pi_min=cfg["pi_min"],
        pi_max=cfg["pi_max"],
    )
    solver.solve(verbose=False)

    out = here / "saved_models"
    out.mkdir(exist_ok=True)
    path = out / "best_model.npz"
    np.savez(
        path,
        t=solver.t_grid,
        x=solver.x,
        W=solver.W_grid,
        V=solver.V,
        pi=solver.pi_fdm,
        gamma=params.gamma,
        mu=params.mu,
        sigma=params.sigma,
        r=params.r,
        T=params.T,
    )
    print(f"Best model saved to {path}")
    return path


if __name__ == "__main__":
    train()
