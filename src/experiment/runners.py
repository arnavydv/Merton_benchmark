"""Train or solve one method and score it on the shared mesh.

PINN policy is the first-order condition applied to V, not a trained
control. BSDE policy is the network output. BSDE does not return V on
the mesh; the only value number is Y_0 at the common initial wealth.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from src.bsde.neural_networks import Z_net, pi_star as PiNet
from src.bsde.training import training_loop
from src.experiment.closed_form import MARKET, policy as closed_pi, value as closed_value
from src.experiment.metrics import error_block, relative_error_block
from src.fdm.fdm_main import MertonFDMSolver, MertonParams
from src.pinn.loss import concavity_loss, hjb_residual, monotonocity_loss, terminal_loss, total_loss
from src.pinn.neural_network import VanillaPINN
from src.pinn.training import _batch


PI_CLIP = (-1.0, 5.0)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_fields(t, w, pi, V=None, pi_clip=PI_CLIP) -> dict:
    ref_pi = closed_pi(t, w)
    out = {"pi": error_block(pi, ref_pi)}
    clipped = np.clip(pi, pi_clip[0], pi_clip[1])
    out["pi_clipped"] = error_block(clipped, ref_pi)
    out["frac_pi_outside_clip"] = float(np.mean((pi < pi_clip[0]) | (pi > pi_clip[1])))
    if V is not None:
        ref_V = closed_value(t, w)
        out["V"] = error_block(V, ref_V)
        out["V_rel"] = relative_error_block(V, ref_V)
    return out


def run_fdm(points: dict, budget: dict, out_dir: Path) -> dict:
    params = MertonParams(
        gamma=MARKET["gamma"], mu=MARKET["mu"], sigma=MARKET["sigma"],
        r=MARKET["rate"], T=MARKET["T"],
    )
    solver = MertonFDMSolver(
        params,
        Nx=int(budget["Nx"]),
        Nt=int(budget["Nt"]),
        x_min=float(budget.get("x_min", -2.5)),
        x_max=float(budget.get("x_max", 4.5)),
        theta=float(budget.get("theta", 1.0)),
        pi_min=float(budget.get("pi_min", -1.0)),
        pi_max=float(budget.get("pi_max", 5.0)),
    )
    t0 = time.perf_counter()
    solver.solve(verbose=False)
    elapsed = time.perf_counter() - t0

    log_w = np.log(points["w"])
    grid = (solver.t_grid, solver.x)
    query = np.column_stack([points["t"], log_w])
    V = RegularGridInterpolator(grid, solver.V, bounds_error=False, fill_value=None)(query)
    pi = RegularGridInterpolator(grid, solver.pi_fdm, bounds_error=False, fill_value=None)(query)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "prediction.npz",
        t=points["t"], w=points["w"], V=V, pi=pi,
    )
    metrics = score_fields(points["t"], points["w"], pi, V)
    return {
        "wall_clock_s": elapsed,
        "device": "cpu",
        "metrics": metrics,
        "pi_star_closed": params.pi_star,
        "A_closed": params.A,
    }


def _pinn_cfg(budget: dict, seed: int) -> dict:
    return {
        "seed": seed,
        "gamma": MARKET["gamma"],
        "mu": MARKET["mu"],
        "sigma": MARKET["sigma"],
        "rate": MARKET["rate"],
        "T": MARKET["T"],
        "w_min": 0.5,
        "w_max": 2.0,
        "lr": float(budget.get("lr", 1e-3)),
        "epochs": int(budget["epochs"]),
        "n_collocation": int(budget.get("n_collocation", 4000)),
        "n_terminal": int(budget.get("n_terminal", 1500)),
    }


def _eval_pinn(model, points, device) -> tuple[np.ndarray, np.ndarray, dict]:
    model.eval()
    w = torch.tensor(points["w"], dtype=torch.float32, device=device).view(-1, 1)
    t = torch.tensor(points["t"], dtype=torch.float32, device=device).view(-1, 1)
    w = w.requires_grad_(True)
    t = t.requires_grad_(True)
    V = model(w, t)
    ones = torch.ones_like(V)
    Vw = torch.autograd.grad(V, w, grad_outputs=ones, create_graph=True)[0]
    Vww = torch.autograd.grad(Vw, w, grad_outputs=torch.ones_like(Vw), create_graph=False)[0]
    mu = MARKET["mu"]
    r = MARKET["rate"]
    sigma = MARKET["sigma"]
    pi = -((mu - r) * Vw) / (sigma ** 2 * w * Vww)
    pi_np = pi.detach().cpu().numpy().ravel()
    V_np = V.detach().cpu().numpy().ravel()
    Vww_np = Vww.detach().cpu().numpy().ravel()
    extra = {
        "frac_Vww_nonnegative": float(np.mean(Vww_np >= 0)),
        "frac_pi_nonfinite": float(np.mean(~np.isfinite(pi_np))),
    }
    pi_np = np.where(np.isfinite(pi_np), pi_np, np.nan)
    return pi_np, V_np, extra


def run_pinn(points: dict, budget: dict, seed: int, out_dir: Path) -> dict:
    cfg = _pinn_cfg(budget, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _device()
    model = VanillaPINN().to(device)
    r = torch.tensor(cfg["rate"], device=device)
    mu = torch.tensor(cfg["mu"], device=device)
    sigma = torch.tensor(cfg["sigma"], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    history = []
    best = float("inf")
    best_state = None
    t0 = time.perf_counter()
    for epoch in range(cfg["epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        w, t, w_tc, t_tc, v_tc = _batch(cfg, device)
        hjb, _, v_x, v_xx = hjb_residual(model, t, w, r, sigma, mu)
        terminal = terminal_loss(model, w_tc, t_tc, v_tc)
        total = total_loss(concavity_loss(v_xx), monotonocity_loss(v_x), terminal, hjb)
        total.backward()
        opt.step()
        current = float(total.detach())
        history.append(current)
        if current < best:
            best = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 100 == 0 or epoch == cfg["epochs"] - 1:
            print(
                f"PINN seed={seed} epoch={epoch} total={current:.4e} best={best:.4e}",
                flush=True,
            )
    elapsed = time.perf_counter() - t0
    model.load_state_dict(best_state)
    pi, V, extra = _eval_pinn(model, points, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "best_loss": best, "config": cfg}, out_dir / "best_model.pt")
    np.savez_compressed(out_dir / "prediction.npz", t=points["t"], w=points["w"], V=V, pi=pi)
    np.save(out_dir / "loss_history.npy", np.asarray(history, dtype=np.float64))
    metrics = score_fields(points["t"], points["w"], pi, V)
    metrics.update(extra)
    metrics["best_train_loss"] = best
    return {"wall_clock_s": elapsed, "device": str(device), "metrics": metrics, "budget_resolved": cfg}


def run_bsde(points: dict, budget: dict, seed: int, out_dir: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _device()
    model = PiNet().to(device)
    z_model = Z_net().to(device)
    lr = float(budget.get("lr", 1e-3))
    opt = torch.optim.Adam(list(model.parameters()) + list(z_model.parameters()), lr=lr)
    epochs = int(budget["epochs"])
    t0 = time.perf_counter()
    history = training_loop(
        epochs=epochs,
        model=model,
        Z_model=z_model,
        optimizer=opt,
        mu=MARKET["mu"],
        sigma=MARKET["sigma"],
        rate=MARKET["rate"],
        w0=MARKET["w0"],
        T=MARKET["T"],
        num_steps=int(budget.get("num_steps", 64)),
        num_paths=int(budget.get("num_paths", 512)),
        gamma=MARKET["gamma"],
        device=device,
        save_best=True,
        best_path=str(out_dir / "best_model.pt"),
        show_live_plot=False,
    )
    elapsed = time.perf_counter() - t0
    # Score the best checkpoint, not the last epoch.
    blob = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(blob["pi_star"])
    model.eval()
    with torch.no_grad():
        t = torch.tensor(points["t"], dtype=torch.float32, device=device).view(-1, 1)
        w = torch.tensor(points["w"], dtype=torch.float32, device=device).view(-1, 1)
        pi = model(t, w).detach().cpu().numpy().ravel()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "prediction.npz", t=points["t"], w=points["w"], pi=pi)
    np.save(out_dir / "loss_history.npy", np.asarray(history, dtype=np.float64))
    metrics = score_fields(points["t"], points["w"], pi, V=None)
    y0 = float(blob.get("y0_mean", np.nan))
    v0 = float(closed_value(0.0, MARKET["w0"]))
    metrics["y0_mean"] = y0
    metrics["V0_closed"] = v0
    metrics["y0_abs_err"] = float(abs(y0 - v0)) if np.isfinite(y0) else None
    metrics["best_train_loss"] = float(blob.get("loss", np.nan))
    return {"wall_clock_s": elapsed, "device": str(device), "metrics": metrics}
