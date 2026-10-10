"""Score the three trained checkpoints. Does not train.

Writes suite=release rows into results/experiments/ledger.jsonl and
saves the meshes those rows are drawn from. The paper may quote only
these rows.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from src.bsde.neural_networks import pi_star as PiNet
from src.experiment.closed_form import MARKET, growth_A, pi_star, policy as closed_pi, value as closed_value
from src.experiment.eval_mesh import load_or_create
from src.experiment.ledger import Ledger, new_run_id, utc_now
from src.experiment.metrics import error_block, relative_error_block
from src.fdm.fdm_main import MertonFDMSolver, MertonParams
from src.pinn.neural_network import VanillaPINN

ROOT = Path(__file__).resolve().parents[2]
LEDGER_ROOT = ROOT / "results" / "experiments"
FDM_NPZ = ROOT / "src" / "fdm" / "saved_models" / "best_model.npz"
PINN_PT = ROOT / "src" / "pinn" / "saved_models" / "best_model.pt"
BSDE_PT = ROOT / "src" / "bsde" / "saved_models" / "best_model.pt"

N_PATHS = 20000
N_STEPS = 252
UTILITY_SEED = 7
PI_CLIP = (-1.0, 5.0)


def _params(gamma=None, sigma=None) -> MertonParams:
    return MertonParams(
        gamma=MARKET["gamma"] if gamma is None else gamma,
        mu=MARKET["mu"],
        sigma=MARKET["sigma"] if sigma is None else sigma,
        r=MARKET["rate"],
        T=MARKET["T"],
    )


def _score(t, w, pi, V=None):
    out = {
        "pi": error_block(pi, closed_pi(t, w)),
        "pi_clipped": error_block(np.clip(pi, *PI_CLIP), closed_pi(t, w)),
        "frac_pi_outside_clip": float(np.mean((pi < PI_CLIP[0]) | (pi > PI_CLIP[1]))),
    }
    if V is not None:
        ref = closed_value(t, w)
        out["V"] = error_block(V, ref)
        out["V_rel"] = relative_error_block(V, ref)
    return out


def _utility_stats(terminal: np.ndarray) -> dict:
    gamma = MARKET["gamma"]
    p = 1.0 - gamma
    u = np.power(terminal, p) / p
    mean_u = float(np.mean(u))
    # U(CE) = mean_u, U(w) = w^p / p, p < 0.
    ce = float((mean_u * p) ** (1.0 / p))
    return {"mean_utility": mean_u, "certainty_equivalent": ce, "mean_terminal_wealth": float(np.mean(terminal))}


def _simulate(pi_of_tw, dw: np.ndarray) -> np.ndarray:
    """Log-Euler wealth. pi_of_tw(t, w_array) -> pi array. dw shape (paths, steps)."""
    mu, r, sigma = MARKET["mu"], MARKET["rate"], MARKET["sigma"]
    dt = MARKET["T"] / dw.shape[1]
    w = np.full(dw.shape[0], MARKET["w0"], dtype=np.float64)
    times = np.linspace(0.0, MARKET["T"], dw.shape[1] + 1)
    for i in range(dw.shape[1]):
        pi = np.asarray(pi_of_tw(times[i], w), dtype=np.float64).reshape(-1)
        pi = np.where(np.isfinite(pi), pi, 0.0)
        growth = (r + pi * (mu - r) - 0.5 * (pi ** 2) * (sigma ** 2)) * dt
        w = w * np.exp(growth + pi * sigma * dw[:, i])
    return w


def _dw(n_paths: int, n_steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = MARKET["T"] / n_steps
    return rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))


def _pinn_policy_fn(model, device):
    mu, r, sigma = MARKET["mu"], MARKET["rate"], MARKET["sigma"]

    def fn(t_scalar, w_np):
        w = torch.tensor(w_np, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
        t = torch.full_like(w, float(t_scalar)).requires_grad_(True)
        V = model(w, t)
        Vw = torch.autograd.grad(V, w, torch.ones_like(V), create_graph=True)[0]
        Vww = torch.autograd.grad(Vw, w, torch.ones_like(Vw))[0]
        pi = -((mu - r) * Vw) / (sigma ** 2 * w * Vww)
        return pi.detach().cpu().numpy().ravel()

    return fn


def _bsde_policy_fn(model, device):
    def fn(t_scalar, w_np):
        w = torch.tensor(w_np, dtype=torch.float32, device=device).view(-1, 1)
        t = torch.full_like(w, float(t_scalar))
        with torch.no_grad():
            return model(t, w).cpu().numpy().ravel()

    return fn


def _fdm_policy_fn(t_grid, x_grid, pi_grid):
    interp = RegularGridInterpolator((t_grid, x_grid), pi_grid, bounds_error=False, fill_value=None)

    def fn(t_scalar, w_np):
        query = np.column_stack([np.full(w_np.shape[0], t_scalar), np.log(np.clip(w_np, 1e-8, None))])
        return interp(query)

    return fn


def _base(method, run_id, budget, source):
    return {
        "run_id": run_id,
        "suite": "release",
        "method": method,
        "seed": 0,
        "status": "ok",
        "started_at": utc_now(),
        "market": dict(MARKET),
        "pi_star_closed": pi_star(),
        "A_closed": growth_A(MARKET["gamma"], MARKET["mu"], MARKET["sigma"], MARKET["rate"]),
        "budget": budget,
        "source_checkpoint": str(source),
        "eval": {"n": 10000, "seed": 12345, "w_min": 0.5, "w_max": 2.0, "t_max": 0.95, "wealth_sampling": "log-uniform"},
        "notes": "Scored from the trained checkpoint. No new training.",
    }


def _save_row(ledger: Ledger, record, out_dir: Path, **arrays):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "prediction.npz", **arrays)
    ledger.append(record)


def score_fdm_file(ledger, points, dw):
    z = np.load(FDM_NPZ)
    t0 = time.perf_counter()
    query = np.column_stack([points["t"], np.log(points["w"])])
    V = RegularGridInterpolator((z["t"], z["x"]), z["V"], bounds_error=False, fill_value=None)(query)
    pi = RegularGridInterpolator((z["t"], z["x"]), z["pi"], bounds_error=False, fill_value=None)(query)
    elapsed = time.perf_counter() - t0
    metrics = _score(points["t"], points["w"], pi, V)
    terminal = _simulate(_fdm_policy_fn(z["t"], z["x"], z["pi"]), dw)
    metrics["utility"] = _utility_stats(terminal)
    run_id = new_run_id("fdm", 0)
    record = _base("fdm", run_id, {"Nx": int(z["x"].shape[0]), "Nt": int(z["t"].shape[0] - 1),
                                   "x_min": float(z["x"][0]), "x_max": float(z["x"][-1])}, FDM_NPZ)
    record["wall_clock_s"] = elapsed
    record["wall_clock_is"] = "interpolation of the saved grid onto the mesh, not the original solve"
    record["metrics"] = metrics
    _save_row(ledger, record, LEDGER_ROOT / "runs" / run_id, t=points["t"], w=points["w"], V=V, pi=pi)
    return run_id


def score_fdm_grid(ledger, points, Nx, Nt):
    solver = MertonFDMSolver(_params(), Nx=Nx, Nt=Nt, x_min=-2.5, x_max=4.5, theta=1.0, pi_min=-1.0, pi_max=5.0)
    t0 = time.perf_counter()
    solver.solve(verbose=False)
    elapsed = time.perf_counter() - t0
    query = np.column_stack([points["t"], np.log(points["w"])])
    V = RegularGridInterpolator((solver.t_grid, solver.x), solver.V, bounds_error=False, fill_value=None)(query)
    pi = RegularGridInterpolator((solver.t_grid, solver.x), solver.pi_fdm, bounds_error=False, fill_value=None)(query)
    metrics = _score(points["t"], points["w"], pi, V)
    run_id = new_run_id("fdm", 0)
    record = _base("fdm", run_id, {"Nx": Nx, "Nt": Nt, "x_min": -2.5, "x_max": 4.5, "theta": 1.0}, "re-solved for the convergence ladder")
    record["wall_clock_s"] = elapsed
    record["wall_clock_is"] = "solve time of this grid"
    record["metrics"] = metrics
    _save_row(ledger, record, LEDGER_ROOT / "runs" / run_id, t=points["t"], w=points["w"], V=V, pi=pi)
    print(f"FDM Nx={Nx} pi_mae={metrics['pi']['mae']:.6g} solve_s={elapsed:.3f}", flush=True)
    return run_id


def score_pinn(ledger, points, dw, device):
    blob = torch.load(PINN_PT, map_location=device, weights_only=False)
    model = VanillaPINN().to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    w = torch.tensor(points["w"], dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    t = torch.tensor(points["t"], dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    t0 = time.perf_counter()
    V = model(w, t)
    Vw = torch.autograd.grad(V, w, torch.ones_like(V), create_graph=True)[0]
    Vww = torch.autograd.grad(Vw, w, torch.ones_like(Vw))[0]
    mu, r, sigma = MARKET["mu"], MARKET["rate"], MARKET["sigma"]
    pi_t = -((mu - r) * Vw) / (sigma ** 2 * w * Vww)
    elapsed = time.perf_counter() - t0
    pi = pi_t.detach().cpu().numpy().ravel()
    V_np = V.detach().cpu().numpy().ravel()
    metrics = _score(points["t"], points["w"], pi, V_np)
    metrics["frac_Vww_nonnegative"] = float((Vww.detach().cpu().numpy().ravel() >= 0).mean())
    metrics["best_train_loss"] = float(blob["loss"])
    metrics["best_epoch"] = int(blob["epoch"])
    raw = _simulate(_pinn_policy_fn(model, device), dw)
    clip_fn = lambda ts, ww, fn=_pinn_policy_fn(model, device): np.clip(fn(ts, ww), *PI_CLIP)
    clipped = _simulate(clip_fn, dw)
    metrics["utility"] = _utility_stats(raw)
    metrics["utility_clipped"] = _utility_stats(clipped)
    run_id = new_run_id("pinn", 0)
    record = _base("pinn", run_id, blob.get("config") or {}, PINN_PT)
    record["wall_clock_s"] = elapsed
    record["wall_clock_is"] = "evaluation of the saved network on the mesh, not training time"
    record["metrics"] = metrics
    out = LEDGER_ROOT / "runs" / run_id
    _save_row(ledger, record, out, t=points["t"], w=points["w"], V=V_np, pi=pi)
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "best_loss": float(blob["loss"]), "config": blob.get("config")}, out / "best_model.pt")
    print(f"PINN pi_mae={metrics['pi']['mae']:.6g} U={metrics['utility']['mean_utility']:.6g}", flush=True)
    return run_id


def score_bsde(ledger, points, dw, device):
    blob = torch.load(BSDE_PT, map_location=device, weights_only=False)
    model = PiNet().to(device)
    model.load_state_dict(blob["pi_star"])
    model.eval()
    w = torch.tensor(points["w"], dtype=torch.float32, device=device).view(-1, 1)
    t = torch.tensor(points["t"], dtype=torch.float32, device=device).view(-1, 1)
    t0 = time.perf_counter()
    with torch.no_grad():
        pi = model(t, w).cpu().numpy().ravel()
    elapsed = time.perf_counter() - t0
    metrics = _score(points["t"], points["w"], pi, None)
    metrics["y0_mean"] = float(blob["y0_mean"])
    metrics["V0_closed"] = float(closed_value(0.0, MARKET["w0"]))
    metrics["y0_abs_err"] = abs(metrics["y0_mean"] - metrics["V0_closed"])
    metrics["best_train_loss"] = float(blob["loss"])
    metrics["best_epoch"] = int(blob["epoch"])
    terminal = _simulate(_bsde_policy_fn(model, device), dw)
    metrics["utility"] = _utility_stats(terminal)
    run_id = new_run_id("bsde", 0)
    record = _base("bsde", run_id, blob.get("config") or {}, BSDE_PT)
    record["wall_clock_s"] = elapsed
    record["wall_clock_is"] = "evaluation of the saved network on the mesh, not training time"
    record["metrics"] = metrics
    out = LEDGER_ROOT / "runs" / run_id
    _save_row(ledger, record, out, t=points["t"], w=points["w"], pi=pi)
    torch.save({"pi_star": model.state_dict(), "loss": float(blob["loss"]), "y0_mean": float(blob["y0_mean"]),
                "epoch": int(blob["epoch"]), "config": blob.get("config")}, out / "best_model.pt")
    print(f"BSDE pi_mae={metrics['pi']['mae']:.6g} U={metrics['utility']['mean_utility']:.6g}", flush=True)
    return run_id


def score_closed_form_utility(ledger, dw):
    terminal = _simulate(lambda t, w: np.full_like(w, pi_star()), dw)
    metrics = {"utility": _utility_stats(terminal), "V0_closed": float(closed_value(0.0, 1.0))}
    run_id = new_run_id("closed_form", 0)
    record = _base("closed_form", run_id, {"policy": "constant pi*"}, "analytical")
    record["metrics"] = metrics
    record["notes"] = f"Same {dw.shape[0]} paths and {dw.shape[1]} steps as the three solvers. Seed {UTILITY_SEED}."
    ledger.append(record)
    print(f"closed U={metrics['utility']['mean_utility']:.6g} CE={metrics['utility']['certainty_equivalent']:.6g}", flush=True)


def main():
    ledger = Ledger(LEDGER_ROOT)
    points = load_or_create(LEDGER_ROOT)
    dw = _dw(N_PATHS, N_STEPS, UTILITY_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    score_closed_form_utility(ledger, dw)
    score_fdm_file(ledger, points, dw)
    for Nx, Nt in ((64, 64), (128, 128), (256, 252), (500, 252)):
        score_fdm_grid(ledger, points, Nx, Nt)
    score_pinn(ledger, points, dw, device)
    score_bsde(ledger, points, dw, device)
    print("release rows appended", flush=True)


if __name__ == "__main__":
    main()
