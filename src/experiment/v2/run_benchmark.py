"""Pre-registered corrected-drift benchmark.

From merton_comparative_study:

    python -m src.experiment.v2.run_benchmark --exp smoke
    python -m src.experiment.v2.run_benchmark --exp E1
    python -m src.experiment.v2.run_benchmark --exp E3

E3, E4, and E5 loop seeds 0..4. E8 and E10 are the seed-1 repeats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import qmc

from src.experiment.metrics import error_block
from src.experiment.v2.check_market import assert_market, load_market
from src.experiment.v2.crra_pinn import CRRAValue, FlexibleValue
from src.experiment.v2.deep_bsde import PolicyNet, ZNet, rollout
from src.experiment.v2.factor import FACTOR, constant_reduction_error, interpolants, solve_factor, value_from_f
from src.fdm.fdm_main import MertonFDMSolver, MertonParams
from src.pinn.loss import concavity_loss, hjb_residual, monotonocity_loss, terminal_loss, total_loss
from src.pinn.neural_network import VanillaPINN

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "v2"
RUNS = OUT / "runs"
LEDGER = OUT / "ledger.jsonl"

EPOCHS = 2000
FACTOR_EPOCHS = 800
LR = 1e-3
N_COL = 25000
N_TERM = 8000
MESH_N = 10000
MESH_SEED = 12345
PATHS = 20000
PATH_SEED = 7
UTILITY_STEPS = 252


def market_hash() -> str:
    raw = (HERE / "market.json").read_bytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def closed_pi(mkt, mu=None, sigma=None, gamma=None):
    mu = mkt["mu"] if mu is None else mu
    sigma = mkt["sigma"] if sigma is None else sigma
    gamma = mkt["gamma"] if gamma is None else gamma
    return (mu - mkt["rate"]) / (gamma * sigma ** 2)


def closed_value(t, w, mkt, mu=None, sigma=None, gamma=None):
    mu = mkt["mu"] if mu is None else mu
    sigma = mkt["sigma"] if sigma is None else sigma
    gamma = mkt["gamma"] if gamma is None else gamma
    p = 1.0 - gamma
    excess = (mu - mkt["rate"]) ** 2 / (2.0 * gamma * sigma ** 2)
    A = p * (mkt["rate"] + excess)
    return np.exp(A * (mkt["T"] - t)) * np.power(w, p) / p


def mesh(n=MESH_N, seed=MESH_SEED, t_hi=1.0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, t_hi, n)
    w = np.exp(rng.uniform(np.log(0.5), np.log(2.0), n))
    return t, w


def surface_axes():
    t = np.linspace(0.0, 0.98, 41)
    w = np.linspace(0.5, 2.0, 51)
    tt, ww = np.meshgrid(t, w, indexing="ij")
    return tt, ww


def batch(n_col, n_term, T, w_min, w_max, gamma, device, seed):
    """Latin hypercube, redrawn each epoch.

    Interior: t uniform on [0, T], w uniform on [w_min, w_max], joint LHS.
    Terminal: t fixed at T, w uniform on [w_min, w_max], target U(w).
    """
    col = qmc.LatinHypercube(d=2, seed=int(seed))
    scaled = qmc.scale(col.random(n_col), [0.0, w_min], [T, w_max])
    w = torch.tensor(scaled[:, 1], dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    t = torch.tensor(scaled[:, 0], dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    term = qmc.LatinHypercube(d=1, seed=int(seed) + 17)
    w_tc = torch.tensor(qmc.scale(term.random(n_term), w_min, w_max), dtype=torch.float32, device=device)
    t_tc = torch.full((n_term, 1), T, dtype=torch.float32, device=device)
    v_tc = w_tc.clamp(min=1e-8).pow(1.0 - gamma) / (1.0 - gamma)
    w_tc = w_tc.requires_grad_(True)
    t_tc = t_tc.requires_grad_(True)
    return w, t, w_tc, t_tc, v_tc


def constrained_residual(model, t, w, r, sigma, mu, pi_min, pi_max):
    v = model(w, t)
    ones = torch.ones_like(v)
    v_t = torch.autograd.grad(v, t, grad_outputs=ones, create_graph=True)[0]
    v_w = torch.autograd.grad(v, w, grad_outputs=ones, create_graph=True)[0]
    v_ww = torch.autograd.grad(v_w, w, grad_outputs=torch.ones_like(v_w), create_graph=True)[0]
    foc = -((mu - r) * v_w) / (sigma ** 2 * w.clamp(min=1e-8) * v_ww.clamp(max=-1e-6))
    pi = foc.clamp(pi_min, pi_max)
    drift = (r + pi * (mu - r)) * w
    diff = 0.5 * (pi * sigma * w) ** 2
    resid = v_t + drift * v_w + diff * v_ww
    return torch.mean(resid ** 2), v_w, v_ww


def append(row: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row["market_sha256_16"] = market_hash()
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run_dir(exp, seed):
    p = RUNS / f"{exp}_s{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def fdm_solve(mkt, Nx, Nt, pi_min=-1.0, pi_max=5.0, mu=None, sigma=None, gamma=None):
    params = MertonParams(
        gamma if gamma is not None else mkt["gamma"],
        mu if mu is not None else mkt["mu"],
        sigma if sigma is not None else mkt["sigma"],
        mkt["rate"],
        mkt["T"],
    )
    solver = MertonFDMSolver(
        params, Nx=Nx, Nt=Nt, x_min=-2.5, x_max=4.5, theta=1.0, pi_min=pi_min, pi_max=pi_max
    )
    t0 = time.perf_counter()
    solver.solve(verbose=False)
    elapsed = time.perf_counter() - t0
    return solver, elapsed


def score_fdm(solver, mkt, t, w, mu=None, sigma=None, gamma=None):
    pi_it = RegularGridInterpolator((solver.t_grid, solver.x), solver.pi_fdm, bounds_error=False, fill_value=None)
    v_it = RegularGridInterpolator((solver.t_grid, solver.x), solver.V, bounds_error=False, fill_value=None)
    pi = pi_it(np.column_stack([t, np.log(w)]))
    V = v_it(np.column_stack([t, np.log(w)]))
    ref_pi = np.full_like(pi, closed_pi(mkt, mu, sigma, gamma))
    ref_v = closed_value(t, w, mkt, mu, sigma, gamma)
    return pi, V, error_block(pi, ref_pi), error_block(V, ref_v)


def trade(pi_of, mkt, n_paths=PATHS, n_steps=UTILITY_STEPS, seed=PATH_SEED, mu=None, sigma=None):
    mu = mkt["mu"] if mu is None else mu
    sigma = mkt["sigma"] if sigma is None else sigma
    rng = np.random.default_rng(seed)
    dt = mkt["T"] / n_steps
    w = np.full(n_paths, mkt["w0"], dtype=float)
    sqrt_dt = np.sqrt(dt)
    for i in range(n_steps):
        t = np.full(n_paths, i * dt)
        pi = np.clip(np.asarray(pi_of(t, w), dtype=float), -5.0, 5.0)
        z = rng.normal(size=n_paths)
        growth = (mkt["rate"] + pi * (mu - mkt["rate"]) - 0.5 * pi ** 2 * sigma ** 2) * dt
        w = w * np.exp(growth + pi * sigma * sqrt_dt * z)
    p = 1.0 - mkt["gamma"]
    u = np.power(np.maximum(w, 1e-12), p) / p
    eu = float(np.mean(u))
    ce = float((p * eu) ** (1.0 / p))
    return {"mean_utility": eu, "certainty_equivalent": ce, "n_paths": n_paths, "n_steps": n_steps, "seed": seed}


def pinn_policy_and_value(model, t, w, mkt, device):
    tw = torch.tensor(w, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    tt = torch.tensor(t, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    V = model(tw, tt)
    Vw = torch.autograd.grad(V, tw, torch.ones_like(V), create_graph=True)[0]
    Vww = torch.autograd.grad(Vw, tw, torch.ones_like(Vw))[0]
    pi = -((mkt["mu"] - mkt["rate"]) * Vw) / (mkt["sigma"] ** 2 * tw * Vww)
    return pi.detach().cpu().numpy().ravel(), V.detach().cpu().numpy().ravel(), Vww.detach().cpu().numpy().ravel()


def train_vanilla(mkt, seed, epochs, device, n_col, n_term, constrained=None):
    torch.manual_seed(seed)
    model = FlexibleValue(mkt["gamma"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    r = torch.tensor(mkt["rate"], device=device)
    mu = torch.tensor(mkt["mu"], device=device)
    sigma = torch.tensor(mkt["sigma"], device=device)
    best, best_state, hist = float("inf"), None, []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        w, t, w_tc, t_tc, v_tc = batch(n_col, n_term, mkt["T"], 0.5, 2.0, mkt["gamma"], device, seed * 100000 + epoch)
        if constrained is None:
            hjb, _, vx, vxx = hjb_residual(model, t, w, r, sigma, mu)
            term = terminal_loss(model, w_tc, t_tc, v_tc)
            loss = total_loss(concavity_loss(vxx), monotonocity_loss(vx), term, hjb)
        else:
            hjb, vx, vxx = constrained_residual(model, t, w, r, sigma, mu, constrained[0], constrained[1])
            term = terminal_loss(model, w_tc, t_tc, v_tc)
            loss = total_loss(concavity_loss(vxx), monotonocity_loss(vx), term, hjb)
        loss.backward()
        opt.step()
        val = float(loss.detach())
        hist.append(val)
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 100 == 0:
            print(f"  vanilla seed {seed} epoch {epoch} loss {val:.4e}", flush=True)
    model.load_state_dict(best_state)
    return model, best, hist, time.perf_counter() - t0


def train_crra(mkt, seed, epochs, device, n_col, n_term):
    torch.manual_seed(seed)
    model = CRRAValue(mkt["gamma"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    r = torch.tensor(mkt["rate"], device=device)
    mu = torch.tensor(mkt["mu"], device=device)
    sigma = torch.tensor(mkt["sigma"], device=device)
    best, best_state, hist = float("inf"), None, []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        w, t, w_tc, t_tc, v_tc = batch(n_col, n_term, mkt["T"], 0.5, 2.0, mkt["gamma"], device, seed * 100000 + epoch)
        hjb, _, vx, vxx = hjb_residual(model, t, w, r, sigma, mu)
        term = terminal_loss(model, w_tc, t_tc, v_tc)
        loss = total_loss(concavity_loss(vxx), monotonocity_loss(vx), term, hjb)
        loss.backward()
        opt.step()
        val = float(loss.detach())
        hist.append(val)
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, hist, time.perf_counter() - t0


def train_bsde(mkt, seed, epochs, device, n_paths, n_steps):
    torch.manual_seed(seed)
    policy, znet = PolicyNet().to(device), ZNet().to(device)
    p = 1.0 - mkt["gamma"]
    y0 = nn.Parameter(torch.tensor([mkt["w0"] ** p / p], dtype=torch.float32, device=device))
    opt = torch.optim.Adam(list(policy.parameters()) + list(znet.parameters()) + [y0], lr=LR)
    dt = mkt["T"] / n_steps
    best, best_state, hist = float("inf"), None, []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        dw = torch.randn(n_paths, n_steps, device=device) * np.sqrt(dt)
        loss, y_mean, u_mean = rollout(
            policy, znet, y0, dw, mkt["mu"], mkt["sigma"], mkt["rate"], mkt["w0"], mkt["T"], mkt["gamma"]
        )
        loss.backward()
        opt.step()
        val = float(loss.detach())
        hist.append(val)
        if val < best:
            best = val
            best_state = {
                "policy": {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()},
                "z": {k: v.detach().cpu().clone() for k, v in znet.state_dict().items()},
                "y0": float(y0.detach().cpu()),
            }
        if epoch % 100 == 0:
            print(f"  bsde seed {seed} epoch {epoch} loss {val:.4e}", flush=True)
    policy.load_state_dict(best_state["policy"])
    znet.load_state_dict(best_state["z"])
    return policy, znet, best, hist, best_state, time.perf_counter() - t0


def bsde_policy_np(policy, t, w, device):
    with torch.no_grad():
        tt = torch.tensor(t, dtype=torch.float32, device=device).view(-1, 1)
        ww = torch.tensor(w, dtype=torch.float32, device=device).view(-1, 1)
        return policy(tt, ww).cpu().numpy().ravel()


def log_row(exp, seed, status, payload):
    row = {"exp": exp, "seed": seed, "status": status, "mu": payload.get("mu"), "m_log": payload.get("m_log"), "sigma": payload.get("sigma")}
    row.update(payload)
    append(row)
    path = run_dir(exp, seed) / "row.json"
    path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(exp, seed, status, flush=True)


def do_fdm_reference(mkt, exp, Nx, Nt, seed=0, pi_min=-1.0, pi_max=5.0, mu=None, sigma=None, gamma=None):
    solver, elapsed = fdm_solve(mkt, Nx, Nt, pi_min, pi_max, mu, sigma, gamma)
    t, w = mesh()
    pi, V, e_pi, e_v = score_fdm(solver, mkt, t, w, mu, sigma, gamma)
    # Point value at t=0, W=1 via nearest log-wealth node.
    ix = int(np.argmin(np.abs(solver.x - 0.0)))
    v01 = float(solver.V[0, ix])
    closed01 = float(closed_value(0.0, 1.0, mkt, mu, sigma, gamma))
    util = trade(lambda tt, ww: np.full_like(ww, closed_pi(mkt, mu, sigma, gamma), dtype=float), mkt, mu=mu, sigma=sigma)
    util_fdm = trade(
        lambda tt, ww: RegularGridInterpolator((solver.t_grid, solver.x), solver.pi_fdm, bounds_error=False, fill_value=None)(
            np.column_stack([tt, np.log(ww)])
        ),
        mkt, mu=mu, sigma=sigma,
    )
    folder = run_dir(exp, seed)
    np.savez(folder / "prediction.npz", t=t, w=w, pi=pi, V=V, x=solver.x, t_grid=solver.t_grid, pi_grid=solver.pi_fdm, V_grid=solver.V)
    tt, ww = surface_axes()
    pi_it = RegularGridInterpolator((solver.t_grid, solver.x), solver.pi_fdm, bounds_error=False, fill_value=None)
    v_it = RegularGridInterpolator((solver.t_grid, solver.x), solver.V, bounds_error=False, fill_value=None)
    np.savez(
        folder / "surface.npz",
        t=tt, w=ww,
        pi=pi_it(np.column_stack([tt.ravel(), np.log(ww.ravel())])).reshape(tt.shape),
        V=v_it(np.column_stack([tt.ravel(), np.log(ww.ravel())])).reshape(tt.shape),
    )
    payload = {
        "method": "fdm", "seconds": elapsed, "Nx": Nx, "Nt": Nt,
        "mu": mu if mu is not None else mkt["mu"], "m_log": mkt["m_log"],
        "sigma": sigma if sigma is not None else mkt["sigma"],
        "gamma": gamma if gamma is not None else mkt["gamma"],
        "pi_min": pi_min, "pi_max": pi_max,
        "pi_star": closed_pi(mkt, mu, sigma, gamma),
        "pi": e_pi, "V": e_v, "V01_grid": v01, "V01_closed": closed01,
        "utility_closed": util, "utility_fdm": util_fdm,
    }
    log_row(exp, seed, "ok", payload)
    return payload


def do_pinn(mkt, exp, seed, epochs, device, n_col, n_term, constrained=None):
    model, best, hist, elapsed = train_vanilla(mkt, seed, epochs, device, n_col, n_term, constrained)
    t, w = mesh()
    pi, V, vww = pinn_policy_and_value(model, t, w, mkt, device)
    ref_pi = np.full_like(pi, closed_pi(mkt))
    if constrained is not None:
        ref_pi = np.clip(ref_pi, constrained[0], constrained[1])
        # Reference for the constrained 1D problem is the grid, stored by the caller in the row comparison later.
    ref_v = closed_value(t, w, mkt)
    clipped = np.clip(pi, -1.0, 5.0)
    folder = run_dir(exp, seed)
    torch.save({"model": model.state_dict(), "loss": best, "epoch_budget": epochs}, folder / "best_model.pt")
    np.save(folder / "loss_history.npy", np.asarray(hist, dtype=np.float32))
    np.savez(folder / "prediction.npz", t=t, w=w, pi=pi, V=V, vww=vww)
    tt, ww = surface_axes()
    pi_s, V_s, _ = pinn_policy_and_value(model, tt.ravel(), ww.ravel(), mkt, device)
    np.savez(folder / "surface.npz", t=tt, w=ww, pi=pi_s.reshape(tt.shape), V=V_s.reshape(tt.shape))
    def pi_of(tt, ww):
        p, _, _ = pinn_policy_and_value(model, tt, ww, mkt, device)
        return p
    payload = {
        "method": "flexible_power_pinn", "seconds": elapsed, "epochs": epochs, "best_loss": best,
        "head": "V=exp(g(t,log w))*w^(1-gamma)/(1-gamma); g may depend on wealth",
        "n_collocation": n_col, "n_terminal": n_term,
        "collocation": "LHS t~U[0,T], w~U[0.5,2], redrawn each epoch",
        "terminal": "t=T, w~U[0.5,2] LHS, target w^(1-gamma)/(1-gamma)",
        "mu": mkt["mu"], "m_log": mkt["m_log"], "sigma": mkt["sigma"], "gamma": mkt["gamma"],
        "pi": error_block(pi, ref_pi), "pi_clipped": error_block(clipped, ref_pi),
        "V": error_block(V, ref_v), "frac_concave": float(np.mean(vww < 0)),
        "utility": trade(pi_of, mkt), "constrained": constrained, "seed": seed,
    }
    log_row(exp, seed, "ok", payload)
    return payload


def do_crra(mkt, exp, seed, epochs, device, n_col, n_term):
    model, best, hist, elapsed = train_crra(mkt, seed, epochs, device, n_col, n_term)
    t, w = mesh()
    pi, V, vww = pinn_policy_and_value(model, t, w, mkt, device)
    ref_pi = np.full_like(pi, closed_pi(mkt))
    ref_v = closed_value(t, w, mkt)
    folder = run_dir(exp, seed)
    torch.save({"model": model.state_dict(), "loss": best, "epoch_budget": epochs}, folder / "best_model.pt")
    np.save(folder / "loss_history.npy", np.asarray(hist, dtype=np.float32))
    np.savez(folder / "prediction.npz", t=t, w=w, pi=pi, V=V)
    tt, ww = surface_axes()
    pi_s, V_s, _ = pinn_policy_and_value(model, tt.ravel(), ww.ravel(), mkt, device)
    np.savez(folder / "surface.npz", t=tt, w=ww, pi=pi_s.reshape(tt.shape), V=V_s.reshape(tt.shape))
    def pi_of(tt, ww):
        return np.full_like(ww, closed_pi(mkt), dtype=float)
    # Policy from the ansatz is 1/gamma by algebra. Utility of the traded
    # constant weight is the closed-form policy, reported as such.
    payload = {
        "method": "crra_ansatz", "seconds": elapsed, "epochs": epochs, "best_loss": best,
        "n_collocation": n_col, "n_terminal": n_term,
        "note": "V_w/(w V_ww)=1/gamma for any g. Policy MAE is a roundoff check. The trained object is V.",
        "mu": mkt["mu"], "m_log": mkt["m_log"], "sigma": mkt["sigma"], "gamma": mkt["gamma"],
        "pi": error_block(pi, ref_pi), "V": error_block(V, ref_v),
        "V01": float(closed_value(0.0, 1.0, mkt)),
        "utility_of_pi_star": trade(pi_of, mkt), "seed": seed,
    }
    # Actual traded utility of the ansatz policy (should match pi*).
    payload["utility"] = trade(lambda tt, ww: pinn_policy_and_value(model, tt, ww, mkt, device)[0], mkt)
    log_row(exp, seed, "ok", payload)
    return payload


def do_bsde(mkt, exp, seed, epochs, device, n_paths, n_steps):
    policy, znet, best, hist, state, elapsed = train_bsde(mkt, seed, epochs, device, n_paths, n_steps)
    t, w = mesh()
    pi = bsde_policy_np(policy, t, w, device)
    ref = np.full_like(pi, closed_pi(mkt))
    folder = run_dir(exp, seed)
    torch.save(state, folder / "best_model.pt")
    np.save(folder / "loss_history.npy", np.asarray(hist, dtype=np.float32))
    np.savez(folder / "prediction.npz", t=t, w=w, pi=pi)
    tt, ww = surface_axes()
    pi_s = bsde_policy_np(policy, tt.ravel(), ww.ravel(), device)
    np.savez(folder / "surface.npz", t=tt, w=ww, pi=pi_s.reshape(tt.shape))
    payload = {
        "method": "deep_bsde_terminal", "seconds": elapsed, "epochs": epochs, "best_loss": best,
        "n_paths": n_paths, "n_steps": n_steps,
        "scheme": "Y <- Y + Z dB with Z evaluated at the pre-update state; loss E[(Y_T-U)^2] - E[U]",
        "mu": mkt["mu"], "m_log": mkt["m_log"], "sigma": mkt["sigma"], "gamma": mkt["gamma"],
        "pi": error_block(pi, ref), "y0": state["y0"],
        "V01_closed": float(closed_value(0.0, 1.0, mkt)),
        "abs_y0_gap": abs(state["y0"] - float(closed_value(0.0, 1.0, mkt))),
        "utility": trade(lambda tt, ww: bsde_policy_np(policy, tt, ww, device), mkt),
        "seed": seed,
    }
    log_row(exp, seed, "ok", payload)
    return payload


class FactorG(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, t, y):
        return self.net(torch.cat([t, y], dim=1))


class FactorPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, t, w, y):
        x = torch.cat([t, torch.log(w.clamp(min=1e-8)), y], dim=1)
        return 2.0 * torch.tanh(self.net(x))


class FactorZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, t, w, y):
        x = torch.cat([t, torch.log(w.clamp(min=1e-8)), y], dim=1)
        return self.net(x)


def train_factor_pinn(mkt, seed, epochs, device, spec):
    torch.manual_seed(seed)
    model = FactorG().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gam = mkt["gamma"]
    p = 1.0 - gam
    sig, r, mu0 = mkt["sigma"], mkt["rate"], mkt["mu"]
    kappa, nu, phi, rho = spec["kappa"], spec["nu"], spec["phi"], spec["rho"]
    best, state, hist = float("inf"), None, []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        rng_seed = seed * 100000 + epoch
        col = qmc.LatinHypercube(d=2, seed=rng_seed)
        scaled = qmc.scale(col.random(4096), [0.0, spec["y_min"]], [mkt["T"], spec["y_max"]])
        t = torch.tensor(scaled[:, 0], dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
        y = torch.tensor(scaled[:, 1], dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
        g = model(t, y)
        f = torch.exp(g)
        ones = torch.ones_like(g)
        f_t = torch.autograd.grad(f, t, ones, create_graph=True)[0]
        f_y = torch.autograd.grad(f, y, ones, create_graph=True)[0]
        f_yy = torch.autograd.grad(f_y, y, torch.ones_like(f_y), create_graph=True)[0]
        e = mu0 + phi * y - r
        raw = (e + rho * sig * nu * (f_y / f.clamp(min=1e-8))) / (gam * sig ** 2)
        pi = raw.clamp(spec["pi_min"], spec["pi_max"])
        alpha = p * (r + pi * e) + 0.5 * (pi ** 2) * (sig ** 2) * p * (p - 1.0)
        beta = -kappa * y + pi * sig * rho * nu * p
        resid = f_t + 0.5 * nu ** 2 * f_yy + beta * f_y + alpha * f
        term_y = torch.empty(1024, 1, device=device).uniform_(spec["y_min"], spec["y_max"])
        tT = torch.full_like(term_y, mkt["T"])
        term = torch.mean((model(tT, term_y)) ** 2)
        loss = torch.mean(resid ** 2) + 10.0 * term
        loss.backward()
        opt.step()
        val = float(loss.detach())
        hist.append(val)
        if val < best:
            best = val
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 100 == 0:
            print(f"  factor pinn seed {seed} epoch {epoch} loss {val:.4e}", flush=True)
    model.load_state_dict(state)
    return model, best, hist, time.perf_counter() - t0


def train_factor_bsde(mkt, seed, epochs, device, spec, n_paths=1024, n_steps=64):
    torch.manual_seed(seed)
    policy, znet = FactorPolicy().to(device), FactorZ().to(device)
    pwr = 1.0 - mkt["gamma"]
    y0 = nn.Parameter(torch.tensor([mkt["w0"] ** pwr / pwr], dtype=torch.float32, device=device))
    opt = torch.optim.Adam(list(policy.parameters()) + list(znet.parameters()) + [y0], lr=LR)
    dt = mkt["T"] / n_steps
    kappa, nu, phi, rho = spec["kappa"], spec["nu"], spec["phi"], spec["rho"]
    best, state, hist = float("inf"), None, []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        z1 = torch.randn(n_paths, n_steps, device=device)
        z2 = rho * z1 + np.sqrt(max(1.0 - rho ** 2, 0.0)) * torch.randn(n_paths, n_steps, device=device)
        dw = z1 * np.sqrt(dt)
        dy = z2 * np.sqrt(dt)
        w = torch.full((n_paths, 1), mkt["w0"], device=device)
        y = torch.zeros((n_paths, 1), device=device)
        Y = y0.expand(n_paths, 1).clone()
        for i in range(n_steps):
            t = torch.full((n_paths, 1), i * dt, device=device)
            pi = policy(t, w, y)
            z = znet(t, w, y)
            mu_y = mkt["mu"] + phi * y
            growth = (mkt["rate"] + pi * (mu_y - mkt["rate"]) - 0.5 * pi ** 2 * mkt["sigma"] ** 2) * dt
            shock = pi * mkt["sigma"] * dw[:, i:i + 1]
            Y = Y + z[:, 0:1] * dw[:, i:i + 1] + z[:, 1:2] * dy[:, i:i + 1]
            w = w * torch.exp(growth + shock)
            y = y - kappa * y * dt + nu * dy[:, i:i + 1]
        terminal = w.clamp(min=1e-8).pow(pwr) / pwr
        loss = torch.mean((Y - terminal) ** 2) - terminal.mean()
        loss.backward()
        opt.step()
        val = float(loss.detach())
        hist.append(val)
        if val < best:
            best = val
            state = {
                "policy": {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()},
                "z": {k: v.detach().cpu().clone() for k, v in znet.state_dict().items()},
                "y0": float(y0.detach().cpu()),
            }
        if epoch % 100 == 0:
            print(f"  factor bsde seed {seed} epoch {epoch} loss {val:.4e}", flush=True)
    policy.load_state_dict(state["policy"])
    znet.load_state_dict(state["z"])
    return policy, best, hist, state, time.perf_counter() - t0


def do_factor(mkt, exp, seed, epochs, device, train_nets=True):
    red = constant_reduction_error(mkt)
    print(f"factor constant-reduction max |f-exp(AT)| = {red:.3e}", flush=True)
    if red > 5e-2:
        raise SystemExit(f"factor grid failed the constant reduction test: {red}")
    grid = solve_factor(mkt)
    f_it, pi_it = interpolants(grid)
    rng = np.random.default_rng(MESH_SEED)
    n = 4000
    t = rng.uniform(0.0, 0.95, n)
    y = rng.uniform(FACTOR["y_min"] * 0.8, FACTOR["y_max"] * 0.8, n)
    w = np.exp(rng.uniform(np.log(0.5), np.log(2.0), n))
    f_ref = f_it(np.column_stack([t, y]))
    pi_ref = pi_it(np.column_stack([t, y]))
    V_ref = value_from_f(f_ref, w, mkt["gamma"])
    folder = run_dir(exp, seed)
    np.savez(
        folder / "fdm_grid.npz", t=grid["t"], y=grid["y"], f=grid["f"], pi=grid["pi"],
        eval_t=t, eval_y=y, eval_w=w, pi_ref=pi_ref, V_ref=V_ref,
    )
    payload = {
        "method": "factor_fdm", "mu": mkt["mu"], "m_log": mkt["m_log"], "sigma": mkt["sigma"],
        "gamma": mkt["gamma"], "spec": FACTOR, "constant_reduction_max_abs": red,
        "V01": float(value_from_f(f_it([[0.0, 0.0]])[0], 1.0, mkt["gamma"])),
        "pi_at_0": float(pi_it([[0.0, 0.0]])[0]),
    }
    if train_nets:
        model, best, hist, elapsed = train_factor_pinn(mkt, seed, epochs, device, FACTOR)
        tt = torch.tensor(t, dtype=torch.float32, device=device).view(-1, 1)
        yy = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
        with torch.no_grad():
            f_hat = torch.exp(model(tt, yy)).cpu().numpy().ravel()
        # Policy from the learned f needs f_y. Finite difference in y.
        dy = 1e-3
        with torch.no_grad():
            fp = torch.exp(model(tt, yy + dy)).cpu().numpy().ravel()
            fm = torch.exp(model(tt, yy - dy)).cpu().numpy().ravel()
        fy = (fp - fm) / (2 * dy)
        e = mkt["mu"] + FACTOR["phi"] * y - mkt["rate"]
        raw = (e + FACTOR["rho"] * mkt["sigma"] * FACTOR["nu"] * (fy / np.maximum(f_hat, 1e-8))) / (
            mkt["gamma"] * mkt["sigma"] ** 2
        )
        pi_hat = np.clip(raw, FACTOR["pi_min"], FACTOR["pi_max"])
        V_hat = value_from_f(f_hat, w, mkt["gamma"])
        torch.save({"model": model.state_dict(), "loss": best}, folder / "pinn.pt")
        np.save(folder / "pinn_loss.npy", np.asarray(hist, dtype=np.float32))
        policy, bloss, bhist, state, belapsed = train_factor_bsde(mkt, seed, epochs, device, FACTOR)
        with torch.no_grad():
            pi_b = policy(
                tt,
                torch.tensor(w, dtype=torch.float32, device=device).view(-1, 1),
                yy,
            ).cpu().numpy().ravel()
        torch.save(state, folder / "bsde.pt")
        np.save(folder / "bsde_loss.npy", np.asarray(bhist, dtype=np.float32))
        np.savez(folder / "prediction.npz", t=t, y=y, w=w, pi_pinn=pi_hat, V_pinn=V_hat, pi_bsde=pi_b, pi_ref=pi_ref, V_ref=V_ref)
        payload.update({
            "pinn_seconds": elapsed, "pinn_best_loss": best, "epochs": epochs,
            "pinn_pi": error_block(pi_hat, pi_ref), "pinn_V": error_block(V_hat, V_ref),
            "bsde_seconds": belapsed, "bsde_best_loss": bloss, "bsde_y0": state["y0"],
            "bsde_pi": error_block(pi_b, pi_ref),
            "abs_y0_gap_vs_fdm": abs(state["y0"] - payload["V01"]),
        })
    log_row(exp, seed, "ok", payload)
    return payload


def do_constrained(mkt, exp, seed, epochs, device, n_col, n_term, train_net=True):
    pi_min, pi_max = 0.0, 0.4
    solver, elapsed = fdm_solve(mkt, 400, 200, pi_min, pi_max)
    t, w = mesh()
    pi, V, _, _ = score_fdm(solver, mkt, t, w)
    # The box binds: unconstrained pi* is about 0.60, so the reference is not pi*.
    star = np.full_like(pi, closed_pi(mkt))
    folder = run_dir(exp, seed)
    np.savez(folder / "fdm_prediction.npz", t=t, w=w, pi=pi, V=V, x=solver.x, t_grid=solver.t_grid, pi_grid=solver.pi_fdm, V_grid=solver.V)
    ix = int(np.argmin(np.abs(solver.x - 0.0)))
    payload = {
        "method": "constrained_fdm", "seconds": elapsed, "pi_min": pi_min, "pi_max": pi_max,
        "mu": mkt["mu"], "m_log": mkt["m_log"], "sigma": mkt["sigma"], "gamma": mkt["gamma"],
        "pi_star_unconstrained": closed_pi(mkt),
        "gap_vs_unconstrained_pi": error_block(pi, star),
        "V01_grid": float(solver.V[0, ix]),
        "note": "Cap [0, 0.4] binds because pi* is about 0.601. Networks are scored against this grid.",
    }
    if train_net:
        model, best, hist, pelapsed = train_vanilla(
            mkt, seed, epochs, device, n_col, n_term, constrained=(pi_min, pi_max)
        )
        pi_hat, V_hat, vww = pinn_policy_and_value(model, t, w, mkt, device)
        pi_hat = np.clip(pi_hat, pi_min, pi_max)
        torch.save({"model": model.state_dict(), "loss": best}, folder / "best_model.pt")
        np.save(folder / "loss_history.npy", np.asarray(hist, dtype=np.float32))
        np.savez(folder / "prediction.npz", t=t, w=w, pi=pi_hat, V=V_hat, pi_ref=pi, V_ref=V)
        payload.update({
            "pinn_seconds": pelapsed, "pinn_best_loss": best, "epochs": epochs,
            "n_collocation": n_col, "n_terminal": n_term,
            "pi_vs_grid": error_block(pi_hat, pi), "V_vs_grid": error_block(V_hat, V),
            "frac_concave": float(np.mean(vww < 0)),
        })
    log_row(exp, seed, "ok", payload)
    return payload


def finished(exp, seed, epochs):
    path = RUNS / f"{exp}_s{seed}" / "row.json"
    if not path.exists():
        return False
    row = json.loads(path.read_text(encoding="utf-8"))
    return row.get("status") == "ok" and int(row.get("epochs", -1)) >= int(epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    report = assert_market()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "market_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    mkt = load_market()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = args.exp
    smoke = exp == "smoke"
    epochs = args.epochs if args.epochs is not None else (3 if smoke else EPOCHS)
    n_col = 1024 if smoke else N_COL
    n_term = 256 if smoke else N_TERM
    print(f"device {device} exp {exp} epochs {epochs}", flush=True)
    if smoke:
        do_fdm_reference(mkt, "smoke", 80, 40)
        do_pinn(mkt, "smoke_pinn", 0, epochs, device, n_col, n_term)
        do_crra(mkt, "smoke_crra", 0, epochs, device, n_col, n_term)
        do_bsde(mkt, "smoke_bsde", 0, epochs, device, 128, 8)
        do_factor(mkt, "smoke_factor", 0, epochs, device, train_nets=False)
        return
    if exp == "E1":
        payload = do_fdm_reference(mkt, "E1", 500, 252)
        if payload["pi"]["mae"] > 2e-3:
            raise SystemExit(f"FDM policy MAE too large: {payload['pi']['mae']}")
        if abs(payload["V01_grid"] - payload["V01_closed"]) > 5e-3:
            raise SystemExit(f"FDM V(0,1) gap too large: {payload['V01_grid']} vs {payload['V01_closed']}")
        return
    if exp == "E2":
        for nx in (128, 256, 500):
            for nt in (64, 128, 252):
                do_fdm_reference(mkt, "E2", nx, nt, seed=nx * 1000 + nt)
        return
    if exp == "E3":
        for seed in range(5):
            if finished("E3", seed, epochs):
                print(f"skip E3 seed {seed}", flush=True)
                continue
            do_pinn(mkt, "E3", seed, epochs, device, n_col, n_term)
        return
    if exp == "E4":
        for seed in range(5):
            if finished("E4", seed, epochs):
                print(f"skip E4 seed {seed}", flush=True)
                continue
            do_crra(mkt, "E4", seed, epochs, device, n_col, n_term)
        return
    if exp == "E5":
        for seed in range(5):
            if finished("E5", seed, epochs):
                print(f"skip E5 seed {seed}", flush=True)
                continue
            do_bsde(mkt, "E5", seed, epochs, device, 1024, 64)
        return
    if exp == "E6":
        for gamma in (3.0, 7.0):
            do_fdm_reference(mkt, "E6", 256, 128, seed=int(gamma * 10), gamma=gamma)
        for scale, tag in ((0.8, 8), (1.2, 12)):
            do_fdm_reference(mkt, "E6", 256, 128, seed=tag, sigma=mkt["sigma"] * scale)
        return
    if exp == "E7":
        ep = FACTOR_EPOCHS if args.epochs is None else epochs
        if not finished("E7", 0, ep):
            do_factor(mkt, "E7", 0, ep, device, True)
        else:
            print("skip E7", flush=True)
        return
    if exp == "E8":
        ep = FACTOR_EPOCHS if args.epochs is None else epochs
        if not finished("E8", 1, ep):
            do_factor(mkt, "E8", 1, ep, device, True)
        else:
            print("skip E8", flush=True)
        return
    if exp == "E9":
        if not finished("E9", 0, epochs):
            do_constrained(mkt, "E9", 0, epochs, device, n_col, n_term, True)
        else:
            print("skip E9", flush=True)
        return
    if exp == "E10":
        if not finished("E10", 1, epochs):
            do_constrained(mkt, "E10", 1, epochs, device, n_col, n_term, True)
        else:
            print("skip E10", flush=True)
        return
    raise SystemExit(f"unknown exp {exp}")


if __name__ == "__main__":
    main()
