"""One-factor Merton problem that does not collapse to a wealth ODE.

Excess drift is mu + phi * Y. Y is Ornstein-Uhlenbeck and correlated
with the stock, so the CRRA hedge term depends on f_y/f.

Wealth separates: V = f(t,y) * w^(1-gamma) / (1-gamma).
f is solved on a (t,y) grid. Constant coefficients (phi=0, rho=0)
must reproduce exp(A(T-t)).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


FACTOR = {
    "kappa": 1.2,
    "nu": 0.30,
    "phi": 0.20,
    "rho": 0.45,
    "y_min": -1.5,
    "y_max": 1.5,
    "Ny": 81,
    "Nt": 200,
    "pi_min": -1.0,
    "pi_max": 5.0,
}


def _A_constant(mkt):
    p = 1.0 - mkt["gamma"]
    excess = (mkt["mu"] - mkt["rate"]) ** 2 / (2.0 * mkt["gamma"] * mkt["sigma"] ** 2)
    return p * (mkt["rate"] + excess)


def solve_factor(mkt, spec=None, pi_cap=None):
    spec = dict(FACTOR if spec is None else spec)
    gam = float(mkt["gamma"])
    p = 1.0 - gam
    sig = float(mkt["sigma"])
    r = float(mkt["rate"])
    mu0 = float(mkt["mu"])
    T = float(mkt["T"])
    kappa, nu, phi, rho = spec["kappa"], spec["nu"], spec["phi"], spec["rho"]
    y = np.linspace(spec["y_min"], spec["y_max"], int(spec["Ny"]))
    dy = float(y[1] - y[0])
    nt = int(spec["Nt"])
    dt = T / nt
    f = np.ones_like(y)
    pi_lo = spec["pi_min"] if pi_cap is None else pi_cap[0]
    pi_hi = spec["pi_max"] if pi_cap is None else pi_cap[1]
    pi_path = np.zeros((nt + 1, y.size))
    f_path = np.zeros((nt + 1, y.size))
    pi_path[-1] = np.clip((mu0 + phi * y - r) / (gam * sig ** 2), pi_lo, pi_hi)
    f_path[-1] = f
    D = 0.5 * nu ** 2
    for k in range(nt - 1, -1, -1):
        fy = np.gradient(f, dy)
        e = mu0 + phi * y - r
        raw = (e + rho * sig * nu * (fy / np.maximum(f, 1e-8))) / (gam * sig ** 2)
        pi = np.clip(raw, pi_lo, pi_hi)
        alpha = p * (r + pi * e) + 0.5 * (pi ** 2) * (sig ** 2) * p * (p - 1.0)
        beta = -kappa * y + pi * sig * rho * nu * p
        # (I + dt L) f_new = f, L = D dyy + beta dy + alpha, upwind drift.
        n = y.size
        lower = np.zeros(n - 1)
        diag = np.ones(n)
        upper = np.zeros(n - 1)
        for i in range(1, n - 1):
            bp = max(beta[i], 0.0)
            bm = min(beta[i], 0.0)
            a_l = D / dy ** 2 + bp / dy
            a_r = D / dy ** 2 - bm / dy
            a_c = -2.0 * D / dy ** 2 - bp / dy + bm / dy + alpha[i]
            lower[i - 1] = -dt * a_l
            diag[i] = 1.0 - dt * a_c
            upper[i] = -dt * a_r
        # Neumann ends: one-sided first derivative zero, diffusion dropped.
        diag[0] = 1.0 - dt * alpha[0]
        upper[0] = 0.0
        diag[-1] = 1.0 - dt * alpha[-1]
        lower[-1] = 0.0
        ab = np.zeros((3, n))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        from scipy.linalg import solve_banded

        f = solve_banded((1, 1), ab, f)
        f = np.maximum(f, 1e-8)
        pi_path[k] = pi
        f_path[k] = f
    t = np.linspace(0.0, T, nt + 1)
    return {
        "t": t,
        "y": y,
        "f": f_path,
        "pi": pi_path,
        "spec": spec,
        "pi_cap": [pi_lo, pi_hi],
    }


def interpolants(grid):
    f_it = RegularGridInterpolator((grid["t"], grid["y"]), grid["f"], bounds_error=False, fill_value=None)
    pi_it = RegularGridInterpolator((grid["t"], grid["y"]), grid["pi"], bounds_error=False, fill_value=None)
    return f_it, pi_it


def value_from_f(f, w, gamma):
    p = 1.0 - gamma
    return f * np.power(w, p) / p


def constant_reduction_error(mkt) -> float:
    spec = dict(FACTOR)
    spec["phi"] = 0.0
    spec["rho"] = 0.0
    grid = solve_factor(mkt, spec)
    A = _A_constant(mkt)
    # f(0, y) should be exp(A T)
    target = np.exp(A * mkt["T"])
    return float(np.max(np.abs(grid["f"][0] - target)))
