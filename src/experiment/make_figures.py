"""Build the figures and tables named in paper/HJB_Comparative_Study.tex.

Reads results/experiments/ledger.jsonl. For each method it keeps the
latest status=ok run of the requested suite (default: benchmark). FDM
slices are recomputed from the logged grid. PINN and BSDE slices are
recomputed from the logged checkpoint. Nothing is trained here.

    python -m src.experiment.make_figures
    python -m src.experiment.make_figures --suite benchmark

Images land in results/experiments/figures/ under the filenames the
tex file already includes. A manifest.json records which run_id each
curve came from. Do not drop those images into paper/ until the manifest
lists the training budget you intend to cite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.bsde.neural_networks import pi_star as PiNet
from src.experiment.closed_form import MARKET, policy as closed_pi, value as closed_value, pi_star
from src.experiment.ledger import Ledger
from src.fdm.fdm_main import MertonFDMSolver, MertonParams
from src.pinn.neural_network import VanillaPINN

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "results" / "experiments"

COLORS = {"closed": "#222222", "fdm": "#1b6ca8", "pinn": "#c45c26", "bsde": "#2e7d4f"}
LABELS = {"fdm": "FDM", "pinn": "PINN", "bsde": "BSDE"}


def latest_ok(rows: list[dict], suite: str | None) -> dict[str, dict]:
    chosen: dict[str, dict] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if suite is not None and row.get("suite") != suite:
            continue
        method = row.get("method")
        if method in ("fdm", "pinn", "bsde"):
            chosen[method] = row  # ledger is chronological; last write wins
    return chosen


def _params() -> MertonParams:
    return MertonParams(
        gamma=MARKET["gamma"], mu=MARKET["mu"], sigma=MARKET["sigma"],
        r=MARKET["rate"], T=MARKET["T"],
    )


def solve_fdm(budget: dict) -> MertonFDMSolver:
    solver = MertonFDMSolver(
        _params(),
        Nx=int(budget["Nx"]),
        Nt=int(budget["Nt"]),
        x_min=float(budget.get("x_min", -2.5)),
        x_max=float(budget.get("x_max", 4.5)),
        theta=float(budget.get("theta", 1.0)),
        pi_min=float(budget.get("pi_min", -1.0)),
        pi_max=float(budget.get("pi_max", 5.0)),
    )
    solver.solve(verbose=False)
    return solver


def _load_pinn(path: Path, device) -> VanillaPINN:
    blob = torch.load(path, map_location=device, weights_only=False)
    model = VanillaPINN().to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def _load_bsde(path: Path, device) -> PiNet:
    blob = torch.load(path, map_location=device, weights_only=False)
    model = PiNet().to(device)
    model.load_state_dict(blob["pi_star"])
    model.eval()
    return model


def pinn_on(model, t, w, device):
    tt = torch.tensor(t, dtype=torch.float32, device=device).reshape(-1, 1).requires_grad_(True)
    ww = torch.tensor(w, dtype=torch.float32, device=device).reshape(-1, 1).requires_grad_(True)
    V = model(ww, tt)
    ones = torch.ones_like(V)
    Vw = torch.autograd.grad(V, ww, grad_outputs=ones, create_graph=True)[0]
    Vww = torch.autograd.grad(Vw, ww, grad_outputs=torch.ones_like(Vw), create_graph=True)[0]
    Vt = torch.autograd.grad(V, tt, grad_outputs=ones, create_graph=False)[0]
    mu, r, sigma = MARKET["mu"], MARKET["rate"], MARKET["sigma"]
    pi = -((mu - r) * Vw) / (sigma ** 2 * ww * Vww)
    kappa = ((mu - r) / sigma) ** 2
    multiplied = Vt * Vww + r * ww * Vww * Vw - 0.5 * kappa * Vw ** 2
    residual = multiplied / (Vww.abs() + 1e-4)
    return (
        V.detach().cpu().numpy().ravel(),
        pi.detach().cpu().numpy().ravel(),
        residual.detach().cpu().numpy().ravel(),
    )


def bsde_on(model, t, w, device):
    tt = torch.tensor(np.asarray(t, dtype=np.float32), device=device).reshape(-1, 1)
    ww = torch.tensor(np.asarray(w, dtype=np.float32), device=device).reshape(-1, 1)
    with torch.no_grad():
        pi = model(tt, ww).detach().cpu().numpy().ravel()
    return pi


def fdm_sample(solver: MertonFDMSolver, t, w):
    from scipy.interpolate import RegularGridInterpolator
    grid = (solver.t_grid, solver.x)
    query = np.column_stack([np.asarray(t, float).ravel(), np.log(np.asarray(w, float).ravel())])
    V = RegularGridInterpolator(grid, solver.V, bounds_error=False, fill_value=None)(query)
    pi = RegularGridInterpolator(grid, solver.pi_fdm, bounds_error=False, fill_value=None)(query)
    return V, pi


def fdm_residual(solver: MertonFDMSolver) -> np.ndarray:
    """Absolute reduced-HJB residual on the interior of the log-wealth grid."""
    p = solver.p
    W = solver.W_grid[None, :]
    Vx = np.gradient(solver.V, solver.dx, axis=1)
    Vxx = np.gradient(Vx, solver.dx, axis=1)
    Vt = np.gradient(solver.V, solver.dt, axis=0)
    Vw = Vx / W
    Vww = (Vxx - Vx) / W ** 2
    kappa = ((p.mu - p.r) / p.sigma) ** 2
    resid = Vt + p.r * W * Vw - 0.5 * kappa * (Vw ** 2) / (Vww + 1e-12)
    interior = resid[1:-1, 2:-2]
    return np.abs(interior).ravel()


def _style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.frameon": False,
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig, folder: Path, name: str):
    path = folder / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def plot_policy_wealth(folder, curves, star):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    w = curves["w"]
    for ax, zoom in zip(axes, (True, False)):
        ax.axhline(star, color=COLORS["closed"], lw=1.6, label=rf"closed form $\pi^*\approx{star:.3f}$")
        for key in ("fdm", "bsde", "pinn"):
            if key not in curves:
                continue
            ax.plot(w, curves[key]["pi_w"], color=COLORS[key], lw=1.5, label=LABELS[key])
        ax.set_xlabel("wealth $W$ at $t=0$")
        if zoom:
            ax.set_ylim(-0.2, 1.4)
            ax.set_title("zoom on the closed form")
            ax.set_ylabel(r"risky weight $\pi$")
        else:
            ax.set_title("full range, including the PINN")
        ax.legend(fontsize=8)
    save(fig, folder, "policy_comparison.png")


def plot_policy_time(folder, curves, star):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    t = curves["t"]
    for ax, zoom in zip(axes, (True, False)):
        ax.axhline(star, color=COLORS["closed"], lw=1.6, label=rf"closed form $\pi^*\approx{star:.3f}$")
        for key in ("fdm", "bsde", "pinn"):
            if key not in curves:
                continue
            ax.plot(t, curves[key]["pi_t"], color=COLORS[key], lw=1.5, label=LABELS[key])
        ax.set_xlabel("time $t$ at $W=1$")
        if zoom:
            ax.set_ylim(-0.2, 1.4)
            ax.set_title("zoom on the closed form")
            ax.set_ylabel(r"risky weight $\pi$")
        else:
            ax.set_title("full range, including the PINN")
        ax.legend(fontsize=8)
    save(fig, folder, "policy_comparison_timeslice.png")


def plot_value_slice(folder, curves):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    w = curves["w"]
    ax.plot(w, curves["V_closed"], color=COLORS["closed"], lw=1.6, label="closed form")
    for key in ("fdm", "pinn"):
        if key in curves and "V_w" in curves[key]:
            ax.plot(w, curves[key]["V_w"], color=COLORS[key], lw=1.5, label=LABELS[key])
    ax.set_xlabel("wealth $W$ at $t=0.5$")
    ax.set_ylabel(r"value $V$")
    ax.legend()
    save(fig, folder, "wealth_slice_comparison.png")


def plot_heat(folder, heat):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, key in zip(axes, ("fdm", "pinn")):
        if key not in heat:
            ax.set_visible(False)
            continue
        im = ax.pcolormesh(heat["W"], heat["T"], heat[key], shading="auto", cmap="magma")
        ax.set_title(LABELS[key])
        ax.set_xlabel("wealth $W$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("time $t$")
    fig.suptitle(r"$|V-V_{\mathrm{closed}}|$", y=1.02)
    save(fig, folder, "error_heatmap_comparison.png")


def plot_residual(folder, samples: dict):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, values in samples.items():
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        values = values[values > 0]
        if values.size == 0:
            continue
        ax.hist(np.log10(values), bins=40, density=True, histtype="step",
                lw=1.6, color=COLORS[key], label=LABELS[key])
    ax.set_xlabel(r"$\log_{10}|$HJB residual$|$")
    ax.set_ylabel("density")
    ax.legend()
    save(fig, folder, "residual_histogram.png")


def plot_convergence(folder, rows: list[dict]):
    fdm_rows = [r for r in rows if r.get("method") == "fdm" and r.get("status") == "ok"]
    # one point per Nx, last occurrence
    by_nx = {}
    for r in fdm_rows:
        by_nx[int(r["budget"]["Nx"])] = r
    if len(by_nx) < 2:
        return
    xs = np.array(sorted(by_nx))
    pi = np.array([by_nx[n]["metrics"]["pi"]["mae"] for n in xs])
    V = np.array([by_nx[n]["metrics"]["V"]["mae"] for n in xs])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.loglog(xs, pi, "o-", color=COLORS["fdm"], label=r"$\pi$ MAE")
    ax.loglog(xs, V, "s--", color="#7a4e2d", label=r"$V$ MAE")
    ax.set_xlabel(r"wealth nodes $N_x$")
    ax.set_ylabel("error vs closed form")
    ax.legend()
    save(fig, folder, "convergence_fdm.png")


def plot_utility(folder, rows: list[dict]):
    """Certainty equivalent on the shared paths. One bar per method that logged it."""
    order = ["closed_form", "fdm", "bsde", "pinn"]
    labels, ces, ces_clip = [], [], []
    seen = set()
    for row in reversed(rows):
        method = row.get("method")
        if method in seen or method not in order:
            continue
        util = (row.get("metrics") or {}).get("utility")
        if not util:
            continue
        seen.add(method)
        ce = util["certainty_equivalent"]
        if not np.isfinite(ce) or not np.isfinite(util["mean_utility"]):
            ce = np.nan
        labels.append({"closed_form": "closed form", "fdm": "FDM", "bsde": "BSDE", "pinn": "PINN"}[method])
        ces.append(ce)
        clipped = (row.get("metrics") or {}).get("utility_clipped")
        ces_clip.append(clipped["certainty_equivalent"] if clipped else np.nan)
    if not labels:
        return
    # Restore the display order.
    rank = {name: i for i, name in enumerate(["closed form", "FDM", "BSDE", "PINN"])}
    packed = sorted(zip(labels, ces, ces_clip), key=lambda x: rank[x[0]])
    labels, ces, ces_clip = map(list, zip(*packed))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(labels))
    ax.bar(x - 0.16, ces, width=0.32, color="#1b6ca8", label="policy as produced")
    if np.isfinite(ces_clip).any():
        ax.bar(x + 0.16, ces_clip, width=0.32, color="#c45c26", label="PINN clipped to [-1, 5]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("certainty equivalent of terminal wealth")
    ax.legend()
    save(fig, folder, "utility_gap.png")


def plot_loss(folder, runs_root: Path, chosen: dict):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    drew = False
    for key in ("pinn", "bsde"):
        row = chosen.get(key)
        if not row:
            continue
        path = runs_root / row["run_id"] / "loss_history.npy"
        if not path.exists():
            continue
        hist = np.load(path)
        ax.semilogy(np.arange(1, hist.size + 1), np.clip(hist, 1e-16, None),
                    color=COLORS[key], lw=1.3, label=LABELS[key])
        drew = True
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.legend()
    save(fig, folder, "training_loss.png")


def plot_gamma(folder):
    gammas = np.array([2.0, 3.0, 5.0, 10.0, 20.0])
    closed, fdm = [], []
    base = _params()
    for g in gammas:
        params = MertonParams(gamma=float(g), mu=base.mu, sigma=base.sigma, r=base.r, T=base.T)
        solver = MertonFDMSolver(params, Nx=256, Nt=128, x_min=-2.5, x_max=4.5)
        solver.solve(verbose=False)
        # read pi at (t=0, W=1)
        ix = int(np.argmin(np.abs(solver.W_grid - 1.0)))
        closed.append(params.pi_star)
        fdm.append(float(solver.pi_fdm[0, ix]))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(gammas, closed, color=COLORS["closed"], lw=1.6, label="closed form")
    ax.plot(gammas, fdm, "o", color=COLORS["fdm"], label="FDM at $W=1$, $t=0$")
    ax.set_xlabel(r"risk aversion $\gamma$")
    ax.set_ylabel(r"$\pi^*$")
    ax.legend()
    save(fig, folder, "sensitivity_gamma.png")
    return [{"gamma": float(g), "pi_closed": float(c), "pi_fdm": float(f),
             "abs_err": float(abs(c - f))} for g, c, f in zip(gammas, closed, fdm)]


def plot_sigma(folder):
    sigmas = np.linspace(0.15, 0.45, 7)
    base = _params()
    closed, fdm = [], []
    for s in sigmas:
        params = MertonParams(gamma=base.gamma, mu=base.mu, sigma=float(s), r=base.r, T=base.T)
        solver = MertonFDMSolver(params, Nx=256, Nt=128, x_min=-2.5, x_max=4.5)
        solver.solve(verbose=False)
        ix = int(np.argmin(np.abs(solver.W_grid - 1.0)))
        closed.append(params.pi_star)
        fdm.append(float(solver.pi_fdm[0, ix]))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(sigmas, closed, color=COLORS["closed"], lw=1.6, label="closed form")
    ax.plot(sigmas, fdm, "o", color=COLORS["fdm"], label="FDM at $W=1$, $t=0$")
    ax.set_xlabel(r"volatility $\sigma$")
    ax.set_ylabel(r"$\pi^*$")
    ax.legend()
    save(fig, folder, "sensitivity_sigma.png")
    return [{"sigma": float(s), "pi_closed": float(c), "pi_fdm": float(f)}
            for s, c, f in zip(sigmas, closed, fdm)]


def write_tables(folder: Path, chosen: dict, gamma_rows, sigma_rows, conv_rows):
    def mae(row, *keys):
        cur = row.get("metrics") or {}
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return ""
            cur = cur[k]
        return cur

    lines = ["method,run_id,pi_mae,pi_rmse,pi_max,pi_clipped_mae,V_mae,V_rel_mae,wall_clock_s"]
    for key in ("fdm", "pinn", "bsde"):
        row = chosen.get(key)
        if not row:
            continue
        lines.append(",".join(str(x) for x in [
            key, row["run_id"],
            mae(row, "pi", "mae"), mae(row, "pi", "rmse"), mae(row, "pi", "max_ae"),
            mae(row, "pi_clipped", "mae"), mae(row, "V", "mae"), mae(row, "V_rel", "mae"),
            row.get("wall_clock_s", ""),
        ]))
    (folder / "table_main.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    conv = ["Nx,Nt,pi_mae,V_mae,wall_clock_s,run_id"]
    for row in conv_rows:
        b = row["budget"]
        conv.append(",".join(str(x) for x in [
            b["Nx"], b["Nt"], mae(row, "pi", "mae"), mae(row, "V", "mae"),
            row.get("wall_clock_s", ""), row["run_id"],
        ]))
    (folder / "table_fdm_convergence.csv").write_text("\n".join(conv) + "\n", encoding="utf-8")

    g = ["gamma,pi_closed,pi_fdm,abs_err"]
    g += [f"{r['gamma']},{r['pi_closed']},{r['pi_fdm']},{r['abs_err']}" for r in gamma_rows]
    (folder / "table_gamma.csv").write_text("\n".join(g) + "\n", encoding="utf-8")
    s = ["sigma,pi_closed,pi_fdm"]
    s += [f"{r['sigma']},{r['pi_closed']},{r['pi_fdm']}" for r in sigma_rows]
    (folder / "table_sigma.csv").write_text("\n".join(s) + "\n", encoding="utf-8")


def build(ledger_root: Path, suite: str | None):
    _style()
    folder = ledger_root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(ledger_root)
    rows = ledger.read()
    chosen = latest_ok(rows, suite)
    if not chosen:
        raise SystemExit(f"No status=ok runs for suite={suite!r} in {ledger.path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w = np.linspace(0.5, 2.0, 200)
    t_line = np.linspace(0.0, 0.95, 200)
    star = float(pi_star())

    curves = {"w": w, "t": t_line, "V_closed": closed_value(0.5, w)}
    heat_w = np.linspace(0.5, 2.0, 80)
    heat_t = np.linspace(0.0, 0.95, 60)
    WW, TT = np.meshgrid(heat_w, heat_t)
    heat = {"W": heat_w, "T": heat_t}
    residuals = {}

    # Reference FDM curve: finest logged grid if several exist.
    fdm_rows = [r for r in rows if r.get("method") == "fdm" and r.get("status") == "ok"
                and (suite is None or r.get("suite") == suite)]
    fdm_row = max(fdm_rows, key=lambda r: int(r["budget"]["Nx"])) if fdm_rows else None
    if fdm_row:
        solver = solve_fdm(fdm_row["budget"])
        _, pi_w = fdm_sample(solver, np.zeros_like(w), w)
        _, pi_t = fdm_sample(solver, t_line, np.ones_like(t_line))
        V_w, _ = fdm_sample(solver, np.full_like(w, 0.5), w)
        V_grid, _ = fdm_sample(solver, TT.ravel(), WW.ravel())
        err = np.abs(V_grid - closed_value(TT.ravel(), WW.ravel())).reshape(TT.shape)
        curves["fdm"] = {"pi_w": pi_w, "pi_t": pi_t, "V_w": V_w}
        heat["fdm"] = err
        residuals["fdm"] = fdm_residual(solver)
        chosen["fdm"] = fdm_row

    if "pinn" in chosen:
        model = _load_pinn(ledger_root / "runs" / chosen["pinn"]["run_id"] / "best_model.pt", device)
        V_w, pi_w, _ = pinn_on(model, np.zeros_like(w), w, device)
        _, pi_t, _ = pinn_on(model, t_line, np.ones_like(t_line), device)
        V_slice, _, _ = pinn_on(model, np.full_like(w, 0.5), w, device)
        V_grid, _, resid = pinn_on(model, TT.ravel(), WW.ravel(), device)
        curves["pinn"] = {"pi_w": pi_w, "pi_t": pi_t, "V_w": V_slice}
        heat["pinn"] = np.abs(V_grid - closed_value(TT.ravel(), WW.ravel())).reshape(TT.shape)
        residuals["pinn"] = np.abs(resid)

    if "bsde" in chosen:
        model = _load_bsde(ledger_root / "runs" / chosen["bsde"]["run_id"] / "best_model.pt", device)
        curves["bsde"] = {
            "pi_w": bsde_on(model, np.zeros_like(w), w, device),
            "pi_t": bsde_on(model, t_line, np.ones_like(t_line), device),
        }

    plot_policy_wealth(folder, curves, star)
    plot_policy_time(folder, curves, star)
    plot_value_slice(folder, curves)
    plot_heat(folder, heat)
    plot_residual(folder, residuals)
    plot_convergence(folder, fdm_rows)
    plot_loss(folder, ledger_root / "runs", chosen)
    plot_utility(folder, [r for r in rows if suite is None or r.get("suite") == suite])
    gamma_rows = plot_gamma(folder)
    sigma_rows = plot_sigma(folder)
    write_tables(folder, chosen, gamma_rows, sigma_rows, fdm_rows)

    manifest = {
        "suite_filter": suite,
        "note": "Draft figures from whatever ok runs are in the ledger. Replace by rerunning after the training budget you will cite.",
        "runs": {k: {"run_id": v["run_id"], "suite": v.get("suite"), "budget": v.get("budget")}
                 for k, v in chosen.items()},
        "files": sorted(p.name for p in folder.iterdir()),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["runs"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Write paper figures from the experiment ledger.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--suite", default="benchmark",
                        help="Use this suite. Pass 'any' to take the latest ok run of each method.")
    args = parser.parse_args()
    suite = None if args.suite == "any" else args.suite
    build(args.ledger, suite)


if __name__ == "__main__":
    main()
