"""Figures and tables from results/v2/ledger.jsonl. Does not train.

    python -m src.experiment.v2.make_figures
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v2"
FIG = OUT / "figures"
RUNS = OUT / "runs"
LEDGER = OUT / "ledger.jsonl"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "figure.dpi": 140,
    "savefig.bbox": "tight",
})


def load_rows():
    rows = []
    if not LEDGER.exists():
        return rows
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    # Last completed row for each (exp, seed) wins. Smoke rows stay out of the paper.
    kept = {}
    for row in rows:
        if str(row.get("exp", "")).startswith("smoke"):
            continue
        if row.get("status") != "ok":
            continue
        kept[(row["exp"], row.get("seed"))] = row
    return list(kept.values())


def by_exp(rows, exp):
    return sorted([r for r in rows if r["exp"] == exp], key=lambda r: r.get("seed", 0))


def write_csv(path, header, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(records)


def mae(block):
    if not isinstance(block, dict):
        return float("nan")
    return block.get("mae", float("nan"))


def mean_std(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def surface_error(ref, pred):
    err = pred - ref
    finite = np.isfinite(err)
    if not np.any(finite):
        return err
    lo, hi = np.quantile(err[finite], [0.02, 0.98])
    return np.clip(err, lo, hi), lo, hi


def plot_surface(ax, t, w, z, title, zlabel):
    ax.plot_surface(t, w, z, cmap="viridis", linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.set_xlabel("t")
    ax.set_ylabel("W")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=28, azim=-125)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if not rows:
        raise SystemExit("no logged rows")
    mkt = json.loads((ROOT / "src" / "experiment" / "v2" / "market.json").read_text(encoding="utf-8"))
    half = 0.5 * mkt["sigma"] ** 2
    v01 = None
    e1 = by_exp(rows, "E1")
    if e1:
        v01 = e1[0].get("V01_closed")
    write_csv(
        FIG / "table_market.csv",
        ["quantity", "value"],
        [
            ["m_log", mkt["m_log"]],
            ["half_sigma_sq", half],
            ["mu", mkt["mu"]],
            ["sigma", mkt["sigma"]],
            ["rate", mkt["rate"]],
            ["gamma", mkt["gamma"]],
            ["pi_star", mkt["pi_star"]],
            ["V01_closed", v01 if v01 is not None else ""],
            ["window", "AAPL adjusted close 2023-05-23 to 2026-05-22"],
        ],
    )

    def summarize(group, method):
        pi = [mae(r.get("pi")) for r in group]
        vv = [mae(r.get("V")) for r in group]
        ygap = [r.get("abs_y0_gap", float("nan")) for r in group]
        ce = [r.get("utility", {}).get("certainty_equivalent", float("nan")) for r in group]
        return method, *mean_std(pi), *mean_std(vv), *mean_std(ygap), *mean_std(ce), len(group)

    main_rows = []
    if e1:
        r = e1[0]
        main_rows.append([
            "FDM 500x252", 1,
            mae(r.get("pi")), 0.0,
            mae(r.get("V")), 0.0,
            "", "",
            r.get("utility_fdm", {}).get("certainty_equivalent", ""),
            "",
        ])
    for exp, name in (("E4", "CRRA ansatz"), ("E3", "Vanilla PINN"), ("E5", "Deep BSDE")):
        group = by_exp(rows, exp)
        if not group:
            continue
        method, pm, ps, vm, vs, ym, ys, cm, cs, n = summarize(group, name)
        main_rows.append([method, n, pm, ps, vm, vs, ym, ys, cm, cs])
    write_csv(
        FIG / "table_main.csv",
        ["method", "n_seeds", "pi_mae_mean", "pi_mae_std", "V_mae_mean", "V_mae_std", "y0_gap_mean", "y0_gap_std", "CE_mean", "CE_std"],
        main_rows,
    )

    e2 = by_exp(rows, "E2")
    write_csv(
        FIG / "table_fdm_convergence.csv",
        ["Nx", "Nt", "pi_mae", "V_mae", "V01_grid", "V01_closed"],
        [[r["Nx"], r["Nt"], mae(r["pi"]), mae(r["V"]), r["V01_grid"], r["V01_closed"]] for r in e2],
    )

    e6 = by_exp(rows, "E6")
    write_csv(
        FIG / "table_sensitivity.csv",
        ["seed", "gamma", "sigma", "pi_star", "pi_mae", "V_mae"],
        [[r["seed"], r["gamma"], r["sigma"], r["pi_star"], mae(r["pi"]), mae(r["V"])] for r in e6],
    )

    factor_rows = []
    for exp in ("E7", "E8"):
        for r in by_exp(rows, exp):
            factor_rows.append([
                exp, r["seed"], r.get("constant_reduction_max_abs"),
                mae(r.get("pinn_pi")), mae(r.get("pinn_V")),
                mae(r.get("bsde_pi")), r.get("bsde_y0"), r.get("abs_y0_gap_vs_fdm"),
                r.get("pi_at_0"), r.get("V01"),
            ])
    write_csv(
        FIG / "table_factor.csv",
        ["exp", "seed", "reduction", "pinn_pi_mae", "pinn_V_mae", "bsde_pi_mae", "bsde_y0", "y0_gap", "pi0", "V01"],
        factor_rows,
    )

    cap_rows = []
    for exp in ("E9", "E10"):
        for r in by_exp(rows, exp):
            cap_rows.append([
                exp, r["seed"], r.get("pi_min"), r.get("pi_max"),
                mae(r.get("gap_vs_unconstrained_pi")),
                mae(r.get("pi_vs_grid")), mae(r.get("V_vs_grid")),
                r.get("V01_grid"), r.get("pi_star_unconstrained"),
            ])
    write_csv(
        FIG / "table_constraint.csv",
        ["exp", "seed", "pi_min", "pi_max", "fdm_vs_unconstrained_pi_mae", "pinn_pi_mae", "pinn_V_mae", "V01_grid", "pi_star"],
        cap_rows,
    )

    # Loss curves, seed 0 of the three 1D networks.
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for exp, label, fname in (
        ("E3", "Vanilla PINN", "loss_history.npy"),
        ("E4", "CRRA ansatz", "loss_history.npy"),
        ("E5", "Deep BSDE", "loss_history.npy"),
    ):
        path = RUNS / f"{exp}_s0" / fname
        if path.exists():
            h = np.load(path)
            ax.semilogy(h, label=label, lw=1.2)
    ax.set_xlabel("Adam epoch")
    ax.set_ylabel("Training loss")
    ax.legend(frameon=False)
    fig.savefig(FIG / "training_loss.png")
    plt.close(fig)

    def load_surface(exp, seed):
        path = RUNS / f"{exp}_s{seed}" / "surface.npz"
        if not path.exists():
            return None
        return np.load(path)

    e1s = load_surface("E1", 0)
    p0 = load_surface("E3", 0)
    c0 = load_surface("E4", 0)
    b0 = load_surface("E5", 0)
    if e1s is not None:
        fig = plt.figure(figsize=(9.2, 3.6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        plot_surface(ax1, e1s["t"], e1s["w"], e1s["V"], "FDM value", "V")
        plot_surface(ax2, e1s["t"], e1s["w"], e1s["pi"], "FDM policy", "pi")
        fig.savefig(FIG / "surface_fdm.png")
        plt.close(fig)
        # Closed form on the same axes.
        from src.experiment.v2.run_benchmark import closed_value, closed_pi
        fig = plt.figure(figsize=(9.2, 3.6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        Vc = closed_value(e1s["t"], e1s["w"], mkt)
        Pc = np.full_like(Vc, closed_pi(mkt))
        plot_surface(ax1, e1s["t"], e1s["w"], Vc, "Closed-form value", "V")
        plot_surface(ax2, e1s["t"], e1s["w"], Pc, "Closed-form policy", "pi")
        fig.savefig(FIG / "surface_closed.png")
        plt.close(fig)
        if p0 is not None:
            err, lo, hi = surface_error(Vc, p0["V"])
            fig = plt.figure(figsize=(6.4, 4.2))
            ax = fig.add_subplot(111, projection="3d")
            plot_surface(ax, p0["t"], p0["w"], err, f"Vanilla PINN value error, clip [{lo:.3f}, {hi:.3f}]", "error")
            fig.savefig(FIG / "surface_pinn_error.png")
            plt.close(fig)
            (FIG / "surface_pinn_error_clip.txt").write_text(f"{lo},{hi}\n", encoding="utf-8")

    # Slices at t closest to 0 and 0.5 from the random mesh of E1 and networks.
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    for ax, t_target, title in zip(axes, (0.05, 0.5), ("t near 0", "t near 0.5")):
        if e1s is not None:
            idx = int(np.argmin(np.abs(e1s["t"][:, 0] - t_target)))
            ax.plot(e1s["w"][idx], e1s["pi"][idx], label="FDM", color="black", lw=1.4)
        if b0 is not None:
            idx = int(np.argmin(np.abs(b0["t"][:, 0] - t_target)))
            ax.plot(b0["w"][idx], b0["pi"][idx], label="Deep BSDE", lw=1.1)
        if c0 is not None:
            idx = int(np.argmin(np.abs(c0["t"][:, 0] - t_target)))
            ax.plot(c0["w"][idx], c0["pi"][idx], label="CRRA ansatz", lw=1.0, ls="--")
        ax.axhline(mkt["pi_star"], color="0.4", lw=0.8, label="pi*")
        ax.set_ylim(-0.2, 1.4)
        ax.set_xlabel("W")
        ax.set_ylabel("pi")
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "policy_slices.png")
    plt.close(fig)

    if p0 is not None and e1s is not None:
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        idx = int(np.argmin(np.abs(p0["t"][:, 0] - 0.05)))
        ax.plot(p0["w"][idx], p0["pi"][idx], label="Vanilla PINN")
        ax.axhline(mkt["pi_star"], color="black", lw=1.0, label="pi*")
        ax.set_xlabel("W")
        ax.set_ylabel("pi")
        ax.set_title("Vanilla PINN policy at t near 0")
        ax.legend(frameon=False)
        fig.savefig(FIG / "policy_pinn_slice.png")
        plt.close(fig)

        err = np.abs(p0["V"] - closed_value(p0["t"], p0["w"], mkt))
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        im = ax.pcolormesh(p0["w"], p0["t"], err, shading="auto", cmap="magma")
        fig.colorbar(im, ax=ax, label="|V error|")
        ax.set_xlabel("W")
        ax.set_ylabel("t")
        ax.set_title("Vanilla PINN absolute value error")
        fig.savefig(FIG / "error_heatmap_pinn.png")
        plt.close(fig)

        err_f = np.abs(e1s["V"] - closed_value(e1s["t"], e1s["w"], mkt))
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        im = ax.pcolormesh(e1s["w"], e1s["t"], err_f, shading="auto", cmap="magma")
        fig.colorbar(im, ax=ax, label="|V error|")
        ax.set_xlabel("W")
        ax.set_ylabel("t")
        ax.set_title("FDM absolute value error")
        fig.savefig(FIG / "error_heatmap_fdm.png")
        plt.close(fig)

    # Factor surface at the saved grid, t=0.
    fgrid = RUNS / "E7_s0" / "fdm_grid.npz"
    if fgrid.exists():
        g = np.load(fgrid)
        fig = plt.figure(figsize=(6.4, 4.2))
        ax = fig.add_subplot(111, projection="3d")
        # pi(t,y) at all t is a surface over (t,y). Wealth is separable.
        tt, yy = np.meshgrid(g["t"], g["y"], indexing="ij")
        ax.plot_surface(tt, yy, g["pi"], cmap="viridis", linewidth=0, antialiased=True, rstride=2, cstride=2)
        ax.set_xlabel("t")
        ax.set_ylabel("factor Y")
        ax.set_zlabel("pi")
        ax.set_title("Factor-model policy")
        ax.view_init(elev=28, azim=-125)
        fig.savefig(FIG / "surface_factor_policy.png")
        plt.close(fig)

    manifest = []
    for path in sorted(FIG.iterdir()):
        if path.suffix.lower() in {".png", ".csv", ".txt"}:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            manifest.append({"file": path.name, "sha256_16": digest})
    manifest.append({"mu": mkt["mu"], "m_log": mkt["m_log"], "sigma": mkt["sigma"], "n_rows": len(rows)})
    (FIG / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {FIG}", flush=True)


if __name__ == "__main__":
    main()
