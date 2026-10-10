"""Confirm the frozen GBM coefficients against a fresh yfinance pull.

m = 252 * mean(log return) is not the drift in dS/S.
sigma = sqrt(252) * sample std(log return, ddof=1).
mu = m + sigma^2/2.
The frozen file keeps the stored pair. A live download may differ in
the 8th digit; the check fails if the gap exceeds 1e-6.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_market() -> dict:
    with open(HERE / "market.json", encoding="utf-8") as f:
        return json.load(f)


def live_moments():
    import yfinance as yf

    df = yf.download(
        "AAPL", start="2023-05-23", end="2026-05-23", auto_adjust=True, progress=False
    )
    px = df["Close"].dropna()
    if hasattr(px, "columns"):
        px = px.iloc[:, 0]
    px = px.astype(float)
    lr = np.log(px.to_numpy()[1:] / px.to_numpy()[:-1])
    m = float(252.0 * lr.mean())
    sigma = float(np.sqrt(252.0) * lr.std(ddof=1))
    return {
        "n_sessions": int(px.shape[0]),
        "n_returns": int(lr.size),
        "first": str(px.index[0].date()),
        "last": str(px.index[-1].date()),
        "m_log": m,
        "sigma": sigma,
        "mu": m + 0.5 * sigma ** 2,
    }


def assert_market(mkt: dict | None = None, live: dict | None = None) -> dict:
    mkt = mkt or load_market()
    m = float(mkt["m_log"])
    sigma = float(mkt["sigma"])
    mu = float(mkt["mu"])
    half = 0.5 * sigma ** 2
    if abs(mu - (m + half)) > 1e-12:
        raise SystemExit(f"mu {mu} != m + sigma^2/2 {m + half}")
    pi = (mu - float(mkt["rate"])) / (float(mkt["gamma"]) * sigma ** 2)
    if abs(pi - float(mkt["pi_star"])) > 1e-12:
        raise SystemExit(f"pi* {pi} != stored {mkt['pi_star']}")
    report = {
        "m_log": m,
        "sigma": sigma,
        "half_sigma_sq": half,
        "mu": mu,
        "pi_star": pi,
        "rate": float(mkt["rate"]),
        "gamma": float(mkt["gamma"]),
        "identity_ok": True,
    }
    if live is None:
        live = live_moments()
    report["live"] = live
    if abs(live["m_log"] - m) > 1e-6 or abs(live["sigma"] - sigma) > 1e-6:
        raise SystemExit(f"live sample disagrees with frozen market: {report}")
    if live["n_sessions"] != 753 or live["last"] != "2026-05-22":
        raise SystemExit(f"unexpected window: {live}")
    return report


if __name__ == "__main__":
    print(json.dumps(assert_market(), indent=2))
