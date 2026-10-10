"""Closed-form Merton value and policy used as the benchmark reference.

GBM drift is the annualized mean log return plus sigma^2/2.
The configs still store the mean log return in the mu field. New
figures and new solves use MARKET["mu"] below, which is the drift.
"""
from __future__ import annotations

import numpy as np

MARKET = {
    "gamma": 5.0,
    "m_log": 0.20164940530246397,
    "mu": 0.23456419554737107,
    "sigma": 0.2565727586666484,
    "rate": 0.0368,
    "T": 1.0,
    "w0": 1.0,
}


def crra_exponent(gamma: float) -> float:
    return 1.0 - gamma


def growth_A(gamma: float, mu: float, sigma: float, rate: float) -> float:
    p = crra_exponent(gamma)
    return p * (rate + (mu - rate) ** 2 / (2.0 * gamma * sigma ** 2))


def pi_star(mu: float | None = None, rate: float | None = None,
            gamma: float | None = None, sigma: float | None = None) -> float:
    mu = MARKET["mu"] if mu is None else mu
    rate = MARKET["rate"] if rate is None else rate
    gamma = MARKET["gamma"] if gamma is None else gamma
    sigma = MARKET["sigma"] if sigma is None else sigma
    return (mu - rate) / (gamma * sigma ** 2)


def value(t, w, **market) -> np.ndarray:
    gamma = market.get("gamma", MARKET["gamma"])
    mu = market.get("mu", MARKET["mu"])
    sigma = market.get("sigma", MARKET["sigma"])
    rate = market.get("rate", MARKET["rate"])
    T = market.get("T", MARKET["T"])
    p = crra_exponent(gamma)
    A = growth_A(gamma, mu, sigma, rate)
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)
    return np.exp(A * (T - t)) * np.power(w, p) / p


def policy(t, w, **market) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)
    star = pi_star(
        market.get("mu", MARKET["mu"]),
        market.get("rate", MARKET["rate"]),
        market.get("gamma", MARKET["gamma"]),
        market.get("sigma", MARKET["sigma"]),
    )
    return np.full(np.broadcast_shapes(t.shape, w.shape), star)
