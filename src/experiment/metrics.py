"""Pointwise error summaries against the closed form."""
from __future__ import annotations

import numpy as np


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.isfinite(a)


def error_block(pred: np.ndarray, ref: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    ok = _finite(pred) & _finite(ref)
    n = int(pred.size)
    n_ok = int(ok.sum())
    if n_ok == 0:
        return {"n": n, "n_finite": 0}
    e = pred[ok] - ref[ok]
    ae = np.abs(e)
    return {
        "n": n,
        "n_finite": n_ok,
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "median_ae": float(np.median(ae)),
        "p95_ae": float(np.quantile(ae, 0.95)),
        "max_ae": float(ae.max()),
        "bias": float(e.mean()),
    }


def relative_error_block(pred: np.ndarray, ref: np.ndarray, floor: float = 1e-8) -> dict:
    pred = np.asarray(pred, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    ok = _finite(pred) & _finite(ref)
    if not np.any(ok):
        return {"n_finite": 0}
    scale = np.maximum(np.abs(ref[ok]), floor)
    rel = np.abs(pred[ok] - ref[ok]) / scale
    return {
        "n_finite": int(ok.sum()),
        "mae": float(rel.mean()),
        "rmse": float(np.sqrt(np.mean(rel ** 2))),
        "median_ae": float(np.median(rel)),
        "p95_ae": float(np.quantile(rel, 0.95)),
        "max_ae": float(rel.max()),
    }
