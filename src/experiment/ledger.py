"""Append-only experiment ledger.

Each finished run is one JSON object per line in ledger.jsonl, plus a
copy of that object under runs/<run_id>.json. Nothing here is overwritten
except summary.csv, which is rebuilt from the ledger.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(method: str, seed: int | None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    seed_bit = "na" if seed is None else str(seed)
    return f"{stamp}_{method}_s{seed_bit}"


class Ledger:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        self.path = self.root / "ledger.jsonl"

    def append(self, record: dict) -> Path:
        record = dict(record)
        record.setdefault("logged_at", utc_now())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")
        run_path = self.root / "runs" / f"{record['run_id']}.json"
        run_path.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")
        self.write_summary()
        return run_path

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def write_summary(self) -> Path:
        rows = self.read()
        out = self.root / "summary.csv"
        fields = [
            "run_id", "suite", "method", "seed", "status",
            "wall_clock_s", "epochs",
            "pi_mae", "pi_rmse", "pi_max_ae",
            "pi_clipped_mae",
            "V_mae", "V_rel_mae",
            "y0_abs_err",
            "device",
        ]
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                m = row.get("metrics") or {}
                pi = m.get("pi") or {}
                pi_c = m.get("pi_clipped") or {}
                V = m.get("V") or {}
                Vrel = m.get("V_rel") or {}
                writer.writerow({
                    "run_id": row.get("run_id"),
                    "suite": row.get("suite"),
                    "method": row.get("method"),
                    "seed": row.get("seed"),
                    "status": row.get("status"),
                    "wall_clock_s": _fmt(row.get("wall_clock_s")),
                    "epochs": (row.get("budget") or {}).get("epochs", ""),
                    "pi_mae": _fmt(pi.get("mae")),
                    "pi_rmse": _fmt(pi.get("rmse")),
                    "pi_max_ae": _fmt(pi.get("max_ae")),
                    "pi_clipped_mae": _fmt(pi_c.get("mae")),
                    "V_mae": _fmt(V.get("mae")),
                    "V_rel_mae": _fmt(Vrel.get("mae")),
                    "y0_abs_err": _fmt(m.get("y0_abs_err")),
                    "device": row.get("device", ""),
                })
        return out


def _fmt(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))
