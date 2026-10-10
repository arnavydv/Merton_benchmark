"""Value network with the CRRA shape built in.

V(t,w) = exp(g_theta(t)) * w^(1-gamma) / (1-gamma).
The ratio V_w / (w V_ww) equals 1/gamma for any g_theta.
The network is scored on whether g_theta matches A*(T-t), not on whether
automatic differentiation can recover that ratio.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FlexibleValue(nn.Module):
    """V = exp(g(t, log w)) * w^(1-gamma)/(1-gamma).

    g may depend on wealth, so V_w/(w V_ww) is not forced to 1/gamma.
    The constant-in-wealth case is exactly the CRRA ansatz.
    """

    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = float(gamma)
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([torch.log(w.clamp(min=1e-8)), t], dim=1)
        g = self.net(x)
        p = 1.0 - self.gamma
        return torch.exp(g) * w.clamp(min=1e-8).pow(p) / p


class CRRAValue(nn.Module):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = float(gamma)
        self.g = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def log_f(self, t: torch.Tensor) -> torch.Tensor:
        return self.g(t)

    def forward(self, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p = 1.0 - self.gamma
        f = torch.exp(self.log_f(t))
        return f * w.clamp(min=1e-8).pow(p) / p
