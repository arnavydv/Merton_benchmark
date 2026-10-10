"""Terminal-mean-square deep BSDE for undiscounted terminal utility.

Y starts at a trainable scalar and is stepped forward,
    Y <- Y + Z dB,
with no driver, because there is no running cost. The loss is
E[(Y_T - U(W_T))^2]. The policy enters only through W.
This is the Han-Jentzen-E scheme for this problem. It is not the
martingale objective that assigns Y_T = U and scores Y_0.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, t, w):
        x = torch.cat([t, torch.log(w.clamp(min=1e-8))], dim=1)
        return 2.0 * torch.tanh(self.net(x))


class ZNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, t, w):
        x = torch.cat([t, torch.log(w.clamp(min=1e-8))], dim=1)
        return self.net(x)


def rollout(policy, z_net, y0, dw, mu, sigma, rate, w0, T, gamma):
    n_paths, n_steps = dw.shape
    dt = T / n_steps
    w = torch.full((n_paths, 1), w0, device=dw.device, dtype=dw.dtype)
    y = y0.expand(n_paths, 1)
    p = 1.0 - gamma
    for i in range(n_steps):
        t = torch.full((n_paths, 1), i * dt, device=dw.device, dtype=dw.dtype)
        pi = policy(t, w)
        z = z_net(t, w)
        growth = (rate + pi * (mu - rate) - 0.5 * pi ** 2 * sigma ** 2) * dt
        shock = pi * sigma * dw[:, i:i + 1]
        y = y + z * dw[:, i:i + 1]
        w = w * torch.exp(growth + shock)
    terminal = w.clamp(min=1e-8).pow(p) / p
    mse = torch.mean((y - terminal) ** 2)
    # The mean-square term trains (Y0, Z). The utility term trains the policy.
    # Without it, Z can match U(W_T) for a poor control and the weight does not move.
    loss = mse - terminal.mean()
    return loss, y.mean(), terminal.mean()
