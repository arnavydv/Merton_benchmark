import numpy as np
import torch

try:
    from brownian_motion import brownian_increments
except ImportError:
    from src.bsde.brownian_motion import brownian_increments


def forward_simulation(model, mu, sigma, w0, rate, T, num_steps, num_paths):
    """Simulate Merton wealth under a policy pi(t, W).

    The network is queried on calendar time and current wealth, never on
    the Brownian path. Each step is the exact geometric increment for a
    frozen policy, so wealth stays positive without a softplus clip.

        dW = [r + pi (mu - r)] W dt + pi sigma W dB
    """
    device = next(model.parameters()).device
    dt = T / num_steps
    dw = brownian_increments(num_paths, num_steps, dt, device=device)
    time_grid = np.linspace(0, T, num_steps + 1)

    sigma_t = torch.as_tensor(sigma, dtype=dw.dtype, device=device)
    mu_t = torch.as_tensor(mu, dtype=dw.dtype, device=device)
    rate_t = torch.as_tensor(rate, dtype=dw.dtype, device=device)
    dt_t = torch.as_tensor(dt, dtype=dw.dtype, device=device)

    w = torch.as_tensor(w0, dtype=dw.dtype, device=device).expand(num_paths)
    wealth_list = [w]

    for i in range(num_steps):
        t_col = torch.full((num_paths, 1), float(time_grid[i]), dtype=dw.dtype, device=device)
        w_col = wealth_list[-1].view(-1, 1)
        pi = model(t_col, w_col)

        # Log-Euler: exact GBM step when pi is held constant on [t_i, t_{i+1}].
        growth = (
            rate_t
            + pi * (mu_t - rate_t)
            - 0.5 * (pi ** 2) * (sigma_t ** 2)
        ) * dt_t
        shock = pi * sigma_t * dw[:, i : i + 1]
        wealth_next = w_col * torch.exp(growth + shock)
        wealth_list.append(wealth_next.squeeze(1))

    wealth_grid = torch.stack(wealth_list, dim=1)
    return time_grid, wealth_grid, dw
