import torch

try:
    from utility_function import utility_function
except ImportError:
    from src.bsde.utility_function import utility_function


def _ensure_col(x):
    if isinstance(x, torch.Tensor) and x.ndim == 1:
        return x.unsqueeze(1)
    return x


def backward_simulation(
    wealth_grid,
    time_grid,
    dw,
    Z_model,
    gamma,
    utility_fn=utility_function,
):
    """Martingale BSDE for undiscounted terminal utility.

    With no running cost, Y_t = E[U(W_T) | F_t] satisfies dY = Z dB.
    The closed-form Merton growth rate is not used. Z is evaluated at
    (t, wealth), the Markov state of the control problem.

    Euler step, backward:
        Y_n = Y_{n+1} - Z(t_n, W_n) * dB_n
        Y_T = U(W_T)
    """
    if not isinstance(wealth_grid, torch.Tensor):
        wealth_grid = torch.as_tensor(wealth_grid)
    if not isinstance(dw, torch.Tensor):
        dw = torch.as_tensor(dw)

    num_paths, num_steps_plus_1 = wealth_grid.shape
    num_steps = num_steps_plus_1 - 1

    Y_T = utility_fn(wealth_grid[:, -1], gamma).view(-1)
    Y_list = [None] * (num_steps + 1)
    Y_list[-1] = Y_T

    Y_next = Y_T
    for n in reversed(range(num_steps)):
        t_n = torch.full(
            (num_paths, 1),
            float(time_grid[n]),
            dtype=wealth_grid.dtype,
            device=wealth_grid.device,
        )
        W_n = _ensure_col(wealth_grid[:, n])
        dw_n = dw[:, n]
        z_n = Z_model(t_n, W_n)
        z_n = z_n.squeeze(1) if z_n.ndim == 2 and z_n.shape[1] == 1 else z_n
        Y_curr = Y_next - z_n * dw_n
        Y_list[n] = Y_curr
        Y_next = Y_curr

    return torch.stack(Y_list, dim=1)
