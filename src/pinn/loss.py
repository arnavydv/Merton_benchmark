import torch


def hjb_residual(model, t, w, r, sigma, mu):
    """Mean square of the reduced Merton residual.

    R = V_t + r W V_w - (1/2) ((mu-r)/sigma)^2 (V_w^2 / V_ww)

    The quotient is the residual that evaluation reports. Dividing the
    equivalent multiplied form by |V_ww| recovers R, with a floor so a
    vanishing second derivative cannot hide a large numerator.
    """
    v = model(w, t)
    v_t = torch.autograd.grad(v, t, create_graph=True, grad_outputs=torch.ones_like(v))[0]
    v_x = torch.autograd.grad(v, w, create_graph=True, grad_outputs=torch.ones_like(v))[0]
    v_xx = torch.autograd.grad(v_x, w, create_graph=True, grad_outputs=torch.ones_like(v_x))[0]
    kappa = ((mu - r) / sigma) ** 2
    multiplied = v_t * v_xx + r * w * v_xx * v_x - 0.5 * kappa * v_x ** 2
    residual = multiplied / (v_xx.abs() + 1e-4)
    return torch.mean(residual ** 2), v_t, v_x, v_xx


def terminal_loss(model, w_tc, t_tc, v_tc):
    """Relative MSE against U(W). Matches level, not a z-scored shape."""
    v_pred = model(w_tc, t_tc)
    scale = v_tc.abs() + 1e-6
    return torch.mean(((v_pred - v_tc) / scale) ** 2)


def concavity_loss(v_xx):
    """Soft V_ww < 0. CRRA value is concave; this is not a pi* penalty."""
    return torch.mean(torch.relu(v_xx) ** 2)


def monotonocity_loss(v_x):
    """Soft V_w > 0. More wealth is better for this utility."""
    return torch.mean(torch.relu(-v_x) ** 2)


def total_loss(concavity, monotonicity, terminal, hjb):
    return hjb + 10.0 * terminal + 0.1 * monotonicity + 0.1 * concavity
