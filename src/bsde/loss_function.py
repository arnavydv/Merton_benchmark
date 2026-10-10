import torch


def loss_function(Y: torch.Tensor) -> torch.Tensor:
    """Control objective plus BSDE consistency.

    W_0 is the same on every path, so Y_0 = V(0, W_0) must be one number.
    The policy enters only through simulated wealth. Minimising -E[Y_0]
    maximises expected terminal utility. The variance term trains Z so the
    backward step is a martingale. There is no penalty toward a known pi*.
    """
    if not isinstance(Y, torch.Tensor):
        Y = torch.as_tensor(Y)
    y0 = Y[:, 0]
    return -y0.mean() + y0.var(unbiased=False)
