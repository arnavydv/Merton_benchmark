import numpy as np
import torch


def brownian_increments(num_paths, num_steps, dt, device=None):
    """Independent Brownian increments. These are noise, not a state variable."""
    device = device or torch.device("cpu")
    return torch.randn(num_paths, num_steps, device=device) * np.sqrt(dt)
