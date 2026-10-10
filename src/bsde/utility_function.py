import torch

def utility_function(Wealth, gamma):
    """CRRA utility: U(W) = W^(1-γ)/(1-γ) for γ≠1, U(W) = log(W) for γ=1."""
    if gamma == 1:
        return torch.log(Wealth) if isinstance(Wealth, torch.Tensor) else torch.tensor(Wealth).log()
    return (Wealth ** (1 - gamma)) / (1 - gamma)
