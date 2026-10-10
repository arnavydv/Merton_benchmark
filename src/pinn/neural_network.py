import torch 
import torch.nn as nn

class VanillaPINN(nn.Module):
    def __init__(self):
        super(VanillaPINN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, w, t):
        # Log-transform wealth: handles the power-law scale of CRRA naturally
        # w ∈ [0.1, 2.0] → log(w) ∈ [-2.3, 0.7] (nice normalized range)
        log_w = torch.log(w.clamp(min=1e-8))
        # Concatenate features
        x = torch.cat([log_w, t], dim=1)
        # Network forward pass
        raw_output = self.network(x)
        return raw_output