import torch 
from torch import nn
import json
from pathlib import Path

with open(Path(__file__).resolve().parent / "config.json", "r") as file:
    config = json.load(file)

input_dim_pi=config["neural_dim_pi_model"]["input_dimension"]
output_dim_pi=config["neural_dim_pi_model"]["output_dimension"]
hidden_dim_pi=config["neural_dim_pi_model"]["hidden_dimension"]
input_dim_z=config["neural_dim_z_model"]["input_dimension"]
output_dim_z=config["neural_dim_z_model"]["output_dimension"]
hidden_dim_z=config["neural_dim_z_model"]["hidden_dimension"]

class pi_star(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim_pi, hidden_dim_pi),
            nn.ReLU(),
            nn.Linear(hidden_dim_pi, hidden_dim_pi),
            nn.ReLU(),
            nn.Linear(hidden_dim_pi, hidden_dim_pi),
            nn.ReLU(),
            nn.Linear(hidden_dim_pi, hidden_dim_pi),
            nn.ReLU(),
            nn.Linear(hidden_dim_pi, output_dim_pi),
        )

    def forward(self, t: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Policy network pi(t, W).

        Expected shapes from forward_simulation_equations.forward_simulation():
          - t: (batch, 1)
          - W: (batch, 1)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if W.dim() == 1:
            W = W.unsqueeze(1)

        # Log-wealth keeps the feature O(1) on a multiplicative domain.
        # The SDE itself is still integrated in wealth, not in log-wealth.
        log_w = torch.log(W.clamp(min=1e-8))
        input_tensor = torch.cat([t, log_w], dim=1)
        raw = self.network(input_tensor)
        # Interior leverage bound. pi* for the paper parameters is about 0.5,
        # so the bound does not sit on the optimum.
        return 2.0 * torch.tanh(raw)
    
class Z_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim_z, hidden_dim_z),
            nn.ReLU(),
            nn.Linear(hidden_dim_z, hidden_dim_z),
            nn.ReLU(),
            nn.Linear(hidden_dim_z, hidden_dim_z),
            nn.ReLU(),
            nn.Linear(hidden_dim_z, hidden_dim_z),
            nn.ReLU(),
            nn.Linear(hidden_dim_z, output_dim_z),
        )

    def forward(self, t, W):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if W.dim() == 1:
            W = W.unsqueeze(1)
        log_w = torch.log(W.clamp(min=1e-8))
        input_tensor = torch.cat([t, log_w], dim=1)
        return self.network(input_tensor)