"""Train the value PINN. The only file to run in this folder.

Writes src/pinn/saved_models/best_model.pt whenever the total loss
improves. The last epoch is not saved unless it is the best.
"""
import json
from pathlib import Path

import torch

try:
    from collocation_points import collocation_points, terminal_points
    from loss import concavity_loss, hjb_residual, monotonocity_loss, terminal_loss, total_loss
    from neural_network import VanillaPINN
except ImportError:
    from src.pinn.collocation_points import collocation_points, terminal_points
    from src.pinn.loss import concavity_loss, hjb_residual, monotonocity_loss, terminal_loss, total_loss
    from src.pinn.neural_network import VanillaPINN


def _batch(cfg, device):
    w, t = collocation_points(cfg["n_collocation"], cfg["T"], cfg["w_max"], cfg["w_min"])
    w_tc, t_tc, v_tc = terminal_points(
        cfg["n_terminal"], cfg["T"], cfg["w_max"], cfg["w_min"], cfg["gamma"]
    )
    return (
        w.to(device),
        t.to(device),
        w_tc.to(device),
        t_tc.to(device),
        v_tc.to(device),
    )


def _objective(model, w, t, w_tc, t_tc, v_tc, r, sigma, mu):
    hjb, _, v_x, v_xx = hjb_residual(model, t, w, r, sigma, mu)
    terminal = terminal_loss(model, w_tc, t_tc, v_tc)
    total = total_loss(concavity_loss(v_xx), monotonocity_loss(v_x), terminal, hjb)
    return total, hjb, terminal


def train():
    here = Path(__file__).resolve().parent
    with open(here / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    torch.manual_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaPINN().to(device)
    r = torch.tensor(cfg["rate"], device=device)
    mu = torch.tensor(cfg["mu"], device=device)
    sigma = torch.tensor(cfg["sigma"], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    save_dir = here / "saved_models"
    save_dir.mkdir(exist_ok=True)
    best_path = save_dir / "best_model.pt"
    best_loss = float("inf")

    def save_best(epoch, loss_value):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": loss_value,
                "config": cfg,
            },
            best_path,
        )

    print(f"PINN on {device} | Adam epochs={cfg['epochs']} | lr={cfg['lr']}")
    for epoch in range(int(cfg["epochs"])):
        model.train()
        opt.zero_grad()
        w, t, w_tc, t_tc, v_tc = _batch(cfg, device)
        total, hjb, terminal = _objective(model, w, t, w_tc, t_tc, v_tc, r, sigma, mu)
        total.backward()
        opt.step()
        current = total.item()
        if current < best_loss:
            best_loss = current
            save_best(epoch + 1, best_loss)
        if epoch % 100 == 0 or epoch == int(cfg["epochs"]) - 1:
            print(
                f"Epoch {epoch:04d} | total={current:.6e} | pde={hjb.item():.6e} | "
                f"terminal={terminal.item():.6e} | best={best_loss:.6e}"
            )

    print(f"Best model saved to {best_path} | loss={best_loss:.6e}")
    return best_path


if __name__ == "__main__":
    train()
