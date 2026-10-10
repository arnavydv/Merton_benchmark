import json
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt

try:
    from forward_simulation_equations import forward_simulation
    from backward_simulation_equation import backward_simulation
    from neural_networks import pi_star, Z_net
    from loss_function import loss_function
    from utility_function import utility_function
except ImportError:
    from src.bsde.forward_simulation_equations import forward_simulation
    from src.bsde.backward_simulation_equation import backward_simulation
    from src.bsde.neural_networks import pi_star, Z_net
    from src.bsde.loss_function import loss_function
    from src.bsde.utility_function import utility_function


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = str(Path(__file__).resolve().parent / "config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cpu_state(module: torch.nn.Module) -> dict:
    """Detached CPU copy. state_dict() itself aliases the live weights."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _ensure_parent_dir_for_file(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def training_loop(
    epochs: int,
    model: torch.nn.Module,
    Z_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    mu: float,
    sigma: float,
    rate: float,
    w0: float,
    T: float,
    num_steps: int,
    num_paths: int,
    gamma: float,
    device: torch.device | None = None,
    # Model saving options
    save_best: bool = True,
    best_path: str = str(Path(__file__).resolve().parent / "saved_models" / "best_model.pt"),
    show_live_plot: bool = False,
):
    device = device or torch.device("cpu")
    model.to(device)
    Z_model.to(device)

    if device.type == "cuda":
        # Speed/behavior tweaks for stable training on NVIDIA GPUs
        torch.backends.cudnn.benchmark = True

    best_loss = float("inf")
    best_state = None
    history: list[float] = []

    # Live updating plot setup
    if show_live_plot:
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs Epoch (live)")
        ax.grid(True)

        # Make sure window appears promptly
        plt.show(block=False)

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)

        time_grid, wealth_grid, dw = forward_simulation(
            model=model,
            mu=mu,
            sigma=sigma,
            w0=w0,
            rate=rate,
            T=T,
            num_steps=num_steps,
            num_paths=num_paths,
        )

        Y = backward_simulation(
            wealth_grid=wealth_grid,
            time_grid=time_grid,
            dw=dw,
            Z_model=Z_model,
            gamma=gamma,
            utility_fn=utility_function,
        )

        loss = loss_function(Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(Z_model.parameters()), 1.0
        )
        optimizer.step()

        loss_val = loss.item()
        history.append(loss_val)

        # Track best model (using total loss)
        if save_best and loss_val < best_loss:
            best_loss = loss_val
            best_state = {
                "pi_star": _cpu_state(model),
                "Z_net": _cpu_state(Z_model),
                "epoch": epoch + 1,
                "loss": loss_val,
                "y0_mean": float(Y[:, 0].mean().detach()),
            }

        # Print every epoch with component breakdown
        if epoch % 10 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                t_probe = torch.zeros(1, 1, device=device)
                w_probe = torch.ones(1, 1, device=device)
                pi_at_one = model(t_probe, w_probe).item()
            print(
                f"Epoch {epoch + 1}/{epochs} | loss: {loss_val:.6e} | "
                f"Y0: {Y[:, 0].mean().item():.6f} | pi(0,1): {pi_at_one:.4f}"
            )

        # Update live plot
        if show_live_plot:
            xs = list(range(1, len(history) + 1))
            line.set_data(xs, history)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

    # Save best model only. The last epoch is kept only when it has the lowest loss.
    if save_best and best_state is not None:
        _ensure_parent_dir_for_file(best_path)
        best_state["config"] = {
            "mu": mu,
            "sigma": sigma,
            "rate": rate,
            "w0": w0,
            "T": T,
            "num_steps": num_steps,
            "num_paths": num_paths,
            "gamma": gamma,
        }
        torch.save(best_state, best_path)
        print(f"Best model (loss={best_loss}) saved to {best_path}")

    if show_live_plot:
        plt.ioff()

    return history


if __name__ == "__main__":
    cfg = load_config()
    torch.manual_seed(int(cfg["seed"]))
    mu = cfg["mu"]
    sigma = cfg["sigma"]
    rate = cfg["rate"]
    w0 = cfg["w0"]
    T = cfg["T"]
    num_steps = int(cfg["num_steps"])
    num_paths = int(cfg["num_paths"])
    gamma = cfg["gamma"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = pi_star().to(device)
    Z_model = Z_net().to(device)

    # With 200 steps, use slightly lower LR for stability
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(Z_model.parameters()), lr=cfg.get("lr", 1e-4)
    )

    training_loop(
        epochs=int(cfg.get("epochs", 5000)),
        model=model,
        Z_model=Z_model,
        optimizer=optimizer,
        mu=mu,
        sigma=sigma,
        rate=rate,
        w0=w0,
        T=T,
        num_steps=num_steps,
        num_paths=num_paths,
        gamma=gamma,
        device=device,
        show_live_plot=False,
    )