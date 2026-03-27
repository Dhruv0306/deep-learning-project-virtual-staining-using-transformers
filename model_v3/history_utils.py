"""
V3-specific training history utilities.

Component structure:
    1) save/append CSV helpers
    2) CSV reload helper
    3) visualization helper

CSV schema:
    Epoch, Batch, Loss_DiT, Loss_Perceptual, GradNorm

History structure in memory:
    history[epoch][batch] = {
        "Batch": int,
        "Loss_DiT": float,
        "Loss_Perceptual": float,
        "GradNorm": float,
    }
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_history_to_csv_v3(history, filename):
    """
    Save v3 training history to CSV.
    """
    flattened = _flatten_history(history)
    df = pd.DataFrame(flattened)
    df.to_csv(filename, index=False)
    print(f"\nHistory saved to {filename}")


def append_history_to_csv_v3(history, filename):
    """
    Append v3 training history to CSV in chunks.
    """
    if not history:
        return

    flattened = _flatten_history(history)
    if not flattened:
        return

    df = pd.DataFrame(flattened)
    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    df.to_csv(filename, mode="a", header=write_header, index=False)
    print(f"\nHistory chunk appended to {filename}")


def load_history_from_csv_v3(filename):
    """
    Load v3 training history from CSV.
    """
    if not os.path.exists(filename):
        return {}

    df = pd.read_csv(filename)
    if df.empty:
        return {}

    history = {}
    for _, row in df.iterrows():
        epoch = int(row["Epoch"])
        batch = int(row["Batch"])
        history.setdefault(epoch, {})
        history[epoch][batch] = {
            "Batch": batch,
            "Loss_DiT": float(row["Loss_DiT"]),
            "Loss_Perceptual": float(row["Loss_Perceptual"]),
            "GradNorm": float(row["GradNorm"]),
        }

    return history


def visualize_history_v3(history, model_dir=None):
    """
    Plot v3 training history charts.

    Dataflow:
        history dict -> per-epoch means -> 3 line plots:
        DiT loss, perceptual loss, gradient norm.
    """
    if not history:
        print("No training history to visualize.")
        return

    epochs = list(history.keys())
    avg_loss = []
    avg_loss_perc = []
    avg_grad = []

    for epoch in epochs:
        epoch_data = history[epoch]
        loss_vals = [b["Loss_DiT"] for b in epoch_data.values()]
        perc_vals = [b["Loss_Perceptual"] for b in epoch_data.values()]
        grad_vals = [b["GradNorm"] for b in epoch_data.values()]
        avg_loss.append(np.mean(loss_vals))
        avg_loss_perc.append(np.mean(perc_vals))
        avg_grad.append(np.mean(grad_vals))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("DiT Training History", fontsize=16)

    axes[0].plot(epochs, avg_loss, label="DiT Loss", color="blue", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("DiT Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, avg_loss_perc, label="Perceptual Loss", color="green", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Perceptual Loss")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, avg_grad, label="Grad Norm", color="purple", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Norm")
    axes[2].set_title("Gradient Norm")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    model_dir = (
        os.path.join("data", "E_Staining_DermaRepo", "H_E-Staining_dataset", "models_v3")
        if model_dir is None
        else model_dir
    )
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, "training_history.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Training history plot saved to {output_path}")


def _flatten_history(history):
    flattened = []
    for epoch, batches in history.items():
        for batch, losses in batches.items():
            row = {"Epoch": int(epoch), "Batch": int(batch)}
            row.update(losses)
            flattened.append(row)
    return flattened
