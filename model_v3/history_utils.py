"""
Utilities for saving, loading, and plotting v3 training history.

Component structure:
    1) save/append CSV helpers
    2) CSV reload helper
    3) visualization helper

CSV schema (Phase 2):
    Epoch, Batch, Loss_DiT_A2B, Loss_DiT_B2A, Loss_DiT, Loss_G_Adv, Loss_Cyc, Loss_Id,
    Loss_D_A, Loss_D_B, Lambda_Adv, Lambda_Id, Loss_Perceptual, Loss Total, GradNorm
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_history_to_csv_v3(history, filename):
    """
    Save the full v3 training history dict to a CSV file (overwrites).

    Args:
        history:  Nested dict keyed by epoch -> batch -> loss scalars.
        filename: Output CSV path.
    """
    flattened = _flatten_history(history)
    df = pd.DataFrame(flattened)
    df.to_csv(filename, index=False)
    print(f"\nHistory saved to {filename}")


def append_history_to_csv_v3(history, filename):
    """
    Append a v3 training history chunk to an existing CSV file.

    Creates the file with a header if it does not yet exist.  Used during
    training to flush history every N epochs without holding the full
    history dict in memory.

    Args:
        history:  Nested dict keyed by epoch -> batch -> loss scalars.
        filename: CSV path to append to.
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
    Reload a v3 training history dict from a CSV file.

    Reconstructs the nested epoch -> batch -> loss dict from the Phase 2
    CSV schema.  Missing columns default to 0.0 for backward compatibility
    with older checkpoints.

    Args:
        filename: Path to the CSV written by save/append helpers.

    Returns:
        Nested dict keyed by epoch (int) -> batch (int) -> loss scalars,
        or an empty dict if the file does not exist or is empty.
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

        # Phase 2: load all loss components from the CSV schema.
        history[epoch][batch] = {
            "Batch": batch,
            "Loss_DiT_A2B": float(row.get("Loss_DiT_A2B", 0.0)),
            "Loss_DiT_B2A": float(row.get("Loss_DiT_B2A", 0.0)),
            "Loss_DiT": float(row.get("Loss_DiT", 0.0)),
            "Loss_G_Adv": float(row.get("Loss_G_Adv", 0.0)),
            "Loss_Cyc": float(row.get("Loss_Cyc", 0.0)),
            "Loss_Id": float(row.get("Loss_Id", 0.0)),
            "Loss_D_A": float(row.get("Loss_D_A", 0.0)),
            "Loss_D_B": float(row.get("Loss_D_B", 0.0)),
            "Lambda_Adv": float(row.get("Lambda_Adv", 0.0)),
            "Lambda_Id": float(row.get("Lambda_Id", 0.0)),
            "Loss_Perceptual": float(row.get("Loss_Perceptual", 0.0)),
            "Loss Total": float(row.get("Loss Total", 0.0)),
            "GradNorm": float(row.get("GradNorm", 0.0)),
        }

    return history


def visualize_history_v3(history, model_dir=None):
    """
    Plot v3 training history and save a 3×3 PNG figure.

    Produces one subplot per loss component (denoising A2B/B2A/combined,
    adversarial, cycle, identity, discriminator A/B, gradient norm).
    The figure is saved to ``model_dir/training_history.png``.

    Args:
        history:   Nested dict keyed by epoch -> batch -> loss scalars.
        model_dir: Output directory.  Defaults to the standard v3 models
                   path when None.
    """
    if not history:
        print("No training history to visualize.")
        return

    epochs = list(history.keys())
    avg_loss_a2b = []
    avg_loss_b2a = []
    avg_loss_dit = []
    avg_loss_adv = []
    avg_loss_cyc = []
    avg_loss_id = []
    avg_loss_d_a = []
    avg_loss_d_b = []
    avg_loss_perc = []
    avg_loss_total = []
    avg_grad = []

    for epoch in epochs:
        epoch_data = history[epoch]
        # Phase 2: extract all loss components from history.
        loss_a2b_vals = [b.get("Loss_DiT_A2B", 0.0) for b in epoch_data.values()]
        loss_b2a_vals = [b.get("Loss_DiT_B2A", 0.0) for b in epoch_data.values()]
        loss_dit_vals = [b.get("Loss_DiT", 0.0) for b in epoch_data.values()]
        loss_adv_vals = [b.get("Loss_G_Adv", 0.0) for b in epoch_data.values()]
        loss_cyc_vals = [b.get("Loss_Cyc", 0.0) for b in epoch_data.values()]
        loss_id_vals = [b.get("Loss_Id", 0.0) for b in epoch_data.values()]
        loss_d_a_vals = [b.get("Loss_D_A", 0.0) for b in epoch_data.values()]
        loss_d_b_vals = [b.get("Loss_D_B", 0.0) for b in epoch_data.values()]
        perc_vals = [b.get("Loss_Perceptual", 0.0) for b in epoch_data.values()]
        loss_total_vals = [b.get("Loss Total", 0.0) for b in epoch_data.values()]
        grad_vals = [b.get("GradNorm", 0.0) for b in epoch_data.values()]

        avg_loss_a2b.append(np.mean(loss_a2b_vals) if loss_a2b_vals else 0.0)
        avg_loss_b2a.append(np.mean(loss_b2a_vals) if loss_b2a_vals else 0.0)
        avg_loss_dit.append(np.mean(loss_dit_vals) if loss_dit_vals else 0.0)
        avg_loss_adv.append(np.mean(loss_adv_vals) if loss_adv_vals else 0.0)
        avg_loss_cyc.append(np.mean(loss_cyc_vals) if loss_cyc_vals else 0.0)
        avg_loss_id.append(np.mean(loss_id_vals) if loss_id_vals else 0.0)
        avg_loss_d_a.append(np.mean(loss_d_a_vals) if loss_d_a_vals else 0.0)
        avg_loss_d_b.append(np.mean(loss_d_b_vals) if loss_d_b_vals else 0.0)
        avg_loss_perc.append(np.mean(perc_vals) if perc_vals else 0.0)
        avg_loss_total.append(np.mean(loss_total_vals) if loss_total_vals else 0.0)
        avg_grad.append(np.mean(grad_vals) if grad_vals else 0.0)

    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    fig.suptitle("DiT Phase 2 Training History", fontsize=16)

    # Row 1: Denoising losses
    axes[0, 0].plot(epochs, avg_loss_a2b, label="A2B", color="blue", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Denoising Loss (A2B)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, avg_loss_b2a, label="B2A", color="green", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Denoising Loss (B2A)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, avg_loss_dit, label="Combined", color="purple", linewidth=2)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].set_title("Denoising Loss (Combined)")
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: Phase 2 losses
    axes[1, 0].plot(epochs, avg_loss_adv, label="Adversarial", color="red", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Adversarial Loss (G)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, avg_loss_cyc, label="Cycle", color="orange", linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Cycle Loss")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs, avg_loss_id, label="Identity", color="brown", linewidth=2)
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Loss")
    axes[1, 2].set_title("Identity Loss")
    axes[1, 2].grid(True, alpha=0.3)

    # Row 3: Discriminator and auxiliary losses
    axes[2, 0].plot(epochs, avg_loss_d_a, label="D_A", color="cyan", linewidth=2)
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("Loss")
    axes[2, 0].set_title("Discriminator A Loss")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(epochs, avg_loss_d_b, label="D_B", color="magenta", linewidth=2)
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Loss")
    axes[2, 1].set_title("Discriminator B Loss")
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].plot(epochs, avg_grad, label="Grad Norm", color="black", linewidth=2)
    axes[2, 2].set_xlabel("Epoch")
    axes[2, 2].set_ylabel("Norm")
    axes[2, 2].set_title("Gradient Norm")
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    model_dir = (
        os.path.join(
            "data", "E_Staining_DermaRepo", "H_E-Staining_dataset", "models_v3"
        )
        if model_dir is None
        else model_dir
    )
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, "training_history.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Training history plot saved to {output_path}")


def _flatten_history(history):
    """
    Flatten a nested epoch -> batch -> loss dict into a list of row dicts.

    Each row contains ``Epoch``, ``Batch``, and all loss scalar keys from
    the batch dict, suitable for constructing a pandas DataFrame.
    """
    flattened = []
    for epoch, batches in history.items():
        for batch, losses in batches.items():
            row = {"Epoch": int(epoch), "Batch": int(batch)}
            row.update(losses)
            flattened.append(row)
    return flattened
