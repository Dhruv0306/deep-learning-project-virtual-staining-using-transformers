"""
Training history utilities.

Provides visualization and CSV persistence helpers for CycleGAN training logs.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_history(history, model_dir=None):
    """
    Plot and save training history charts.

    Args:
        history (dict): Nested dict of epoch -> batch -> loss values.
        model_dir (str | None): Directory to save plots and CSVs.
    """
    if not history:
        print("No training history to visualize.")
        return

    epochs = list(history.keys())
    avg_loss_G = []
    avg_loss_D_A = []
    avg_loss_D_B = []

    # Aggregate batch losses into per-epoch averages.
    for epoch in epochs:
        epoch_data = history[epoch]
        batch_loss_G = [batch_data["Loss_G"] for batch_data in epoch_data.values()]
        batch_loss_D_A = [batch_data["Loss_D_A"] for batch_data in epoch_data.values()]
        batch_loss_D_B = [batch_data["Loss_D_B"] for batch_data in epoch_data.values()]
        avg_loss_G.append(np.mean(batch_loss_G))
        avg_loss_D_A.append(np.mean(batch_loss_D_A))
        avg_loss_D_B.append(np.mean(batch_loss_D_B))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("CycleGAN Training History", fontsize=16)

    axes[0, 0].plot(
        epochs, avg_loss_G, label="Generator Loss", color="blue", linewidth=2
    )
    axes[0, 0].plot(
        epochs, avg_loss_D_A, label="Discriminator A Loss", color="red", linewidth=2
    )
    axes[0, 0].plot(
        epochs, avg_loss_D_B, label="Discriminator B Loss", color="green", linewidth=2
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Average Losses per Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, avg_loss_G, color="blue", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Generator Loss")
    axes[0, 1].set_title("Generator Loss Over Time")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(
        epochs, avg_loss_D_A, label="Discriminator A", color="red", linewidth=2
    )
    axes[1, 0].plot(
        epochs, avg_loss_D_B, label="Discriminator B", color="green", linewidth=2
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Discriminator Loss")
    axes[1, 0].set_title("Discriminator Losses Comparison")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot batch-wise losses for the final epoch to show training stability.
    if epochs:
        last_epoch = epochs[-1]
        last_epoch_data = history[last_epoch]
        batch_nums = list(last_epoch_data.keys())
        last_epoch_loss_G = [last_epoch_data[batch]["Loss_G"] for batch in batch_nums]
        last_epoch_loss_D_A = [
            last_epoch_data[batch]["Loss_D_A"] for batch in batch_nums
        ]
        last_epoch_loss_D_B = [
            last_epoch_data[batch]["Loss_D_B"] for batch in batch_nums
        ]

        axes[1, 1].plot(batch_nums, last_epoch_loss_G, label="Generator", alpha=0.7)
        axes[1, 1].plot(
            batch_nums, last_epoch_loss_D_A, label="Discriminator A", alpha=0.7
        )
        axes[1, 1].plot(
            batch_nums, last_epoch_loss_D_B, label="Discriminator B", alpha=0.7
        )
        axes[1, 1].set_xlabel("Batch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].set_title(f"Batch-wise Losses (Epoch {last_epoch})")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    model_dir = (
        "data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models"
        if model_dir is None
        else model_dir
    )
    output_path = f"{model_dir}\\training_history.png"
    os.makedirs(model_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Training history plot saved to {output_path}")

    print("\n=== Training Summary ===")
    print(f"Total Epochs: {len(epochs)}")
    print(f"Final Generator Loss: {avg_loss_G[-1]:.4f}")
    print(f"Final Discriminator A Loss: {avg_loss_D_A[-1]:.4f}")
    print(f"Final Discriminator B Loss: {avg_loss_D_B[-1]:.4f}")
    print(f"Average Generator Loss: {np.mean(avg_loss_G):.4f}")
    print(f"Average Discriminator A Loss: {np.mean(avg_loss_D_A):.4f}")
    print(f"Average Discriminator B Loss: {np.mean(avg_loss_D_B):.4f}")

    plt.show()
    plt.close()


def save_history_to_csv(history, filename):
    """
    Save full training history to a CSV file.

    Args:
        history (dict): Nested dict of epoch -> batch -> loss values.
        filename (str): Output CSV path.
    """
    flattened_data = []
    for epoch, batches in history.items():
        for batch, losses in batches.items():
            row = {"Epoch": epoch, "Batch": batch}
            row.update(losses)
            flattened_data.append(row)

    df = pd.DataFrame(flattened_data)
    df.to_csv(filename, index=False)
    print(f"\nHistory saved to {filename}")


def append_history_to_csv(history, filename):
    """
    Append training history to a CSV file in chunks.

    Args:
        history (dict): Nested dict of epoch -> batch -> loss values.
        filename (str): Output CSV path.
    """
    if not history:
        return

    flattened_data = []
    for epoch, batches in history.items():
        for batch, losses in batches.items():
            row = {"Epoch": int(epoch), "Batch": int(batch)}
            row.update(losses)
            flattened_data.append(row)

    if not flattened_data:
        return

    df = pd.DataFrame(flattened_data)
    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    df.to_csv(filename, mode="a", header=write_header, index=False)
    print(f"\nHistory chunk appended to {filename}")


def load_history_from_csv(filename):
    """
    Load training history from a CSV file.

    Args:
        filename (str): CSV path.

    Returns:
        dict: Nested dict of epoch -> batch -> loss values.
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
            "Loss_G": float(row["Loss_G"]),
            "Loss_D_A": float(row["Loss_D_A"]),
            "Loss_D_B": float(row["Loss_D_B"]),
        }

    return history
