"""
Training entry point for all model variants in this repository.

This script collects run-time parameters from stdin, creates a timestamped
output directory, dispatches to the selected training loop, and persists the
training history as both a figure and CSV.

Supported model versions:
    1 -> Hybrid CycleGAN/UVCGAN baseline (model_v1)
    2 -> True UVCGAN v2 (model_v2)
    3 -> DiT diffusion pipeline (model_v3)
    4 -> Transformer + PatchNCE (model_v4)
"""

import os
from datetime import datetime

from shared.history_utils import save_history_to_csv, visualize_history
from model_v1.training_loop import train_v1
from config import get_8gb_config, get_v4_8gb_config
from model_v2.training_loop import train_v2
from model_v4.training_loop import train_v4


def main():
    """
    Collect user inputs, run the selected training loop, and save artifacts.

    Returns:
        tuple: ``(history, G_AB, G_BA, D_A, D_B)``.

        For model versions 1, 2, and 4, these are the CycleGAN-style generators
        and discriminators returned by the corresponding training loop.

        For model version 3, this return shape is preserved for compatibility
        with downstream tooling by mapping:
            - ``G_AB`` <- ``dit_model``
            - ``G_BA`` <- ``ema_model``
            - ``D_A``  <- ``cond_encoder``
            - ``D_B``  <- ``None``

    Notes:
        model_version=1 launches ``train_v1``.
        model_version=2 launches ``train_v2`` with ``get_8gb_config()``.
        model_version=3 launches ``train_v3`` with ``get_dit_8gb_config()``.
        model_version=4 launches ``train_v4`` with ``get_v4_8gb_config()``.
        All model directories are timestamped to keep each run isolated.
    """

    # User-controlled training parameters.
    epoch_size = int(input("Enter Epoch Size: "))
    num_epochs = int(input("Enter Number of Epochs: "))
    test_size = int(input("Enter Test Size: "))
    model_version = int(
        input(
            "Enter model version you want 1 for Hybrid, 2 for true UVCGAN, "
            "3 for DiT diffusion, 4 for v4 (Transformer + NCE): "
        )
    )

    history = None
    G_AB = None
    G_BA = None
    D_A = None
    D_B = None

    # Run the selected training loop.
    dataset_root = os.path.join("data", "E_Staining_DermaRepo", "H_E-Staining_dataset")
    if model_version == 2:
        # Create a timestamped model directory so each run is isolated.
        model_dir = os.path.join(
            dataset_root,
            f"models_v2_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        )
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model directory: {model_dir}")
        # Validation images are saved per epoch for quick qualitative checks.
        val_dir = os.path.join(model_dir, "validation_images")
        os.makedirs(val_dir, exist_ok=True)
        print(f"Validation image directory: {val_dir}")
        cfg = get_8gb_config()
        history, G_AB, G_BA, D_A, D_B = train_v2(
            epoch_size=epoch_size,
            num_epochs=num_epochs,
            model_dir=model_dir,
            val_dir=val_dir,
            test_size=test_size,
            cfg=cfg,
        )
    elif model_version == 3:
        from config import get_dit_8gb_config
        from model_v3.training_loop import train_v3

        model_dir = os.path.join(
            dataset_root,
            f"models_v3_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        )
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model directory: {model_dir}")
        val_dir = os.path.join(model_dir, "validation_images")
        os.makedirs(val_dir, exist_ok=True)
        print(f"Validation image directory: {val_dir}")
        cfg = get_dit_8gb_config()
        history, dit_model, ema_model, cond_encoder = train_v3(
            epoch_size=epoch_size,
            num_epochs=num_epochs,
            model_dir=model_dir,
            val_dir=val_dir,
            test_size=test_size,
            cfg=cfg,
        )
        G_AB = dit_model
        G_BA = ema_model
        D_A = cond_encoder
        D_B = None
    elif model_version == 4:
        model_dir = os.path.join(
            dataset_root,
            f"models_v4_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        )
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model directory: {model_dir}")
        val_dir = os.path.join(model_dir, "validation_images")
        os.makedirs(val_dir, exist_ok=True)
        print(f"Validation image directory: {val_dir}")
        cfg = get_v4_8gb_config()
        cfg.training.test_size = test_size
        history, G_AB, G_BA, D_A, D_B = train_v4(
            epoch_size=epoch_size,
            num_epochs=num_epochs,
            model_dir=model_dir,
            val_dir=val_dir,
            cfg=cfg,
        )
    else:
        # Create a timestamped model directory so each run is isolated.
        model_dir = os.path.join(
            dataset_root,
            f"models_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        )
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model directory: {model_dir}")
        # Validation images are saved per epoch for quick qualitative checks.
        val_dir = os.path.join(model_dir, "validation_images")
        os.makedirs(val_dir, exist_ok=True)
        print(f"Validation image directory: {val_dir}")
        history, G_AB, G_BA, D_A, D_B = train_v1(
            epoch_size=epoch_size,
            num_epochs=num_epochs,
            model_dir=model_dir,
            val_dir=val_dir,
            test_size=test_size,
        )

    # Persist training history in both visual and CSV form.
    history_visualizer = visualize_history
    history_saver = save_history_to_csv
    if model_version == 3:
        from model_v3.history_utils import visualize_history_v3, save_history_to_csv_v3

        history_visualizer = visualize_history_v3
        history_saver = save_history_to_csv_v3
    history_visualizer(history, model_dir=model_dir)
    history_saver(history, os.path.join(model_dir, "training_history.csv"))

    return history, G_AB, G_BA, D_A, D_B


if __name__ == "__main__":
    main()
