"""
Training entry point for the CycleGAN pipeline.

Prompts for key training parameters, sets up output folders, launches the training
loop, and persists training history for later inspection.
"""

import os
from datetime import datetime

from history_utils import save_history_to_csv, visualize_history
from training_loop import train
from config import get_8gb_config
from training_loop_v2 import train_v2


def main():
    """
    Collect user inputs, run training, and save results.

    Returns:
        tuple: (history, G_AB, G_BA, D_A, D_B)

    Notes:
        model_version=1 launches the hybrid v1 training loop.
        model_version=2 launches the true UVCGAN v2 loop with get_8gb_config().
    """
    
    # User-controlled training parameters.
    epoch_size = int(input("Enter Epoch Size: "))
    num_epochs = int(input("Enter Number of Epochs: "))
    test_size = float(input("Enter Test Size: "))
    model_version = int(input("Enter model version you want 1 for Hybrid and 2 for true UVCGAN: "))

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
            cfg=cfg
        )
    else :
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
        history, G_AB, G_BA, D_A, D_B = train(
            epoch_size=epoch_size,
            num_epochs=num_epochs,
            model_dir=model_dir,
            val_dir=val_dir,
            test_size=test_size,
        )

    # Persist training history in both visual and CSV form.
    visualize_history(history, model_dir=model_dir)
    save_history_to_csv(history, os.path.join(model_dir, "training_history.csv"))

    return history, G_AB, G_BA, D_A, D_B


if __name__ == "__main__":
    main()
