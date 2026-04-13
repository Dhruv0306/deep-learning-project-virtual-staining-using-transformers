"""
trainModel.py — interactive training entry point for all model variants.

Prompts for run-time parameters, creates a timestamped output directory,
dispatches to the selected training loop, and persists training history as
both a PNG figure and a CSV file.

Supported model versions:
    1 — Hybrid CycleGAN/UVCGAN baseline  (model_v1, get_default_config)
    2 — True UVCGAN v2                   (model_v2, get_8gb_config)
    3 — CycleDiT latent diffusion        (model_v3, get_dit_8gb_config)
    4 — Transformer + PatchNCE           (model_v4, get_v4_8gb_config)

Resume support:
    Entering 'resume' at the first prompt requires a path to an existing
    .pth checkpoint.  The run directory is inferred from the checkpoint's
    parent folder so all artifacts stay co-located.  num_epochs must exceed
    the epoch stored in the checkpoint.
"""

import os
import re
from datetime import datetime

from shared.history_utils import save_history_to_csv, visualize_history
from model_v1.training_loop import train_v1
from config import get_8gb_config, get_v4_8gb_config, get_default_config
from model_v2.training_loop import train_v2
from model_v4.training_loop import train_v4


def _parse_checkpoint_epoch(checkpoint_path: str):
    """
    Extract the saved epoch number from a checkpoint filename.

    Supports names like:
        checkpoint_epoch_120.pth
        final_checkpoint_epoch_120.pth

    Returns:
        int | None: Parsed epoch if present, else None.
    """
    filename = os.path.basename(checkpoint_path)
    match = re.search(r"(?:final_)?checkpoint_epoch_(\d+)\.pth$", filename)
    if not match:
        return None
    return int(match.group(1))


def main():
    """
    Collect user inputs, run the selected training loop, and save artifacts.

    Prompts (in order):
        mode        — 'new' starts a fresh run; 'resume' continues from a checkpoint.
        checkpoint  — (resume only) path to an existing .pth file.
        epoch_size  — samples drawn per epoch.
        num_epochs  — total epochs to train (must exceed checkpoint epoch on resume).
        test_size   — images exported during final test-set inference.
        version     — 1 / 2 / 3 / 4 selects the training loop.

    Returns:
        tuple: ``(history, G_AB, G_BA, D_A, D_B)``.

        v1/v2/v4 — trained generators and discriminators from the loop.
        v3       — tuple slots are remapped for API compatibility:
                       G_AB <- dit_model, G_BA <- ema_model,
                       D_A  <- cond_encoder, D_B <- None.
    """

    mode = input("Start new training or resume? Enter 'new' or 'resume': ").strip().lower()
    if mode not in {"new", "resume"}:
        raise ValueError("Invalid mode. Please enter 'new' or 'resume'.")

    resume_checkpoint = None
    is_resume = mode == "resume"
    if is_resume:
        resume_checkpoint = input("Enter resume checkpoint path (.pth): ").strip().strip('"')
        if not resume_checkpoint:
            raise ValueError("Resume mode requires a checkpoint path.")
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_checkpoint}"
            )

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

    dataset_root = os.path.join("data", "E_Staining_DermaRepo", "H_E-Staining_dataset")

    if is_resume and resume_checkpoint is not None:
        resume_epoch = _parse_checkpoint_epoch(resume_checkpoint)
        if resume_epoch is not None and num_epochs <= resume_epoch:
            raise ValueError(
                f"Number of epochs ({num_epochs}) must be greater than checkpoint epoch ({resume_epoch})."
            )

    if model_version == 2:
        model_dir = (
            os.path.dirname(resume_checkpoint)
            if is_resume and resume_checkpoint
            else os.path.join(
                dataset_root,
                f"models_v2_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
            )
        )
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model directory: {model_dir}")
        val_dir = os.path.join(model_dir, "validation_images")
        os.makedirs(val_dir, exist_ok=True)
        print(f"Validation image directory: {val_dir}")
        cfg = get_8gb_config()
        cfg.training.save_checkpoint_every = (
            20
            if cfg.training.save_checkpoint_every <= 0
            else cfg.training.save_checkpoint_every
        )
        history, G_AB, G_BA, D_A, D_B = train_v2(
            epoch_size=epoch_size,
            num_epochs=num_epochs,
            model_dir=model_dir,
            val_dir=val_dir,
            test_size=test_size,
            resume_checkpoint=resume_checkpoint,
            cfg=cfg,
        )
    elif model_version == 3:
        from config import get_dit_8gb_config
        from model_v3.training_loop import train_v3

        model_dir = (
            os.path.dirname(resume_checkpoint)
            if is_resume and resume_checkpoint
            else os.path.join(
                dataset_root,
                f"models_v3_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
            )
        )
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model directory: {model_dir}")
        val_dir = os.path.join(model_dir, "validation_images")
        os.makedirs(val_dir, exist_ok=True)
        print(f"Validation image directory: {val_dir}")
        cfg = get_dit_8gb_config()
        cfg.training.save_checkpoint_every = (
            20
            if cfg.training.save_checkpoint_every <= 0
            else cfg.training.save_checkpoint_every
        )
        history, dit_model, ema_model, cond_encoder = train_v3(
            epoch_size=epoch_size,
            num_epochs=num_epochs,
            model_dir=model_dir,
            val_dir=val_dir,
            test_size=test_size,
            resume_checkpoint=resume_checkpoint,
            cfg=cfg,
        )
        G_AB = dit_model
        G_BA = ema_model
        D_A = cond_encoder
        D_B = None
    elif model_version == 4:
        model_dir = (
            os.path.dirname(resume_checkpoint)
            if is_resume and resume_checkpoint
            else os.path.join(
                dataset_root,
                f"models_v4_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
            )
        )
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model directory: {model_dir}")
        val_dir = os.path.join(model_dir, "validation_images")
        os.makedirs(val_dir, exist_ok=True)
        print(f"Validation image directory: {val_dir}")
        cfg = get_v4_8gb_config()
        cfg.training.test_size = test_size
        cfg.training.save_checkpoint_every = (
            20
            if cfg.training.save_checkpoint_every <= 0
            else cfg.training.save_checkpoint_every
        )
        history, G_AB, G_BA, D_A, D_B = train_v4(
            epoch_size=epoch_size,
            num_epochs=num_epochs,
            model_dir=model_dir,
            val_dir=val_dir,
            resume_checkpoint=resume_checkpoint,
            cfg=cfg,
        )
    elif model_version == 1:
        model_dir = (
            os.path.dirname(resume_checkpoint)
            if is_resume and resume_checkpoint
            else os.path.join(
                dataset_root,
                f"models_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
            )
        )
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model directory: {model_dir}")
        val_dir = os.path.join(model_dir, "validation_images")
        os.makedirs(val_dir, exist_ok=True)
        print(f"Validation image directory: {val_dir}")
        cfg = get_default_config(model_version=model_version)
        cfg.training.test_size = test_size
        cfg.training.save_checkpoint_every = (
            20
            if cfg.training.save_checkpoint_every <= 0
            else cfg.training.save_checkpoint_every
        )
        history, G_AB, G_BA, D_A, D_B = train_v1(
            epoch_size=epoch_size,
            num_epochs=num_epochs,
            model_dir=model_dir,
            val_dir=val_dir,
            test_size=test_size,
            resume_checkpoint=resume_checkpoint,
            cfg=cfg,
        )
    else:
        raise ValueError("Invalid model version selected. Please choose 1, 2, 3, or 4.")

    # v3 uses its own history helpers; all other versions use the shared ones.
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
