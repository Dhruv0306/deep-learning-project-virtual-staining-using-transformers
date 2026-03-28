"""
Training loop for the v3 DiT diffusion model.

Component structure:
    1) LR helper
    2) grad-norm helper
    3) validation runner
    4) train_v3 main loop

Core per-batch shape flow:
    real_A:(N,3,256,256), real_B:(N,3,256,256)
      -> z0 via VAE encode: (N,4,32,32)
      -> z_t via add_noise:  (N,4,32,32)
    -> model_pred from DiT: (N,4,32,32)
"""

from __future__ import annotations

import copy
import math
import os
from typing import Callable, Optional

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from model_v2.losses import VGGPerceptualLossV2
from config import UVCGANConfig, get_dit_config
from shared.data_loader import getDataLoader
from shared.EarlyStopping import EarlyStopping
from model_v3.history_utils import append_history_to_csv_v3, load_history_from_csv_v3
from shared.metrics import MetricsCalculator
from shared.validation import save_images_with_title

from model_v3.generator import getGeneratorV3
from model_v3.noise_scheduler import DDPMScheduler, DDIMSampler
from model_v2.generator import init_weights_v2
from model_v3.vae_wrapper import VAEWrapper
from model_v3.losses import compute_diffusion_loss


def _make_cosine_warmup_lambda(
    warmup: int, total: int, lr_min_ratio: float
) -> Callable[[int], float]:
    """
    Cosine decay with linear warmup.
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return max(1e-8, (epoch + 1) / max(1, warmup))
        if total <= warmup:
            return lr_min_ratio
        progress = (epoch - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

    return lr_lambda


def _global_grad_norm(parameters) -> float:
    grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack([g.norm() for g in grads])))


def _run_validation_v3(
    epoch: int,
    ema_model: torch.nn.Module,
    vae: VAEWrapper,
    sampler: DDIMSampler,
    test_loader,
    device: torch.device,
    save_dir: str,
    calculator: MetricsCalculator,
    num_steps: int,
    writer: SummaryWriter,
    max_batches: int = 50,
    num_samples: int = 6,
    fid_max_samples: int = 200,
    fid_min_samples: int = 50,
    prediction_type: str = "v",
    cfg_scale: float = 1.0,
    is_test: bool = False,
):
    ema_model.eval()
    vae.eval()

    print(f"[{ 'Testing' if is_test else 'Validation' }] Starting run at epoch {epoch}")

    metrics = {"ssim_B": [], "psnr_B": []}
    real_B_list = []
    fake_B_list = []

    os.makedirs(save_dir, exist_ok=True)
    prefix = "Testing" if is_test else "Validation"

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            if i == 0:
                print(f"[{prefix}] Running first batch...")
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            uncond = torch.zeros_like(real_A)
            z0 = sampler.sample(
                ema_model,
                real_A,
                shape=(real_A.size(0), 4, 32, 32),
                device=device,
                num_steps=num_steps,
                eta=0.0,
                prediction_type=prediction_type,
                cfg_scale=cfg_scale,
                uncond_condition=uncond,
            )
            fake_B = vae.decode(z0)

            metrics["ssim_B"].append(calculator.calculate_ssim(real_B, fake_B))
            metrics["psnr_B"].append(calculator.calculate_psnr(real_B, fake_B))

            real_B_list.append(real_B)
            fake_B_list.append(fake_B)

            if i < num_samples:
                row = torch.cat([real_A[:1], fake_B[:1], real_B[:1]], dim=0).cpu()
                out_path = os.path.join(save_dir, f"image_{i + 1}_A.png")
                save_images_with_title(
                    row,
                    labels=["Real A", "Fake B", "Real B"],
                    out_path=out_path,
                    value_range=(-1, 1),
                )
            if (i + 1) % 10 == 0:
                print(f"[{prefix}] Processed {i + 1} batches")

    avg_metrics = {
        "ssim_B": float(sum(metrics["ssim_B"]) / max(1, len(metrics["ssim_B"]))),
        "psnr_B": float(sum(metrics["psnr_B"]) / max(1, len(metrics["psnr_B"]))),
    }

    fid_count = min(fid_max_samples, len(real_B_list))
    if fid_count >= fid_min_samples:
        real_B_tensor = torch.cat(real_B_list[:fid_count])
        fake_B_tensor = torch.cat(fake_B_list[:fid_count])
        avg_metrics["fid"] = calculator.evaluate_fid(real_B_tensor, fake_B_tensor)

    for metric_name, value in avg_metrics.items():
        writer.add_scalar(f"{prefix}/{metric_name}", value, epoch)

    print(
        f"{prefix} Metrics - SSIM_B: {avg_metrics['ssim_B']:.4f}, "
        f"PSNR_B: {avg_metrics['psnr_B']:.2f}"
    )
    if "fid" in avg_metrics:
        print(f"{prefix} FID Score: {avg_metrics['fid']:.2f}")
    print(f"[{prefix}] Completed run at epoch {epoch}")

    # ema_model.train()
    return avg_metrics


def train_v3(
    epoch_size=None,
    num_epochs=None,
    model_dir=None,
    val_dir=None,
    test_size=None,
    cfg: Optional[UVCGANConfig] = None,
):
    """
    Train v3 latent diffusion model (DiT + condition tokenizer + frozen VAE).

    Args:
        epoch_size, num_epochs, model_dir, val_dir, test_size: optional
            overrides for corresponding config fields.
        cfg: full ``UVCGANConfig`` configured for model_version=3.

    Returns:
        tuple: ``(history, dit_model, ema_model, cond_tokenizer)``.
    """
    if cfg is None:
        cfg = get_dit_config()

    if epoch_size is not None:
        cfg.training.epoch_size = epoch_size
    if num_epochs is not None:
        cfg.training.num_epochs = num_epochs
    if model_dir is not None:
        cfg.model_dir = model_dir
    if val_dir is not None:
        cfg.val_dir = val_dir
    if test_size is not None:
        cfg.training.test_size = test_size

    tcfg = cfg.training
    dtcfg = cfg.data
    dcfg = cfg.diffusion
    lcfg = cfg.loss

    assert tcfg.num_epochs is not None, "num_epochs must be specified"
    assert tcfg.epoch_size is not None, "epoch_size must be specified"
    assert tcfg.validation_size is not None, "validation_size must be specified"
    assert (
        tcfg.validation_warmup_epochs < tcfg.early_stopping_warmup
    ), "validation_warmup_epochs must be less than early_stopping_warmup"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- Data ----
    train_loader, test_loader = getDataLoader(
        epoch_size=tcfg.epoch_size,
        image_size=dtcfg.image_size,
        batch_size=dtcfg.batch_size,
        num_workers=dtcfg.num_workers,
    )

    # ---- Models ----
    vae = VAEWrapper(dcfg.vae_model_id).to(device)
    vae.eval()

    dit_model = getGeneratorV3(dcfg, device=device)
    ema_model = copy.deepcopy(dit_model).to(device)
    ema_model.requires_grad_(False)

    scheduler = DDPMScheduler(dcfg.num_timesteps, dcfg.beta_schedule).to(device)
    sampler = DDIMSampler(scheduler)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        list(dit_model.parameters()),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    lr_min_ratio = 1e-6 / 1e-4
    lr_lambda = _make_cosine_warmup_lambda(
        warmup=tcfg.warmup_epochs,
        total=tcfg.num_epochs,
        lr_min_ratio=lr_min_ratio,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    use_amp = tcfg.use_amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # ---- Metrics / early stopping ----
    metrics_calculator = MetricsCalculator(device=device)
    early_stopping = EarlyStopping(
        patience=max(
            1,
            math.ceil(tcfg.early_stopping_patience / tcfg.early_stopping_interval),
        ),
        min_delta=1e-5,
        divergence_threshold=tcfg.divergence_threshold,
        divergence_patience=tcfg.divergence_patience,
    )

    # ---- Perceptual loss ----
    perceptual_loss = None
    if dcfg.lambda_perceptual_v3 > 0.0:
        perceptual_resize = (
            lcfg.perceptual_resize if lcfg.perceptual_resize is not None else 128
        )
        perceptual_loss = VGGPerceptualLossV2(resize_to=perceptual_resize).to(device)

    # ---- Output dirs / TensorBoard ----
    model_dir = cfg.model_dir or os.path.join(
        "data", "E_Staining_DermaRepo", "H_E-Staining_dataset", "models_v3"
    )
    val_dir = cfg.val_dir or os.path.join(model_dir, "validation_images")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    tb_dir = os.path.join(model_dir, "tensorboard_logs")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    history_csv_path = os.path.join(model_dir, "training_history.csv")
    if os.path.exists(history_csv_path):
        os.remove(history_csv_path)

    history = {}
    stopped_epoch = tcfg.num_epochs
    accumulate = max(1, tcfg.accumulate_grads)
    accum_count = 0

    if accumulate > 1:
        print(
            f"[train_v3] Gradient accumulation enabled: accumulate_grads={accumulate}, "
            f"batch_size={dtcfg.batch_size} -> effective batch = {accumulate * dtcfg.batch_size}"
        )
    if dcfg.use_gradient_checkpointing:
        print(
            "[train_v3] Gradient checkpointing enabled: ~30-40% less activation VRAM."
        )

    for epoch in range(tcfg.num_epochs):
        print()
        dit_model.train()

        epoch_step = {}
        epoch_loss = 0.0
        epoch_loss_perc = 0.0
        epoch_grad_norm = 0.0
        t_epoch_sum = 0.0
        t_epoch_sq_sum = 0.0
        t_epoch_count = 0
        accum_count = 0

        writer.add_scalar("Epoch", epoch + 1, epoch + 1)

        for i, batch in enumerate(train_loader, start=1):
            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

            if not (real_A.isfinite().all() and real_B.isfinite().all()):
                print(f"[warn] non-finite input at epoch {epoch+1} batch {i}, skipping")
                continue

            with torch.no_grad():
                z0_A = vae.encode(real_A)
                z0_B = vae.encode(real_B)

            t_A = torch.randint(0, dcfg.num_timesteps, (real_A.size(0),), device=device)
            t_B = torch.randint(0, dcfg.num_timesteps, (real_B.size(0),), device=device)
            noise_A = torch.randn_like(z0_A)
            noise_B = torch.randn_like(z0_B)
            z_t_A = scheduler.add_noise(z0_A, noise_A, t_A)
            z_t_B = scheduler.add_noise(z0_B, noise_B, t_B)

            t_epoch_sum += float(t_A.float().sum().item() + t_B.float().sum().item())
            t_epoch_sq_sum += float(
                (t_A.float().pow(2).sum().item() + t_B.float().pow(2).sum().item())
            )
            t_epoch_count += int(t_A.numel() + t_B.numel())

            if accum_count == 0:
                optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                out_A2B = dit_model(
                    z_t_A,
                    t_A,
                    real_A,
                    target_domain=1,
                    scheduler=scheduler,
                    prediction_type=dcfg.prediction_type,
                )
                out_B2A = dit_model(
                    z_t_B,
                    t_B,
                    real_B,
                    target_domain=0,
                    scheduler=scheduler,
                    prediction_type=dcfg.prediction_type,
                )

            loss_A2B, loss_simple_A2B, loss_perc_A2B = compute_diffusion_loss(
                z0=z0_A,
                z_t=z_t_A,
                t=t_A,
                noise=noise_A,
                model_pred=out_A2B["v_pred"],
                real_B=real_A,
                scheduler=scheduler,
                vae=vae,
                perceptual_loss=perceptual_loss,
                lambda_perc=dcfg.lambda_perceptual_v3,
                prediction_type=dcfg.prediction_type,
                min_snr_gamma=dcfg.min_snr_gamma,
                global_step=(epoch * len(train_loader) + i),
                perceptual_every_n_steps=dcfg.perceptual_every_n_steps,
                perceptual_batch_fraction=dcfg.perceptual_batch_fraction,
            )
            loss_B2A, loss_simple_B2A, loss_perc_B2A = compute_diffusion_loss(
                z0=z0_B,
                z_t=z_t_B,
                t=t_B,
                noise=noise_B,
                model_pred=out_B2A["v_pred"],
                real_B=real_B,
                scheduler=scheduler,
                vae=vae,
                perceptual_loss=perceptual_loss,
                lambda_perc=dcfg.lambda_perceptual_v3,
                prediction_type=dcfg.prediction_type,
                min_snr_gamma=dcfg.min_snr_gamma,
                global_step=(epoch * len(train_loader) + i),
                perceptual_every_n_steps=dcfg.perceptual_every_n_steps,
                perceptual_batch_fraction=dcfg.perceptual_batch_fraction,
            )

            loss = 0.5 * (loss_A2B + loss_B2A)
            loss_simple = 0.5 * (loss_simple_A2B + loss_simple_B2A)
            loss_perc_val = 0.5 * (loss_perc_A2B + loss_perc_B2A)

            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at epoch {epoch+1} batch {i}, skipping")
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                continue

            loss_scaled = loss / accumulate
            scaler.scale(loss_scaled).backward()
            accum_count += 1

            if accum_count == accumulate or (
                i == len(train_loader) and accum_count > 0
            ):
                grad_clip = (
                    dcfg.grad_clip_norm_g
                    if getattr(dcfg, "grad_clip_norm_g", None) is not None
                    else tcfg.grad_clip_norm
                )
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(dit_model.parameters()),
                        grad_clip,
                    )
                grad_norm = _global_grad_norm(list(dit_model.parameters()))
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), dit_model.parameters()):
                        ema_p.data.mul_(0.9999).add_(p.data, alpha=1 - 0.9999)

                accum_count = 0
            else:
                grad_norm = 0.0

            epoch_step[i] = {
                "Batch": i,
                "Loss_DiT_A2B": float(loss_simple_A2B.item()),
                "Loss_DiT_B2A": float(loss_simple_B2A.item()),
                "Loss_DiT": float(loss_simple.item()),
                "Loss_Perceptual": float(loss_perc_val),
                "Loss Total": float(loss.item()),
                "GradNorm": float(grad_norm),
            }
            epoch_loss += float(loss_simple.item())
            epoch_loss_perc += float(loss_perc_val)
            epoch_grad_norm += float(grad_norm)

            if i == 1 or i == len(train_loader) or i % 50 == 0:
                if dcfg.lambda_perceptual_v3 > 0.0:
                    print(
                        f"Epoch [{epoch + 1}/{tcfg.num_epochs}] "
                        f"Batch [{i}/{len(train_loader)}] "
                        f"Loss A2B: {loss_simple_A2B.item():.4f} "
                        f"Loss B2A: {loss_simple_B2A.item():.4f} "
                        f"Loss DiT: {loss_simple.item():.4f} "
                        f"Loss Perceptual: {loss_perc_val:.4f} "
                        f"Loss Total: {loss.item():.4f} "
                        f"GradNorm: {grad_norm:.4f}"
                    )
                else:
                    print(
                        f"Epoch [{epoch + 1}/{tcfg.num_epochs}] "
                        f"Batch [{i}/{len(train_loader)}] "
                        f"Loss A2B: {loss_simple_A2B.item():.4f} "
                        f"Loss B2A: {loss_simple_B2A.item():.4f} "
                        f"Loss DiT: {loss_simple.item():.4f} "
                        f"Loss Total: {loss.item():.4f} "
                        f"GradNorm: {grad_norm:.4f}"
                    )

        n_batches = max(1, len(train_loader))
        avg_loss = epoch_loss / n_batches
        avg_loss_perc = epoch_loss_perc / n_batches
        avg_grad_norm = epoch_grad_norm / n_batches

        history[epoch + 1] = epoch_step
        writer.add_scalar("Loss/DiT", avg_loss, epoch + 1)
        writer.add_scalar("Loss/Perceptual", avg_loss_perc, epoch + 1)
        writer.add_scalar("Diagnostics/GradNorm", avg_grad_norm, epoch + 1)
        if t_epoch_count > 0:
            t_mean = t_epoch_sum / t_epoch_count
            t_var = max(0.0, (t_epoch_sq_sum / t_epoch_count) - (t_mean * t_mean))
            t_std = math.sqrt(t_var)
            writer.add_scalar("Diagnostics/TimestepMean", t_mean, epoch + 1)
            writer.add_scalar("Diagnostics/TimestepStd", t_std, epoch + 1)

        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        writer.add_scalar("LR/DiT", current_lr, epoch + 1)

        print(
            f"Epoch [{epoch + 1}/{tcfg.num_epochs}] "
            f"Avg Loss: {avg_loss:.4f}  "
            f"LR: {current_lr:.6f}"
        )

        if (epoch + 1) % 5 == 0:
            append_history_to_csv_v3(history, history_csv_path)
            history.clear()

        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "dit_state_dict": dit_model.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": dcfg,
                },
                ckpt_path,
            )
            print(f"Checkpoint saved: {ckpt_path}")

        val_metrics = None
        if (epoch + 1) > tcfg.validation_warmup_epochs:
            save_dir = os.path.join(val_dir, f"epoch_{epoch + 1}")
            val_metrics = _run_validation_v3(
                epoch=epoch + 1,
                ema_model=ema_model,
                vae=vae,
                sampler=sampler,
                test_loader=test_loader,
                device=device,
                save_dir=save_dir,
                calculator=metrics_calculator,
                num_steps=dcfg.num_inference_steps,
                writer=writer,
                is_test=False,
                max_batches=tcfg.validation_size,
                num_samples=max(6, tcfg.validation_size),
                fid_max_samples=tcfg.validation_fid_samples,
                fid_min_samples=tcfg.validation_fid_min_samples,
                prediction_type=dcfg.prediction_type,
                cfg_scale=dcfg.cfg_scale,
            )

        if (
            (epoch + 1) % tcfg.early_stopping_interval == 0
            and epoch + 1 >= tcfg.early_stopping_warmup
            and val_metrics is not None
        ):
            avg_ssim = val_metrics.get("ssim_B", 0.0)
            should_stop = early_stopping(ssim=avg_ssim, losses={"DiT": avg_loss})
            writer.add_scalar("EarlyStopping/ssim", avg_ssim, epoch + 1)
            writer.add_scalar(
                "EarlyStopping/counter", early_stopping.counter, epoch + 1
            )
            writer.add_scalar(
                "EarlyStopping/divergence_counter",
                early_stopping.divergence_counter,
                epoch + 1,
            )

            if should_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                stopped_epoch = epoch + 1
                break

    final_ckpt = os.path.join(model_dir, f"final_checkpoint_epoch_{stopped_epoch}.pth")
    torch.save(
        {
            "epoch": stopped_epoch,
            "dit_state_dict": dit_model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dcfg,
        },
        final_ckpt,
    )
    print(f"Final checkpoint saved: {final_ckpt}")

    # ---- Final test-set inference ----
    test_dir = os.path.join(model_dir, "test_images")
    writer.add_scalar("Testing Started", stopped_epoch, stopped_epoch)
    _run_validation_v3(
        epoch=stopped_epoch,
        ema_model=ema_model,
        vae=vae,
        sampler=sampler,
        test_loader=test_loader,
        device=device,
        save_dir=test_dir,
        calculator=metrics_calculator,
        num_steps=dcfg.num_inference_steps,
        writer=writer,
        max_batches=tcfg.test_size,
        num_samples=max(6, tcfg.test_size),
        fid_max_samples=tcfg.validation_fid_samples,
        fid_min_samples=tcfg.validation_fid_min_samples,
        prediction_type=dcfg.prediction_type,
        cfg_scale=dcfg.cfg_scale,
        is_test=True,
    )
    writer.add_scalar("Training Completed", stopped_epoch, stopped_epoch)

    append_history_to_csv_v3(history, history_csv_path)
    history = load_history_from_csv_v3(history_csv_path)

    writer.close()
    return history, dit_model, ema_model, None
