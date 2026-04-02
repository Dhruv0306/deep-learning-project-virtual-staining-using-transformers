# Model V2 Dataflow Pipeline

Model v2 is the true UVCGAN implementation with improved generator/discriminator structure, configurable training, gradient accumulation, and robust validation/early stopping.

## End-to-End Pipeline Diagram

```mermaid
flowchart TD
    A[real_A: unstained] --> GAB[G_AB ViT-UNet v2]
    B[real_B: stained] --> GBA[G_BA ViT-UNet v2]

    GAB --> FB[fake_B]
    GBA --> FA[fake_A]

    FB --> GBA_REC[G_BA]
    FA --> GAB_REC[G_AB]

    GBA_REC --> RA[rec_A]
    GAB_REC --> RB[rec_B]

    A --> IDA_PATH[G_BA]
    B --> IDB_PATH[G_AB]
    IDA_PATH --> IDA[idt_A]
    IDB_PATH --> IDB[idt_B]

    A --> DA[D_A multi-scale]
    FA --> DA
    B --> DB[D_B multi-scale]
    FB --> DB

    DA --> LD[LSGAN + GP discriminator losses]
    DB --> LD
    FB --> LG[Generator total loss]
    FA --> LG
    RA --> LG
    RB --> LG
    IDA --> LG
    IDB --> LG

    LG --> ACC[Grad accumulation + AMP]
    ACC --> OPTG[Update generators]
    LD --> OPTD[Update discriminators]

    OPTG --> VAL[Validation metrics + early stopping]
    OPTD --> VAL
```

## Training Dataflow

1. Read unpaired `real_A` and `real_B` mini-batches.
2. Discriminator stage:
   - freeze generators
   - generate `fake_A/fake_B` under `no_grad`
   - compute `loss_D_A` and `loss_D_B`
   - update discriminators
3. Generator stage:
   - freeze discriminators
   - compute forward translation, cycle, and identity outputs
   - compute generator composite loss from configured weights
   - apply gradient accumulation, clipping, and AMP-scaled update
4. Log per-batch and per-epoch metrics (losses, grad norm, LR).
5. Run periodic validation and metrics computation (SSIM/PSNR/FID), then apply early stopping checks.
6. Save checkpoints and final testing artifacts.

## Files in This Model

- [generator.md](generator.md) - v2 generator internals and architectural changes
- [discriminator.md](discriminator.md) - v2 multi-scale discriminator details
- [losses.md](losses.md) - v2 objective terms and weighting
- [training_loop.md](training_loop.md) - full training orchestration
