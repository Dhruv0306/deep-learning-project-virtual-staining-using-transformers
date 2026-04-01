# `shared/testing.py` — Testing Helper

**Shared by:** Both v1 and v2  
**Role:** Runs inference on the full test set, computes summary cycle and identity losses, saves comparison images to disk, and optionally logs results to TensorBoard. Called at the end of training and available for standalone evaluation.

---

## Function: `run_testing(G_AB, G_BA, test_loader, device, save_dir, writer, epoch, num_samples)`

| Parameter | Default | Description |
|---|---|---|
| `G_AB`, `G_BA` | — | Generator models (set to eval mode internally, restored to train mode on exit) |
| `test_loader` | — | Test `DataLoader` |
| `device` | — | Target device |
| `save_dir` | — | Directory to write comparison PNG images |
| `writer` | `None` | Optional TensorBoard `SummaryWriter` |
| `epoch` | `None` | Epoch index used for TensorBoard step and image filenames. Passed as `"test"` string to image filenames when `None` |
| `num_samples` | `None` | Maximum number of test samples to process. Uses the full test loader when `None` |

---

## What It Computes

For each test batch the function computes the full cycle path:

```
real_A ──► G_AB ──► fake_B ──► G_BA ──► rec_A
real_B ──► G_BA ──► fake_A ──► G_AB ──► rec_B

real_A ──► G_BA ──► idt_A     (identity mapping)
real_B ──► G_AB ──► idt_B
```

**Logged losses (L1):**

| Loss | Formula |
|---|---|
| Cycle A | `L1(rec_A, real_A)` |
| Cycle B | `L1(rec_B, real_B)` |
| Identity A | `L1(idt_A, real_A)` |
| Identity B | `L1(idt_B, real_B)` |

The sum of cycle and identity losses is accumulated across batches and the average over `num_samples` is printed and (optionally) logged to TensorBoard under:
- `Testing/Average Cycle Loss`
- `Testing/Average Identity Loss`

---

## Output Images

For each sample, calls `save_images(is_test=True)` from `shared/validation.py`, which produces two 4-panel PNG files per sample:

| File | Content |
|---|---|
| `image_{i}_A.png` | `[Real A | Fake B | Rec A | Real B]` |
| `image_{i}_B.png` | `[Real B | Fake A | Rec B | Real A]` |

---

## Typical Usage (end of training)

```python
run_testing(
    G_AB=G_AB,
    G_BA=G_BA,
    test_loader=test_loader,
    device=device,
    save_dir=os.path.join(model_dir, "test_images"),
    writer=writer,
    epoch=final_epoch,
    num_samples=cfg.training.test_size,
)
```

