# `validation.py` — Validation Helpers

Source of truth: `../validation.py`

**Shared by:** Both v1 and v2  
**Role:** Provides functions for qualitative and quantitative validation during training. Saves side-by-side comparison images and computes per-epoch metric summaries logged to TensorBoard.

---

## Functions

### `calculate_metrics(calculator, G_AB, G_BA, test_loader, device, writer, epoch)`

Runs quantitative evaluation on up to 50 test batches and logs results to TensorBoard.

| Parameter | Type | Description |
|---|---|---|
| `calculator` | `MetricsCalculator` | Metric computation object from `metrics.py` |
| `G_AB`, `G_BA` | `nn.Module` | Generator models (set to eval mode internally) |
| `test_loader` | `DataLoader` | Test data loader |
| `device` | `torch.device` | Target device |
| `writer` | `SummaryWriter` | TensorBoard writer |
| `epoch` | `int` | Current epoch (used as the TensorBoard x-axis step) |

**Metrics computed:**

| Key | Description |
|---|---|
| `ssim_A` | Mean SSIM between `real_A` and `fake_A = G_BA(real_B)` |
| `ssim_B` | Mean SSIM between `real_B` and `fake_B = G_AB(real_A)` |
| `psnr_A` | Mean PSNR between `real_A` and `fake_A` |
| `psnr_B` | Mean PSNR between `real_B` and `fake_B` |
| `fid` | FID computed on the first 10 batches of `real_B` vs `fake_B` (optional, only if > 10 batches processed) |

All metrics are logged under the `Validation/` TensorBoard prefix. Generators are restored to training mode before returning.

**Returns:** `dict` of average metric values.

---

### `run_validation(epoch, G_AB, G_BA, test_loader, device, save_dir, num_samples, writer)`

Generates and saves visual comparison images for a small number of test samples.

| Parameter | Default | Description |
|---|---|---|
| `epoch` | — | Epoch number, used in filenames and TensorBoard |
| `G_AB`, `G_BA` | — | Generator models |
| `test_loader` | — | Test data loader |
| `device` | — | Target device |
| `save_dir` | — | Directory where image files are written |
| `num_samples` | 3 | Number of test samples to visualise |
| `writer` | `None` | Optional TensorBoard writer for cycle/identity losses |

For each sample, computes `fake_B`, `rec_A`, `fake_A`, `rec_B`, `idt_A`, `idt_B` and saves comparison image grids via `save_images`. Also computes and logs average cycle and identity losses for the validation batch.

**Returns:** `None`.

---

### `save_images(img_id, real_A, fake_B, rec_A, real_B, fake_A, rec_B, epoch, save_dir, is_test)`

Saves two 4-panel image rows (one per domain) as PNG files.

| Parameter | Default | Description |
|---|---|---|
| `img_id` | — | Integer index used to name the file |
| `real_A`, `fake_B`, `rec_A`, `real_B`, `fake_A`, `rec_B` | — | Image tensors `(1, C, H, W)`, only the first sample is used |
| `epoch` | — | Epoch or `"test"` string for filenames |
| `save_dir` | `None` | If provided, files are saved as `{save_dir}/image_{img_id}_A.png` etc. |
| `is_test` | `False` | Controls default path when `save_dir` is `None` |

**Domain A row:** `[Real A | Fake B | Rec A | Real B]` — shows the A→B→A cycle.  
**Domain B row:** `[Real B | Fake A | Rec B | Real A]` — shows the B→A→B cycle.

---

### `save_images_with_title(row_tensor, labels, out_path, value_range, header_h)`

Low-level helper. Creates a 1×4 image grid with a white text header strip above it.

| Parameter | Default | Description |
|---|---|---|
| `row_tensor` | — | `(4, C, H, W)` tensor of four images to tile |
| `labels` | — | List of 4 column header strings |
| `out_path` | — | Output file path |
| `value_range` | `(-1, 1)` | Min/max for `torchvision.utils.make_grid` normalisation |
| `header_h` | 36 | Height of the header strip in pixels |

Uses `PIL.ImageDraw` to draw text. Falls back to `ImageFont.load_default()` if `arial.ttf` is not found on the system.
