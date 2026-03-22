# `metrics.py` — Evaluation Metrics

**Shared by:** Both v1 and v2  
**Role:** Computes image quality and distribution alignment metrics used during validation and testing. Exposes SSIM, PSNR, and FID through a single `MetricsCalculator` class.

---

## Class: `MetricsCalculator`

Bundles all metric computation under one object. Initialises an InceptionV3 feature extractor on construction and reuses it for all FID calls.

### `__init__(device=None)`

| Parameter | Default | Description |
|---|---|---|
| `device` | `None` | Computation device. Defaults to CUDA if available |

Constructs an InceptionV3 model with the final classification layer replaced by `nn.Identity()`, producing 2048-dimensional feature embeddings. The model is set to `eval()` mode and its parameters are not updated.

**InceptionV3 preprocessing transform (applied in `get_inception_features`):**
- `Resize((299, 299))` — InceptionV3 requires 299×299 inputs
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` — ImageNet normalisation

---

### `calculate_ssim(img1, img2)`

Computes the **Structural Similarity Index Measure** between two images using `skimage.metrics.structural_similarity`.

| Parameter | Type | Description |
|---|---|---|
| `img1` | `torch.Tensor` or `np.ndarray` | First image (shape `(C, H, W)` or `(N, C, H, W)`) |
| `img2` | `torch.Tensor` or `np.ndarray` | Second image |

- Tensors are converted to NumPy and transposed to `(H, W, C)` before computing SSIM.
- `data_range=2.0` accounts for the `[-1, 1]` pixel value range.
- For batched inputs `(N, C, H, W)`, SSIM is computed per sample and the batch mean is returned.

**Returns:** `float` in `[-1, 1]` (higher = more similar; 1.0 = identical).

---

### `calculate_psnr(img1, img2)`

Computes the **Peak Signal-to-Noise Ratio** between two image tensors.

| Parameter | Type | Description |
|---|---|---|
| `img1` | `torch.Tensor` | First image |
| `img2` | `torch.Tensor` | Second image |

Formula (for `[-1, 1]` range, max pixel value = 2):
```
PSNR = 20 × log₁₀(2 / √MSE)
```

Returns `float('inf')` when `MSE == 0` (identical images). Otherwise returns a value in dB — higher is better. Typical values for good translations are 25–35 dB.

---

### `get_inception_features(images)`

Extracts 2048-dimensional feature vectors from InceptionV3.

| Parameter | Type | Description |
|---|---|---|
| `images` | `torch.Tensor` | Batch in `[-1, 1]` range, shape `(N, 3, H, W)` |

Internally:
1. Denormalises from `[-1, 1]` to `[0, 1]`.
2. Applies `inception_transform` (resize + ImageNet normalisation).
3. Passes through InceptionV3 (no classification head).

Returns `numpy.ndarray` of shape `(N, 2048)`.

---

### `calculate_fid(real_images, fake_images)`

Computes the **Fréchet Inception Distance** between two image batches.

| Parameter | Type | Description |
|---|---|---|
| `real_images` | `torch.Tensor` | Real image batch |
| `fake_images` | `torch.Tensor` | Generated image batch |

**Formula:**
```
FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))
```
where `(μ₁, Σ₁)` and `(μ₂, Σ₂)` are the mean and covariance of the InceptionV3 features for real and fake images respectively.

If `scipy.linalg.sqrtm` returns a complex matrix (numerical instability), a small diagonal regularisation `ε = 1e-6` is added before retrying.

**Returns:** `float` — lower is better. FID of 0 means the two distributions are identical. Values under 50 are generally considered good for GAN outputs.

---

### `evaluate_batch(real_A, real_B, fake_A, fake_B)`

Convenience wrapper. Computes SSIM and PSNR for both translation directions in one call.

| Returns key | Description |
|---|---|
| `ssim_A` | SSIM between `real_A` and `fake_A` |
| `ssim_B` | SSIM between `real_B` and `fake_B` |
| `psnr_A` | PSNR between `real_A` and `fake_A` |
| `psnr_B` | PSNR between `real_B` and `fake_B` |

---

### `evaluate_fid(real_images, fake_images)`

Thin wrapper around `calculate_fid`. Provided for API symmetry with `evaluate_batch`.

---

## Metric Reference

| Metric | Unit | Direction | Typical range |
|---|---|---|---|
| SSIM | — | higher = better | `[-1, 1]`; good results ≥ 0.6 |
| PSNR | dB | higher = better | 20–40 dB; ≥ 30 dB is very good |
| FID | — | lower = better | < 50 is good; < 20 is excellent |

SSIM and PSNR are computed between generated and real images of the **same domain** (e.g., `fake_A` vs `real_A`). FID is computed over the distribution of generated B images vs real B images (or any batch pair passed to `calculate_fid`).
