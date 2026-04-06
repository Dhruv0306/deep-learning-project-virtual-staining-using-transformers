# model_v3/vae_wrapper.py — VAE Wrapper

Source of truth: `../../model_v3/vae_wrapper.py`

Wraps Stable Diffusion's `AutoencoderKL` for v3 latent diffusion training
and inference.

---

## Public API

| Symbol | Type | Description |
|---|---|---|
| `VAEWrapper` | `nn.Module` | Frozen SD VAE with encode/decode helpers |

---

## `VAEWrapper`

```python
VAEWrapper(
    model_id: str = "stabilityai/sd-vae-ft-mse",
    cache_dir: Optional[str] = None,
    offline_first: bool = True,
)
```

Behaviour:
- Downloads and caches `AutoencoderKL` from HuggingFace on first use (~335 MB).
- Freezes all parameters (`requires_grad_(False)`) and keeps the module in eval mode.
- Applies the SD latent scaling convention: multiply by `0.18215` on encode,
  divide by `0.18215` on decode.

### Constructor args

| Arg | Default | Description |
|---|---|---|
| `model_id` | `"stabilityai/sd-vae-ft-mse"` | HuggingFace model ID or local directory path |
| `cache_dir` | `None` | Optional HuggingFace cache directory override |
| `offline_first` | `True` | Try local cache before downloading |

### Loading strategy (`_load_vae`)

1. If `model_id` is an existing local directory → load from disk only.
2. If `offline_first=True` → try `local_files_only=True` first.
3. Fall back to online download if not found in cache.
4. Raises `RuntimeError` with a descriptive message if both cache and download fail.

---

## `encode`

```python
encode(x: Tensor) -> Tensor
```

| | Shape | Range |
|---|---|---|
| Input `x` | `(N, 3, H, W)` | `[-1, 1]` |
| Output | `(N, 4, H/8, W/8)` | scaled latents |

Pipeline: clamp → VAE encode → sample from latent distribution → multiply by `0.18215`.

---

## `decode`

```python
decode(z: Tensor) -> Tensor
```

| | Shape | Range |
|---|---|---|
| Input `z` | `(N, 4, h, w)` | scaled latents |
| Output | `(N, 3, h*8, w*8)` | `[-1, 1]` |

Pipeline: divide by `0.18215` → cast to float32 → VAE decode → clamp to `[-1, 1]`.

---

## Notes

- The wrapper is always frozen and in eval mode; it does not participate in
  optimiser updates.
- Gradients can still flow through `encode`/`decode` if the caller does not
  use `torch.no_grad()` (e.g. for perceptual losses on decoded outputs).
- A clear `ImportError` is raised at import time when `diffusers` is not
  installed. Install with `pip install diffusers>=0.27.0`.
