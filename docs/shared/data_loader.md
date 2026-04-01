# `shared/data_loader.py` — Data Loading

**Shared by:** Both v1 and v2  
**Role:** Provides the dataset class and data-loader factory used in all training and testing loops. Handles unpaired sampling, image transformations, and normalization.

---

## Dataset Layout

The loader expects a CycleGAN-style four-folder structure, produced by `preprocess_data.py`:

```
data/E_Staining_DermaRepo/H_E-Staining_dataset/
    trainA/   ← unstained (domain A) training patches
    trainB/   ← stained   (domain B) training patches
    testA/    ← unstained test patches
    testB/    ← stained   test patches
```

---

## Transform Pipeline

```
PIL RGB image
    │
    ▼ Resize(image_size × image_size, BILINEAR)
    ▼ ToTensor()                  → float32 tensor in [0, 1]
    ▼ Normalize(mean=0.5, std=0.5) → float32 tensor in [-1, 1]
```

The `[−1, 1]` range matches the `Tanh` output of both generators, so no additional rescaling is needed inside the model.

---

## Classes

### `UnpairedImageDataset`

A `torch.utils.data.Dataset` that returns one image from each domain per call. Domain A is sampled **sequentially** (with modulo wrap-around); domain B is sampled **randomly** from the full domain B list. This avoids creating fixed A/B pairs, which would bias the unpaired translation task.

| Constructor Parameter | Type | Default | Description |
|---|---|---|---|
| `dir_A` | `str` | — | Path to domain A image directory |
| `dir_B` | `str` | — | Path to domain B image directory |
| `transform` | `callable` | `None` | Transform pipeline applied to every loaded image |
| `epoch_size` | `int` | `None` | If set, overrides `__len__` to control the number of batches per epoch. If `None`, length equals `max(len(A), len(B))` so the shorter domain doesn't truncate training |

**`__getitem__(idx)`**

| Step | Detail |
|---|---|
| Domain A | `images_A[idx % len(images_A)]` — sequential, wraps around |
| Domain B | `random.choice(images_B)` — random each call |
| Returns | `{'A': image_tensor_A, 'B': image_tensor_B}` |

Images are loaded with `PIL.Image.open(...).convert("RGB")` to guarantee 3-channel tensors even if the source files are grayscale.

---

## Functions

### `denormalize(t)`

Reverses the `Normalize(mean=0.5, std=0.5)` transform applied by the dataloader.

```
t ([-1, 1]) → t * 0.5 + 0.5 → [0, 1]
```

The output is clamped to `[0, 1]` to handle any minor floating-point overshoot. Used for visualization before `plt.imshow()` or saving images.

| Parameter | Type | Description |
|---|---|---|
| `t` | `torch.Tensor` | Tensor in `[-1, 1]` range |

Returns a `torch.Tensor` in `[0, 1]`.

---

### `getDataLoader(epoch_size, image_size, batch_size, num_workers)`

Factory that builds and returns `(train_loader, test_loader)`.

| Parameter | Default | Description |
|---|---|---|
| `epoch_size` | `None` | Passed to `UnpairedImageDataset` for training. Controls steps per epoch |
| `image_size` | 256 | Spatial size to which every patch is resized |
| `batch_size` | 4 | Training batch size |
| `num_workers` | 4 | DataLoader worker processes |

**Training loader settings:**
- `shuffle=True`
- `pin_memory=True` — copies tensors to pinned (page-locked) host memory for faster GPU transfer
- `persistent_workers=True` — keeps worker processes alive between epochs
- `prefetch_factor=2` — each worker pre-fetches 2 batches
- `drop_last=True` — drops the last batch if smaller than `batch_size` (prevents shape mismatches in batch-norm layers)

**Test loader settings:**
- `batch_size=1` — single-sample inference
- `shuffle=False` — deterministic order

The function prints GPU diagnostics and sanity-checks batch shapes on startup.

**Returns:** `(train_loader, test_loader)` — both are `torch.utils.data.DataLoader` instances.

---

## Data Flow Summary

```
trainA/  ──┐
           ├── UnpairedImageDataset ──► DataLoader (batch_size=4, shuffled)
trainB/  ──┘

testA/   ──┐
           ├── UnpairedImageDataset ──► DataLoader (batch_size=1, ordered)
testB/   ──┘

Each batch: {'A': (N, 3, 256, 256), 'B': (N, 3, 256, 256)}  in [-1, 1]
```

