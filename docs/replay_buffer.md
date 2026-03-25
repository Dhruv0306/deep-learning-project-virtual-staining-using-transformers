# `replay_buffer.py` — Replay Buffer

Source of truth: `../replay_buffer.py`

**Shared by:** Both v1 and v2  
**Role:** Stores a pool of previously generated fake images and returns a mix of old and new fakes to the discriminator during training. This technique reduces model oscillation and improves discriminator stability.

---

## Motivation

Without a replay buffer, the discriminator sees only the most recently generated fakes at each step. The discriminator can over-fit to the current generator output and oscillate as the generator and discriminator chase each other. Returning buffered (older) fakes forces the discriminator to remain robust against past generator states, damping oscillations.

---

## Class: `ReplayBuffer`

A fixed-size FIFO-like pool with probabilistic replacement.

### `__init__(max_size=50)`

| Parameter | Default | Description |
|---|---|---|
| `max_size` | 50 | Maximum number of single-image tensors stored in the buffer |

---

### `push_and_pop(tensors)`

Adds the new batch to the buffer and returns a batch with the same size that mixes old and new samples.

| Parameter | Type | Description |
|---|---|---|
| `tensors` | `torch.Tensor` | Batch of newly generated images `(N, C, H, W)`. Must be detached before calling. |

**Per-sample logic:**
```
for each image in tensors:
    if buffer is not full (< max_size):
        store image and return it as-is

    else with 50% probability:
        pick a random slot in the buffer
        return the stored image from that slot
        replace that slot with the new image

    else (other 50%):
        return the new image as-is (but do NOT store it)
```

**Returns:** `torch.Tensor` — a new batch `(N, C, H, W)` assembled from old/new samples. The returned batch should be detached (`.detach()` is applied to each element before storing, so the buffer never holds gradients).

---

## Usage Example

```python
buffer = ReplayBuffer(max_size=50)

# Inside the training loop, when updating discriminator D_A:
fake_A_for_D = buffer.push_and_pop(fake_A)  # Returns mix of old + new fakes
pred_fake = D_A(fake_A_for_D.detach())
```

The buffer operates independently for each domain — `CycleGANLoss` and `UVCGANLoss` each maintain separate `fake_A_buffer` and `fake_B_buffer` instances.
