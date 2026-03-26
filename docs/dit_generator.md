# `dit_generator.py` — Diffusion Transformer (DiT)

Source of truth: `../dit_generator.py`

**Role:** Implements the DiT backbone and conditioning encoder for v3 diffusion.

---

## Components

### `PatchEmbed`
Splits latent `(N, 4, 32, 32)` into non-overlapping `2x2` patches and projects
each patch into `hidden_dim` tokens.

### `TimestepEmbedding`
Sinusoidal timestep embedding followed by a 2-layer MLP.

### `ConditionEncoder`
Shallow CNN that maps `real_A` `(N, 3, 256, 256)` to a `(N, hidden_dim)` vector.

### `DiTBlock`
Transformer block with adaLN-Zero conditioning:

- LayerNorm without affine
- Self-attention
- MLP
- Conditioning MLP produces `(gamma, beta, alpha)` for attention and MLP
- Gates (`alpha`) are zero-initialized

---

## `DiTGenerator` (Locked Interface)

```python
def forward(self, z_t, t, c):
    """
    z_t: (N, 4, 32, 32)
    t:   (N,)
    c:   (N, hidden_dim)  # precomputed condition
    """
```

`ConditionEncoder` is not inside `DiTGenerator` and must be called externally.

---

## Factory

`getGeneratorV3(cfg)` builds a `DiTGenerator` from `DiffusionConfig`,
applies v2 weight init, and performs a smoke test.
