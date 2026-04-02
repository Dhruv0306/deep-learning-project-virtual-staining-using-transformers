# model_v4/nce_loss.py

Source of truth: ../../model_v4/nce_loss.py

Defines PatchNCE contrastive loss for v4 training.

## PatchNCELoss

PatchNCE uses an InfoNCE objective over sampled query/key patch features.
Each feature layer gets its own projection MLP.

## Constructor

- `PatchNCELoss(temperature=0.07, proj_dim=256)`

State:

- `projections`: `ModuleDict` of lazily-created per-layer projector MLPs

## Internal Methods

## _get_projector(key, in_dim, device, dtype)

- creates projector only when first needed for a layer key
- structure: `Linear(in_dim, proj_dim) -> ReLU -> Linear(proj_dim, proj_dim)`

## _layer_loss(feat_q, feat_k, key)

Inputs:

- `feat_q`, `feat_k`: `(B, N, C)`

Steps:

1. project q/k with layer-specific MLP
2. L2-normalize along channel dimension
3. flatten to `(B*N, proj_dim)`
4. compute pairwise logits via dot product / temperature
5. apply cross-entropy with identity labels (`i` matches `i`)

Output:

- scalar InfoNCE loss for one layer

## forward(feats_q, feats_k)

- iterates over layer pairs
- skips empty tensors
- averages per-layer losses
- returns zero tensor when no valid layers are present

Used in v4 training loop for both directions:

- A->B patches against A features
- B->A patches against B features
