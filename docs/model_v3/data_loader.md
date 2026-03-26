# `model_v3/data_loader.py` — Paired Data Loader

Source of truth: `../../model_v3/data_loader.py`

**Role:** Builds paired A/B dataloaders for the v3 diffusion pipeline by matching filenames across `trainA/trainB` and `testA/testB`.

## Notes

- Uses a `PairedImageDataset` that aligns A/B samples by filename.
- `strict_pairs=True` enforces intersection-only pairing (recommended for aligned datasets).
- Output batches provide `{ "A": ..., "B": ... }` to match the v3 training loop expectations.
