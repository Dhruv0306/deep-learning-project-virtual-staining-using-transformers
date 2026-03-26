# preprocess_data.py Data Preparation

Source of truth: ../preprocess_data.py

This script converts whole-slide source images into patch datasets used by training.

## Pipeline Summary

1. Read source images from:
   - Un_Stained
   - C_Stained
2. Split file names per domain into train/test (default 80/20).
3. Extract 256x256 patches with non-overlapping stride by default.
4. Estimate tissue fraction per patch.
5. Keep all tissue patches and randomly keep a fraction of background patches.
6. Save accepted patches as PNG into:
   - trainA, testA (unstained domain)
   - trainB, testB (stained domain)

## Key Functions

- extract_patches_pil(img, patch_size=256, stride=256)
  - slides a crop window over a PIL image and returns full-size tiles

- estimate_tissue_fraction(patch, white_thresh=220, sat_thresh=0.05)
  - classifies low-information background pixels using white + low-saturation heuristics
  - returns tissue fraction in [0, 1]

- split_filenames(file_list, train_ratio=0.8, seed=42)
  - deterministic split for a fixed seed (list is sorted before shuffle)

- save_patches(...)
  - applies filtering and writes accepted patches as RGB PNG

## Default Filtering In main()

- tissue_threshold = 0.1
- background_keep_ratio = 0.17
- white_thresh = 225
- sat_thresh = 0.05
- random seed = 42

## Notes

- PIL decompression warnings for very large slides are disabled intentionally.
- A sample image is plotted for visual sanity check; this does not change outputs.
- Normalization is not applied here; it is handled later in dataloader transforms.
