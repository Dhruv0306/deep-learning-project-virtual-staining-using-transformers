# model_v3/data_loader.py - v3 Data Loader

Source of truth: ../../model_v3/data_loader.py

Role: Builds paired A/B datasets and DataLoader objects for v3 diffusion training.

---

## Component Structure

1. PairedImageDataset
2. getDataLoaderV3

---

## 1) PairedImageDataset

Purpose:
- returns filename-aligned A/B image pairs

Constructor inputs:
- dir_A, dir_B
- transform
- epoch_size
- strict_pairs

### Dataflow in initialization

Input directories:
- dir_A files: list of names
- dir_B files: list of names

Path A: strict_pairs=True
1. compute filename intersection
2. build pairs list [(name, name), ...]

Path B: strict_pairs=False
1. zip sorted lists to min length
2. build sequential pairs

Internal storage:
- self.pairs length = P

### __len__ dataflow

Input:
- epoch_size and P

Output:
- if epoch_size is set: length = epoch_size
- else: length = P

### __getitem__ dataflow

Input:
- idx scalar

Step 1:
- modulo index into self.pairs
- get (name_A, name_B)

Step 2:
- open image_A and image_B in RGB
- PIL shapes conceptually (H, W, 3)

Step 3:
- optional transform:
  - Resize -> ToTensor -> Normalize

Output dictionary:
- A: tensor (3, image_size, image_size)
- B: tensor (3, image_size, image_size)

---

## 2) getDataLoaderV3

Purpose:
- creates train and test dataloaders for paired diffusion training

Inputs:
- epoch_size
- image_size
- batch_size
- num_workers
- strict_pairs

### Transform dataflow

PIL image -> Resize(image_size,image_size) -> ToTensor -> Normalize(mean=0.5,std=0.5)

Shape/range transitions:
- PIL RGB (H,W,3)
- tensor (3,image_size,image_size) in [0,1]
- normalized tensor (3,image_size,image_size) in [-1,1]

### Train loader dataflow

Dataset item:
- A: (3,image_size,image_size)
- B: (3,image_size,image_size)

Collated train batch:
- A: (batch_size,3,image_size,image_size)
- B: (batch_size,3,image_size,image_size)

### Test loader dataflow

Dataset item same as above.

Collated test batch (batch_size fixed to 1):
- A: (1,3,image_size,image_size)
- B: (1,3,image_size,image_size)

Returns:
- (train_loader, test_loader)

---

## Runtime Checks Printed by Function

- torch version
- CUDA availability and GPU info
- sample train batch shapes
- sample test batch shapes

These are shape sanity checks and do not alter data.
