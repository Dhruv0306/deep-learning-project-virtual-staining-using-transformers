# model_v3/history_utils.py - v3 History Utilities

Source of truth: ../../model_v3/history_utils.py

Role: Save, append, reload, and visualize Phase 2 v3 training history.

---

## Component Structure

1. save_history_to_csv_v3
2. append_history_to_csv_v3
3. load_history_from_csv_v3
4. visualize_history_v3
5. _flatten_history

---

## Shared Data Structure

In-memory history:
- history[epoch][batch] -> dict with keys:
  - Batch
   - Loss_DiT_A2B
   - Loss_DiT_B2A
   - Loss_DiT
   - Loss_G_Adv
   - Loss_Cyc
   - Loss_Id
   - Loss_D_A
   - Loss_D_B
   - Lambda_Adv
   - Lambda_Id
  - Loss_Perceptual
   - Loss Total
  - GradNorm

Flattened row schema:
- Epoch, Batch, Loss_DiT_A2B, Loss_DiT_B2A, Loss_DiT, Loss_G_Adv, Loss_Cyc, Loss_Id,
   Loss_D_A, Loss_D_B, Lambda_Adv, Lambda_Id, Loss_Perceptual, Loss Total, GradNorm

---

## 1) save_history_to_csv_v3

Input:
- history dict
- filename

Dataflow:
1. _flatten_history(history) -> list of row dicts length M
2. pandas DataFrame from rows -> shape (M,15)
3. write CSV

Output:
- CSV file with Phase 2 columns

---

## 2) append_history_to_csv_v3

Input:
- history chunk dict
- filename

Dataflow:
1. if history empty -> return
2. flatten to rows length K
3. DataFrame shape (K,15)
4. append mode write, with header only when file missing/empty

Output:
- CSV appended with K rows

---

## 3) load_history_from_csv_v3

Input:
- filename

Dataflow:
1. read CSV to DataFrame shape (R,15)
2. iterate each row
3. rebuild nested structure history[epoch][batch] with all Phase 2 fields

Output:
- reconstructed history dict

---

## 4) visualize_history_v3

Input:
- history dict
- optional model_dir

Dataflow:
1. collect sorted epoch list, length E
2. per epoch, aggregate batch values for all key losses and grad norm
3. compute per-epoch means for:
   - denoising A2B/B2A/combined
   - generator adversarial, cycle, identity
   - discriminator A/B
   - perceptual, total, grad norm
4. plot 3x3 grid (9 subplots)
5. save figure as training_history.png

Output:
- PNG figure saved to model directory

---

## 5) _flatten_history

Input:
- nested history dict

Dataflow:
1. iterate epochs and batches
2. build row dict per batch

Output:
- list of row dicts, length = total batches across all epochs
