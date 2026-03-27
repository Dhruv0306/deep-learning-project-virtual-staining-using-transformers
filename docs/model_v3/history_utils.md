# model_v3/history_utils.py - v3 History Utilities

Source of truth: ../../model_v3/history_utils.py

Role: Persist and visualize v3 training history.

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
  - Loss_DiT
  - Loss_Perceptual
  - GradNorm

Flattened row schema:
- Epoch, Batch, Loss_DiT, Loss_Perceptual, GradNorm

---

## 1) save_history_to_csv_v3

Input:
- history dict
- filename

Dataflow:
1. _flatten_history(history) -> list of row dicts length M
2. pandas DataFrame from rows -> shape (M,5)
3. write CSV

Output:
- CSV file with 5 columns

---

## 2) append_history_to_csv_v3

Input:
- history chunk dict
- filename

Dataflow:
1. if history empty -> return
2. flatten to rows length K
3. DataFrame shape (K,5)
4. append mode write, with header only when file missing/empty

Output:
- CSV appended with K rows

---

## 3) load_history_from_csv_v3

Input:
- filename

Dataflow:
1. read CSV to DataFrame shape (R,5)
2. iterate each row
3. rebuild nested structure history[epoch][batch]

Output:
- reconstructed history dict

---

## 4) visualize_history_v3

Input:
- history dict
- optional model_dir

Dataflow:
1. collect sorted epoch list, length E
2. per epoch, aggregate batch values:
   - Loss_DiT list length Be
   - Loss_Perceptual list length Be
   - GradNorm list length Be
3. compute per-epoch means:
   - avg_loss: (E,)
   - avg_loss_perc: (E,)
   - avg_grad: (E,)
4. plot 3 subplots (1x3)
5. save figure training_history.png

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
