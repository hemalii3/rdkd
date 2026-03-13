# 📌 How to Resume Notebook Execution

## Problem

Running all cells from the start takes too long (especially feature extraction which processes 17,547 households).

## Solution: Save & Load Checkpoints

### 1. Save Cleaned Data (After Cell 10)

Add and run this cell **after the cleaning visualization** (Cell 10):

```python
# Save cleaned data checkpoint
checkpoint_path = os.path.join(RESULTS_DIR, 'checkpoint_cleaned_data.csv')
df_cleaned.to_csv(checkpoint_path)
print(f"✅ Checkpoint saved: {checkpoint_path}")
print(f"   Size: {os.path.getsize(checkpoint_path) / 1024**2:.2f} MB")
```

### 2. Resume from Cleaned Data (Skip Cells 1-10)

To resume from feature extraction without re-running cleaning:

1. **Run Cell 1** (Setup & Imports) - Always needed
2. **Skip Cells 2-10** (Data loading and cleaning)
3. **Run this cell instead:**

```python
# 📥 RESUME: Load cleaned data from checkpoint
checkpoint_path = os.path.join(RESULTS_DIR, 'checkpoint_cleaned_data.csv')
df_cleaned = pd.read_csv(checkpoint_path, index_col=0)
zero_pct = 2.63  # From previous cleaning run

print(f"✅ Loaded cleaned data checkpoint")
print(f"   - Shape: {df_cleaned.shape}")
print(f"   - Memory: {df_cleaned.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"   - Zero percentage: {zero_pct:.2f}%")
```

4. **Continue with Cell 12** (Feature Extraction)

### 3. Save Features (After Feature Extraction)

If feature extraction completes, it already saves `features_all.csv`. To resume from there:

```python
# Resume from saved features
df_features = pd.read_csv(os.path.join(RESULTS_DIR, 'features_all.csv'), index_col=0)
print(f"✅ Loaded features: {df_features.shape}")
```

## Quick Resume Workflow

```
Cell 1 (Imports)
   ↓
Load checkpoint_cleaned_data.csv
   ↓
Cell 12+ (Feature extraction onwards)
```

## Checkpoint Locations

- **After Cleaning:** `results/checkpoint_cleaned_data.csv`
- **After Features:** `results/features_all.csv` (auto-saved)
- **After Selection:** `results/features_selected.csv` (auto-saved)
- **After Scaling:** `results/features_scaled.csv` (auto-saved)

## Tips

1. **Save checkpoints** after long-running cells
2. **Kernel variables persist** - You only need to re-run if you restart the kernel
3. **Use "Run All Above"** to run everything up to a specific cell
4. **Clear outputs** (Edit → Clear All Outputs) to reduce file size before Git commit

---

**Created:** March 13, 2026  
**Related:** Phase 2 Preprocessing Notebook
