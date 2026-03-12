# Quick Start Guide: Running the EDA Notebook

## Prerequisites

- Python 3.12.6 virtual environment activated
- All dependencies installed (from requirements.txt)
- Data files in `Data/` folder

## Step-by-Step Instructions

### 1. Activate Virtual Environment

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### 2. Start Jupyter Notebook

```powershell
jupyter notebook
```

This will open your browser automatically.

### 3. Navigate to the Notebook

- In the browser, navigate to: `notebooks/`
- Click on: `01_EDA.ipynb`

### 4. Run the Analysis

**Option A: Run All Cells at Once**

- Menu: `Cell` → `Run All`
- Wait for all cells to execute (~2-5 minutes)

**Option B: Run Cell by Cell**

- Click on first cell
- Press `Shift + Enter` to run and move to next cell
- Repeat for each cell

### 5. Expected Output

The notebook will generate:

- **Data quality reports** with statistics
- **15+ visualizations** including:
  - Sample time series plots
  - Distribution histograms
  - Box plots for outliers
  - Day-of-week patterns
  - Monthly seasonality
  - Correlation heatmaps
  - Category scatter plots
- **Statistical summaries** for 17,547 households
- **Saved CSV file**: `results/household_stats_eda.csv`

### 6. Saving Results

The notebook automatically saves household statistics to:

```
results/household_stats_eda.csv
```

This file contains all computed statistics for every household and will be used in future phases.

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution:**

```powershell
$env:PYTHONPATH = "C:\Users\Lundrim\Desktop\RDK Projectt"
jupyter notebook
```

### Issue: "FileNotFoundError: sample_23.csv not found"

**Solution:**

- Ensure `Data/sample_23.csv` and `Data/sample_24.csv` exist
- Check that you're running from the project root directory

### Issue: Kernel keeps dying

**Solution:**

- Large dataset may require more memory
- Close other applications
- Consider analyzing a subset first

## What to Look For

### Key Insights

1. **Data Quality Section:**
   - Check for zero consumption percentages
   - Note any outlier households
   - Review value ranges

2. **Descriptive Statistics:**
   - Mean consumption distribution
   - Variability (CV) patterns
   - Skewness of distributions

3. **Temporal Patterns:**
   - Weekday vs weekend differences
   - Seasonal variations (winter vs summer)
   - Monthly trends

4. **Preliminary Categories:**
   - 7 household groups identified
   - Note which category has most households
   - Check category characteristics

## Next Steps After Running EDA

1. **Review the findings** in the summary section
2. **Discuss with team** which patterns are interesting
3. **Plan Phase 2** based on insights:
   - How to handle zero-consumption days?
   - Which features to extract?
   - How to treat outliers?

## Quick Summary Commands

```powershell
# Full workflow
.\venv\Scripts\Activate.ps1
jupyter notebook
# Then: Open notebooks/01_EDA.ipynb → Run All

# Alternative: Run as Python script (if notebook has issues)
python -m nbconvert --to python notebooks/01_EDA.ipynb
python notebooks/01_EDA.py
```

## Time Estimate

- **First-time run:** 3-5 minutes
- **Subsequent runs:** 2-3 minutes

## Output Files

After running, you'll have:

- `results/household_stats_eda.csv` (~3 MB)
- All visualizations displayed in notebook
- Statistical summaries in notebook cells

---

**Need help?** See `docs/Phase1_EDA_Summary.md` for detailed findings.
