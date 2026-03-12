# Contributing Guidelines

## Code Style

This project follows **PEP8** conventions with some specific guidelines:

### Python Style Guide

1. **Line Length:** Maximum 88 characters (Black formatter default)

2. **Naming Conventions:**
   - Variables and functions: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_CASE`
   - Private methods: `_leading_underscore`

3. **Imports:**
   ```python
   # Standard library
   import os
   import sys
   
   # Third-party
   import numpy as np
   import pandas as pd
   
   # Local imports
   from src.utils import data_loader
   ```

4. **Docstrings:** Use Google style
   ```python
   def compute_mae(predictions, actuals):
       """Compute Mean Absolute Error.
       
       Args:
           predictions (np.ndarray): Predicted values, shape (n_samples,)
           actuals (np.ndarray): Actual values, shape (n_samples,)
       
       Returns:
           float: Mean absolute error
       
       Raises:
           ValueError: If input shapes don't match
       
       Example:
           >>> preds = np.array([1, 2, 3])
           >>> acts = np.array([1.1, 2.2, 2.9])
           >>> compute_mae(preds, acts)
           0.1333
       """
       if predictions.shape != actuals.shape:
           raise ValueError("Input shapes must match")
       return np.mean(np.abs(predictions - actuals))
   ```

5. **Comments:**
   - Use comments to explain **why**, not **what**
   - Keep comments updated with code changes
   - Avoid obvious comments

6. **Type Hints:** Encouraged for complex functions
   ```python
   def load_household_data(household_id: int, year: int) -> pd.DataFrame:
       """Load time series data for a specific household."""
       pass
   ```

---

## Code Formatting

We use **Black** for automatic formatting:

```bash
# Format all Python files
black src/ notebooks/

# Check formatting without changes
black --check src/
```

We use **Flake8** for linting:

```bash
# Check code quality
flake8 src/ --max-line-length=88
```

---

## Git Workflow

### Branch Naming
- Feature: `feature/clustering-kmeans`
- Bugfix: `fix/missing-values-handling`
- Documentation: `docs/update-readme`

### Commit Messages
Follow conventional commits:

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Example:**
```
feat: Add K-Means clustering implementation

- Implemented K-Means with k=3 to 10 range
- Added elbow method visualization
- Computed silhouette scores

Closes #12
```

### Pull Request Process
1. Create feature branch from `main`
2. Make changes and commit
3. Push to remote
4. Create pull request with description
5. Request review from team member
6. Merge after approval

---

## Testing

### Unit Tests
Place tests in `tests/` directory:

```python
# tests/test_preprocessing.py
import pytest
from src.preprocessing import handle_missing_values

def test_handle_missing_values():
    """Test missing value imputation."""
    data = pd.DataFrame({'A': [1, np.nan, 3]})
    result = handle_missing_values(data, method='forward_fill')
    assert result['A'].isna().sum() == 0
```

Run tests:
```bash
pytest tests/
```

---

## File Organization

### Source Code Structure
```
src/
├── __init__.py
├── preprocessing/
│   ├── __init__.py
│   ├── data_loader.py       # Load CSV files
│   ├── cleaning.py          # Missing values, outliers
│   └── features.py          # Feature extraction
├── clustering/
│   ├── __init__.py
│   ├── algorithms.py        # K-Means, DBSCAN, etc.
│   ├── evaluation.py        # Silhouette, elbow method
│   └── visualization.py     # Cluster plots
├── forecasting/
│   ├── __init__.py
│   ├── models.py            # ARIMA, Prophet, XGBoost
│   ├── train.py             # Training pipeline
│   └── predict.py           # Prediction generation
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py           # MAE, RMSE, etc.
│   └── comparison.py        # Cluster vs global
└── utils/
    ├── __init__.py
    ├── config.py            # Configuration constants
    └── helpers.py           # Utility functions
```

### Notebooks Organization
```
notebooks/
├── 01_EDA.ipynb                    # Exploratory analysis
├── 02_Feature_Engineering.ipynb   # Feature extraction
├── 03_Clustering.ipynb            # Clustering experiments
├── 04_Forecasting.ipynb           # Model training
└── 05_Evaluation.ipynb            # Results analysis
```

---

## Configuration Management

Use a config file for constants:

```python
# src/utils/config.py

# Data paths
DATA_DIR = "Data"
TRAIN_FILE = "sample_23.csv"
TEST_FILE = "sample_24.csv"

# Clustering parameters
MIN_CLUSTERS = 2
MAX_CLUSTERS = 20
DEFAULT_K = 5

# Forecasting parameters
FORECAST_HORIZON = 366  # 2024 is leap year
LAG_FEATURES = [1, 7, 14, 30]

# Random seed for reproducibility
RANDOM_SEED = 42
```

---

## Error Handling

Always handle errors gracefully:

```python
def load_csv_data(filepath):
    """Load CSV file with error handling."""
    try:
        data = pd.read_csv(filepath)
        if data.empty:
            raise ValueError("CSV file is empty")
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        raise
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
```

---

## Performance Considerations

1. **Memory Management:**
   - Use chunking for large files
   - Delete unused DataFrames
   - Use appropriate data types (int16 vs int64)

2. **Parallel Processing:**
   ```python
   from joblib import Parallel, delayed
   
   results = Parallel(n_jobs=-1)(
       delayed(process_household)(hh_id) 
       for hh_id in household_ids
   )
   ```

3. **Vectorization:**
   - Prefer pandas/numpy operations over loops
   - Use `.apply()` only when necessary

---

## Documentation

### README Updates
Update README when:
- Adding new dependencies
- Changing project structure
- Adding new features

### Inline Documentation
- Document complex algorithms
- Add references to papers/sources
- Explain non-obvious design choices

---

## Before Committing

**Checklist:**
- [ ] Code follows PEP8 style
- [ ] Functions have docstrings
- [ ] No hardcoded paths (use relative paths)
- [ ] No print statements (use logging)
- [ ] Tests pass (if applicable)
- [ ] Black formatting applied
- [ ] Research diary updated
- [ ] Meaningful commit message

---

## Questions?

Contact team members via Slack or discuss in weekly meetings.

**Last Updated:** March 12, 2026
