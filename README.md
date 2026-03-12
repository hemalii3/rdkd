# RDK Project: Time Series Forecasting with Clustering

**Course:** Recent Developments in KDD (SS 2026)  
**University of Vienna, Faculty of Computer Science**

## 📋 Project Overview

This project explores clustering as a strategic preprocessing step for large-scale time series forecasting. We analyze daily energy consumption data from 17,548 households to improve forecasting accuracy by grouping similar consumption patterns.

### Dataset

- **Training:** 365 days (2023) - 17,548 households
- **Testing:** 366 days (2024) - 17,548 households
- **Total Observations:** ~6.4 million data points per year

### Objectives

1. **Clustering:** Group households with similar consumption patterns
2. **Forecasting:** Predict 366-day ahead consumption (cluster-based vs global model)
3. **Evaluation:** Compare approaches using household-level Mean Absolute Error (MAE)

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Project Structure

```
RDK_Project/
├── Data/                    # Raw CSV files (sample_23.csv, sample_24.csv)
├── src/                     # Production Python code
│   ├── preprocessing/      # Data cleaning and feature extraction
│   ├── clustering/         # Clustering algorithms and evaluation
│   ├── forecasting/        # Time series forecasting models
│   ├── evaluation/         # Metrics and evaluation logic
│   └── utils/              # Helper functions
├── notebooks/              # Jupyter notebooks for exploration
├── results/                # Outputs, figures, metrics
├── docs/                   # Documentation and reports
├── research_diaries/       # Individual work logs
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 📊 How to Run

### Phase 1: Data Exploration

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### Phase 2: Clustering

```python
from src.clustering.cluster_analysis import perform_clustering

# Run clustering pipeline
cluster_assignments = perform_clustering(data='Data/sample_23.csv', k=5)
```

### Phase 3: Forecasting

```python
from src.forecasting.train_models import train_cluster_models, train_global_model

# Train cluster-based models
cluster_models = train_cluster_models(cluster_assignments)

# Train global baseline
global_model = train_global_model()
```

### Phase 4: Evaluation

```python
from src.evaluation.evaluate import evaluate_forecasts

# Evaluate and compare
results = evaluate_forecasts(
    cluster_predictions='results/cluster_preds.csv',
    global_predictions='results/global_preds.csv',
    test_data='Data/sample_24.csv'
)
```

---

## 🛠️ Development Guidelines

### Code Style

- Follow **PEP8** conventions
- Use descriptive variable names
- Add docstrings to all functions (Google style)
- Maximum line length: 88 characters (Black formatter)

### Git Workflow

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit: `git commit -m "descriptive message"`
3. Push to remote: `git push origin feature/your-feature`
4. Create pull request for review

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Check code style
flake8 src/
black --check src/
```

---

## 📚 Key Components

### 1. Data Preprocessing

- Missing value handling
- Outlier detection and treatment
- Feature extraction (statistical, temporal, seasonal)
- Data normalization

### 2. Clustering

- Algorithm comparison (K-Means, Hierarchical, DBSCAN, GMM)
- Optimal K estimation (Elbow, Silhouette, Davies-Bouldin)
- Cluster interpretation and visualization

### 3. Forecasting

- Statistical models (ARIMA, SARIMA, ETS)
- Machine Learning (Random Forest, XGBoost, LightGBM)
- Deep Learning (LSTM, GRU) - optional
- Time series specific (Prophet, N-BEATS)

### 4. Evaluation

- Primary metric: Mean Absolute Error (MAE)
- Additional metrics: RMSE, MAPE, R²
- Household-level and cluster-level analysis
- Statistical comparison tests

---

## 👥 Team Members

- Member 1: [Name] - EDA & Preprocessing
- Member 2: [Name] - Clustering
- Member 3: [Name] - Forecasting
- Member 4: [Name] - Evaluation & Reporting

---

## 📅 Timeline

| Week         | Dates            | Milestone           |
| ------------ | ---------------- | ------------------- |
| 1            | Mar 12-18        | EDA Complete        |
| 2            | Mar 19-25        | Feature Engineering |
| 3            | Mar 26-Apr 1     | Clustering          |
| 4            | Apr 2-8          | Forecasting         |
| 5            | Apr 9-15         | Evaluation          |
| 6            | Apr 16-22        | Report Writing      |
| 7            | Apr 23-29        | Final Polish        |
| **Deadline** | **Apr 30 23:59** | **Submission**      |

---

## 📖 References

- Hyndman, R.J., & Athanasopoulos, G. (2021). _Forecasting: Principles and Practice_
- Aghabozorgi, S., et al. (2015). "Time-series clustering – A decade review"
- Scikit-learn Documentation: https://scikit-learn.org/
- TSlearn Documentation: https://tslearn.readthedocs.io/

---

## 📝 License

This project is for academic purposes only (University of Vienna, SS 2026).

---

## ⚠️ Important Notes

- **DO NOT commit large CSV files to Git** (already in .gitignore)
- **2024 data is for testing ONLY** - never use for training/clustering
- Keep research diaries updated daily
- Cite all external sources and libraries used

---

## 📧 Contact

For questions, use the course forum or contact the tutors.

**Last Updated:** March 12, 2026
