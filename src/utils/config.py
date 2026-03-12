"""Configuration constants and parameters for the project.

This module contains all configuration parameters used throughout the project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
TRAIN_FILE = "sample_23.csv"
TEST_FILE = "sample_24.csv"
TRAIN_PATH = DATA_DIR / TRAIN_FILE
TEST_PATH = DATA_DIR / TEST_FILE

# Data parameters
N_HOUSEHOLDS = 17548
TRAIN_DAYS = 365  # 2023
TEST_DAYS = 366   # 2024 (leap year)
TRAIN_YEAR = 2023
TEST_YEAR = 2024

# Clustering parameters
MIN_CLUSTERS = 2
MAX_CLUSTERS = 20
DEFAULT_K = 5

# Forecasting parameters
FORECAST_HORIZON = 366  # 2024 is leap year
LAG_FEATURES = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]

# Random seed for reproducibility
RANDOM_SEED = 42

# Feature engineering
STATISTICAL_FEATURES = [
    'mean', 'median', 'std', 'min', 'max', 
    'range', 'q25', 'q75', 'skew', 'kurtosis'
]

# Evaluation metrics
PRIMARY_METRIC = 'mae'
ADDITIONAL_METRICS = ['rmse', 'mape', 'r2']

# Visualization
FIGURE_SIZE = (12, 6)
DPI = 100
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Computational
N_JOBS = -1  # Use all CPU cores
VERBOSE = 1

# Model save paths
MODELS_DIR = RESULTS_DIR / "models"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories if they don't exist
for directory in [MODELS_DIR, PREDICTIONS_DIR, FIGURES_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
