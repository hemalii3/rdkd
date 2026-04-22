"""Configuration constants and parameters for the project.

This module contains all configuration parameters used throughout the
project. It is intentionally organized by concern so the clustering
experiment is reproducible and method-specific settings are easy to trace.
"""

from pathlib import Path

# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# =============================================================================
# Data files
# =============================================================================

TRAIN_FILE = "sample_23.csv"
TEST_FILE = "sample_24.csv"
TRAIN_PATH = DATA_DIR / TRAIN_FILE
TEST_PATH = DATA_DIR / TEST_FILE

# =============================================================================
# Generic cleaning defaults
# =============================================================================

MISSING_VALUE_STRATEGY = "interpolate"
NEGATIVE_VALUE_ACTION = "set_zero"

# =============================================================================
# Household screening / special-regime thresholds
# =============================================================================

ZERO_HEAVY_THRESHOLD_PCT = 75.0
NEAR_CONSTANT_STD_THRESHOLD = 0.10

EXCLUDE_ZERO_HEAVY_FROM_MAIN_CLUSTERING = True

DEGENERATE_SPECIAL_LABEL = -99

# =============================================================================
# Feature-based clustering settings
# =============================================================================

FEATURE_CLUSTERING_SCALE_METHOD = "standard"
FEATURE_CORRELATION_THRESHOLD = 0.90
FEATURE_LOW_VARIANCE_THRESHOLD = 1e-6
FEATURE_PCA_VARIANCE_THRESHOLD = 0.90
FEATURE_USE_PCA = False 
FEATURE_WARD_LINKAGE = "ward"
FEATURE_INCLUDE_NORMALIZED_SHAPE_BLOCK = True

USE_FEATURE_BLOCK_WEIGHTS = True

FEATURE_BLOCK_WEIGHTS = {
    "level_dispersion": 0.00,
    "weekly": 1.00,
    "shape_normalized": 1.20,
    "persistence": 0.50,
    "trend": 0.30,
}

# =============================================================================
# k-Shape settings
# =============================================================================

KSHAPE_NORMALIZATION = "zscore_rowwise"

# =============================================================================
# Cluster selection and stability
# =============================================================================

MIN_CLUSTER_SIZE = 300
USE_STABILITY_FOR_K_SELECTION = True
STABILITY_N_RUNS = 10
STABILITY_SAMPLE_FRAC = 0.80
LEVEL_DOMINANCE_WARNING_THRESHOLD = 3.0

# =============================================================================
# Reporting / interpretation
# =============================================================================

SAVE_CLUSTER_DIAGNOSTICS = True
SAVE_REPRESENTATIVE_SERIES = True
SAVE_PHASE1_SUMMARY = True

RANDOM_SEED = 42

# =============================================================================
# Data parameters
# =============================================================================

# Based on the actual uploaded 2023 file.
TRAIN_DAYS = 365  # 2023
TEST_DAYS = 366   # 2024 (leap year)
TRAIN_YEAR = 2023
TEST_YEAR = 2024

# =============================================================================
# Experiment design
# =============================================================================

CLUSTERING_METHODS = [
    "feature_ward",   # main scalable candidate
    "kshape",         # main sequence-shape benchmark
]

PRIMARY_CLUSTERING_METHOD = "feature_ward"

# =============================================================================
# Clustering search space
# =============================================================================

K_SEARCH_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

SPECIAL_CLUSTER_LABELS = {
    -1: "all_zero",
    -2: "near_constant",
    -3: "zero_heavy",
    -99: "degenerate_other"
}

KEEP_SPECIAL_CLUSTERS_FOR_FORECASTING = True
MIN_SPECIAL_CLUSTER_SIZE_FOR_FORECASTING = 300

# =============================================================================
# Task 2 forecasting settings
# =============================================================================

FORECAST_HORIZON = 366  # 2024 is leap year
LAG_FEATURES = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]

# =============================================================================
# Evaluation metrics
# =============================================================================

PRIMARY_METRIC = "mae"
ADDITIONAL_METRICS = ["rmse", "mape", "r2"]

# =============================================================================
# Visualization
# =============================================================================

FIGURE_SIZE = (12, 6)
DPI = 100
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# =============================================================================
# Computational settings
# =============================================================================

N_JOBS = -1
VERBOSE = 1

# =============================================================================
# Output paths
# =============================================================================

MODELS_DIR = RESULTS_DIR / "models"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"

for directory in [MODELS_DIR, PREDICTIONS_DIR, FIGURES_DIR, METRICS_DIR, DIAGNOSTICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

