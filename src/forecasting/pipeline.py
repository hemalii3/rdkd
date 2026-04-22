"""
pipeline.py
-----------
Trains forecasting models and saves prediction CSVs for 2024.

Three configurations:
  feature_ward  — one LightGBM model per Feature+Ward cluster
  kshape        — one LightGBM model per k-Shape cluster
  global        — single global model, no clustering (dataset-level baseline)

All models are trained on 2023 data only.
Output is one predictions CSV per configuration saved to results/predictions/.

Predictions CSV columns: household_id, date, predicted, cluster, model_used
  - household_id : household identifier
  - date         : forecast date (2024-01-01 to 2024-12-31)
  - predicted    : forecast daily consumption (kWh), clipped to >= 0
  - cluster      : cluster label the household belongs to
  - model_used   : which model produced this prediction

CLI usage (from project root):
    python -m src.forecasting.pipeline                      # all three
    python -m src.forecasting.pipeline --method feature_ward
    python -m src.forecasting.pipeline --method kshape
    python -m src.forecasting.pipeline --method global
    python -m src.forecasting.pipeline --no-save-models
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent          # src/forecasting -> src -> project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import (
    RANDOM_SEED, LAG_FEATURES, ROLLING_WINDOWS,
    RESULTS_DIR, PREDICTIONS_DIR, DIAGNOSTICS_DIR,
)
from src.utils.data_loader import load_train_data, load_test_data

from src.forecasting.features       import prepare_features
from src.forecasting.cluster_loader import load_cluster_assignments
from src.forecasting.models         import (
    train_cluster_models, train_global_model, save_point_models,
)
from src.forecasting.predict        import predict_point

# ---------------------------------------------------------------------------
# !! Update these to match notebook 03 selected k values !!
# ---------------------------------------------------------------------------
CLUSTERING_CONFIG = {
    "feature_ward": {"k": 3},
    "kshape":       {"k": 2},
}


def _load_and_intersect(diagnostics_dir, method, k, train_wide):
    """Load cluster assignments and intersect with available training households."""
    cluster_series, degenerate_ids, regular_ids = load_cluster_assignments(
        diagnostics_dir, method, k
    )
    common         = cluster_series.index.intersection(train_wide.index)
    cluster_series = cluster_series.loc[common]
    regular_ids    = cluster_series[cluster_series >= 0].index
    degenerate_ids = cluster_series[cluster_series <  0].index
    cluster_labels = sorted(cluster_series[cluster_series >= 0].unique().tolist())
    return cluster_series, regular_ids, degenerate_ids, cluster_labels


# ---------------------------------------------------------------------------
# Cluster-based pipeline
# ---------------------------------------------------------------------------

def run_cluster_pipeline(
    method: str,
    k: int,
    train_wide: pd.DataFrame,
    test_wide: pd.DataFrame,
    save_models: bool = True,
) -> pd.DataFrame:
    """Train one model per cluster and produce 366 predictions per household.

    Parameters
    ----------
    method      : "feature_ward" or "kshape".
    k           : Number of regular clusters.
    train_wide  : Wide 2023 DataFrame (households × 365 dates). Training only.
    test_wide   : Wide 2024 DataFrame (households × 366 dates). Inference only.
    save_models : Whether to write .pkl model files to results/models/.

    Returns
    -------
    DataFrame of predictions saved to results/predictions/predictions_2024_{method}.csv
    """
    print(f"\n{'='*60}")
    print(f"  {method}  (k={k})")
    print(f"{'='*60}")

    cluster_series, regular_ids, degenerate_ids, cluster_labels = \
        _load_and_intersect(DIAGNOSTICS_DIR, method, k, train_wide)

    # Feature engineering — trained on 2023, test features use 2023 as lookback
    print("\n[1/3] Building features...")
    train_feats, test_feats, feature_cols, _ = prepare_features(
        train_wide, test_wide, regular_ids, LAG_FEATURES, ROLLING_WINDOWS
    )
    train_feats["cluster"] = train_feats["household_id"].map(cluster_series)
    test_feats["cluster"]  = test_feats["household_id"].map(cluster_series)
    print(f"  Train: {train_feats.shape}  |  Test: {test_feats.shape}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # Train
    print("\n[2/3] Training models on 2023 data...")
    cluster_models = train_cluster_models(
        train_feats, feature_cols, cluster_labels, RANDOM_SEED
    )
    # Global model is also trained here — used as fallback for unmapped households
    global_model = train_global_model(train_feats, feature_cols, RANDOM_SEED)

    if save_models:
        save_point_models(cluster_models, global_model, RESULTS_DIR / "models" / method)

    # Predict 2024
    print("\n[3/3] Predicting 2024...")
    test_degen  = test_wide.loc[degenerate_ids.intersection(test_wide.index)]
    predictions = predict_point(
        test_feats, test_degen, train_wide, feature_cols,
        cluster_labels, cluster_models, global_model,
        cluster_series, degenerate_ids,
    )

    out_path = PREDICTIONS_DIR / f"predictions_2024_{method}.csv"
    predictions.to_csv(out_path, index=False)

    n_hh   = predictions["household_id"].nunique()
    n_days = predictions.groupby("household_id").size()
    print(f"\n  Saved: {out_path}")
    print(f"  Households: {n_hh:,}  |  Days per household: {n_days.min()}–{n_days.max()}")
    return predictions


# ---------------------------------------------------------------------------
# Global (no-clustering) baseline pipeline
# ---------------------------------------------------------------------------

def run_global_pipeline(
    train_wide: pd.DataFrame,
    test_wide: pd.DataFrame,
    save_models: bool = True,
) -> pd.DataFrame:
    """Train one global model on all 2023 households and predict 2024.

    Uses feature_ward cluster assignments only to identify degenerate
    households — the global model itself ignores cluster structure entirely.

    Parameters
    ----------
    train_wide  : Wide 2023 DataFrame. Training only.
    test_wide   : Wide 2024 DataFrame. Inference only.
    save_models : Whether to write model .pkl files.

    Returns
    -------
    DataFrame of predictions saved to results/predictions/predictions_2024_global.csv
    """
    print(f"\n{'='*60}")
    print(f"  global baseline (no clustering)")
    print(f"{'='*60}")

    cfg = CLUSTERING_CONFIG["feature_ward"]
    cluster_series, regular_ids, degenerate_ids, _ = \
        _load_and_intersect(DIAGNOSTICS_DIR, "feature_ward", cfg["k"], train_wide)

    # All regular households treated as one group
    global_cs = cluster_series.copy()
    global_cs.loc[regular_ids] = 0

    print("\n[1/3] Building features...")
    train_feats, test_feats, feature_cols, _ = prepare_features(
        train_wide, test_wide, regular_ids, LAG_FEATURES, ROLLING_WINDOWS
    )
    train_feats["cluster"] = 0
    test_feats["cluster"]  = 0
    print(f"  Train: {train_feats.shape}  |  Test: {test_feats.shape}")

    print("\n[2/3] Training global model on 2023 data...")
    global_model = train_global_model(train_feats, feature_cols, RANDOM_SEED)

    if save_models:
        save_point_models({0: global_model}, global_model, RESULTS_DIR / "models" / "global")

    print("\n[3/3] Predicting 2024...")
    test_degen  = test_wide.loc[degenerate_ids.intersection(test_wide.index)]
    predictions = predict_point(
        test_feats, test_degen, train_wide, feature_cols,
        [0], {0: global_model}, global_model,
        global_cs, degenerate_ids,
    )
    predictions.loc[predictions["cluster"] == 0, "model_used"] = "global_lgbm"

    out_path = PREDICTIONS_DIR / "predictions_2024_global.csv"
    predictions.to_csv(out_path, index=False)

    n_hh   = predictions["household_id"].nunique()
    n_days = predictions.groupby("household_id").size()
    print(f"\n  Saved: {out_path}")
    print(f"  Households: {n_hh:,}  |  Days per household: {n_days.min()}–{n_days.max()}")
    return predictions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run forecasting pipeline.")
    parser.add_argument(
        "--method",
        choices=["feature_ward", "kshape", "global", "all"],
        default="all",
    )
    parser.add_argument("--no-save-models", action="store_true")
    args   = parser.parse_args()
    save   = not args.no_save_models

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_wide = load_train_data()
    test_wide  = load_test_data()
    print(f"  Train: {train_wide.shape}  |  Test: {test_wide.shape}")

    if args.method in ("feature_ward", "all"):
        cfg = CLUSTERING_CONFIG["feature_ward"]
        run_cluster_pipeline("feature_ward", cfg["k"], train_wide, test_wide, save)

    if args.method in ("kshape", "all"):
        cfg = CLUSTERING_CONFIG["kshape"]
        run_cluster_pipeline("kshape", cfg["k"], train_wide, test_wide, save)

    if args.method in ("global", "all"):
        run_global_pipeline(train_wide, test_wide, save)

    print("\nDone. Prediction CSVs saved to:", PREDICTIONS_DIR)


if __name__ == "__main__":
    main()
