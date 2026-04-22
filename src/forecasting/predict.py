"""
predict.py
----------
Generate 366 daily predictions per household for 2024.

Three household types:
  - Regular (cluster >= 0)  : cluster-specific LightGBM model
  - Unmapped regular         : global model fallback
  - Degenerate (cluster < 0): household's own 2023 historical mean

Output columns: household_id, date, predicted, cluster, model_used
"""

import numpy as np
import pandas as pd

from src.forecasting.features import wide_to_long


def predict_point(
    test_features_clean: pd.DataFrame,
    test_degenerate: pd.DataFrame,
    train_wide: pd.DataFrame,
    feature_cols: list,
    cluster_labels: list,
    cluster_models: dict,
    global_model,
    cluster_series: pd.Series,
    degenerate_ids,
) -> pd.DataFrame:
    """Generate one point prediction per household-day for all of 2024.

    Parameters
    ----------
    test_features_clean : Feature rows for regular 2024 households.
    test_degenerate     : Wide 2024 DataFrame for degenerate households.
    train_wide          : Wide 2023 DataFrame — used only to compute
                          degenerate household means (no 2024 data used).
    feature_cols        : Feature column names.
    cluster_labels      : List of regular cluster label integers.
    cluster_models      : {cluster_label -> trained LGBMRegressor}.
    global_model        : Global LGBMRegressor (fallback for unmapped rows).
    cluster_series      : Series mapping household_id -> cluster label.
    degenerate_ids      : Index of degenerate household IDs.

    Returns
    -------
    DataFrame with columns: household_id, date, predicted, cluster, model_used
    Sorted by (household_id, date). One row per household per day = 366 rows/household.
    """
    results = []

    # --- Regular households: one cluster model each ---
    for label in cluster_labels:
        mask = test_features_clean["cluster"] == label
        df_c = test_features_clean[mask]
        if df_c.empty:
            continue
        preds = np.clip(
            cluster_models.get(label, global_model).predict(df_c[feature_cols].values),
            0, None,
        )
        out = df_c[["household_id", "date"]].copy()
        out["predicted"]  = preds
        out["cluster"]    = label
        out["model_used"] = "cluster_lgbm"
        results.append(out)

    # --- Unmapped regular households: global model fallback ---
    unmapped = test_features_clean["cluster"].isna()
    if unmapped.sum() > 0:
        df_u  = test_features_clean[unmapped]
        preds = np.clip(global_model.predict(df_u[feature_cols].values), 0, None)
        out   = df_u[["household_id", "date"]].copy()
        out["predicted"]  = preds
        out["cluster"]    = -99
        out["model_used"] = "global_fallback"
        results.append(out)
        print(f"  Warning: {unmapped.sum()} rows unmapped — global fallback used")

    reg_df = pd.concat(results, ignore_index=True)

    # --- Degenerate households: 2023 historical mean repeated for all 366 days ---
    # Mean computed from train_wide (2023 only) — no 2024 data used here.
    degen_train = train_wide.loc[train_wide.index.intersection(degenerate_ids)]
    degen_means = degen_train.mean(axis=1)

    test_degen_long = wide_to_long(test_degenerate)
    test_degen_long["predicted"]  = (
        test_degen_long["household_id"].map(degen_means).fillna(0).clip(lower=0)
    )
    test_degen_long["cluster"]    = test_degen_long["household_id"].map(cluster_series)
    test_degen_long["model_used"] = "historical_mean_baseline"

    degen_df = test_degen_long[
        ["household_id", "date", "predicted", "cluster", "model_used"]
    ].copy()

    all_preds = pd.concat([reg_df, degen_df], ignore_index=True)
    all_preds = all_preds.sort_values(["household_id", "date"]).reset_index(drop=True)
    return all_preds
