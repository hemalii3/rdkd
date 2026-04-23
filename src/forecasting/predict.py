from collections.abc import Sequence
from numbers import Real
import numpy as np
import pandas as pd

from src.forecasting.features import wide_to_long


def predict_point(
    test_wide: pd.DataFrame,
    train_wide: pd.DataFrame,
    feature_cols: list,
    lag_features: list,
    rolling_windows: list,
    cluster_labels: list,
    cluster_models: dict,
    global_model,
    cluster_series: pd.Series,
    household_means: pd.Series,
    cluster_dow_profile: pd.DataFrame | None = None,
    cluster_month_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate leakage-free 2024 point forecasts recursively.

    For each household, each day's features are built from:
      - 2023 observed history
      - previously predicted 2024 values only
      - fixed cluster-derived priors computed from 2023 only

    No true 2024 consumption values are used for forecasting.

    Forecasting expects a forecasting-ready handoff: all households must already
    have nonnegative cluster assignments.
    """
    results = []

    cluster_numeric = pd.to_numeric(cluster_series, errors="coerce")
    if cluster_numeric.isna().any():
        raise ValueError("cluster_series contains missing/non-numeric cluster labels.")
    if (cluster_numeric < 0).any():
        raise ValueError(
            "predict_point() received negative cluster labels, but forecasting "
            "expects reassigned nonnegative handoff labels only."
        )

    regular_ids = cluster_numeric.index.intersection(test_wide.index)
    test_dates = pd.to_datetime(test_wide.columns)

    hh_cluster = cluster_numeric.reindex(regular_ids).astype(int)

    history = {
        hh: train_wide.loc[hh].tolist()
        for hh in regular_ids.intersection(train_wide.index)
    }

    def _rolling_mean(values, window):
        vals = values[-window:] if len(values) >= window else values
        if len(vals) == 0:
            return 0.0
        arr = np.asarray(vals, dtype=np.float64)  # force real float dtype
        return float(arr.mean())

    def _safe_float_or_nan(value: object) -> float:
        if isinstance(value, Real):  # excludes complex
            return float(value)
        return float("nan")

    def _lookup_cluster_dow_mean(cluster_value, dow_value):
        if cluster_dow_profile is None or cluster_dow_profile.empty:
            return np.nan
        col = f"cluster_dow_{int(dow_value)}"
        if cluster_value not in cluster_dow_profile.index or col not in cluster_dow_profile.columns:
            return np.nan
        v = cluster_dow_profile.loc[cluster_value, col]
        return _safe_float_or_nan(v)

    def _lookup_cluster_month_mean(cluster_value, month_value):
        if cluster_month_profile is None or cluster_month_profile.empty:
            return np.nan
        col = f"cluster_month_{int(month_value)}"
        if cluster_value not in cluster_month_profile.index or col not in cluster_month_profile.columns:
            return np.nan
        v = cluster_month_profile.loc[cluster_value, col]
        return _safe_float_or_nan(v)

    for date in test_dates:
        day_rows = []

        for hh in regular_ids:
            if hh not in history:
                history[hh] = []

            hist = history[hh]
            hh_mean = float(household_means.get(hh, 0.0))
            cluster_value = int(hh_cluster.get(hh, 0))

            row = {
                "household_id": hh,
                "date": date,
                "day_of_week": date.dayofweek,
                "month": date.month,
                "day_of_year": date.dayofyear,
                "is_weekend": int(date.dayofweek >= 5),
                "week_of_year": int(date.isocalendar().week),
                "day_of_year_sin": float(np.sin(2 * np.pi * date.dayofyear / 366.0)),
                "day_of_year_cos": float(np.cos(2 * np.pi * date.dayofyear / 366.0)),
                "week_of_year_sin": float(np.sin(2 * np.pi * int(date.isocalendar().week) / 53.0)),
                "week_of_year_cos": float(np.cos(2 * np.pi * int(date.isocalendar().week) / 53.0)),
                "household_mean": hh_mean,
                "cluster": cluster_value,
            }

            for lag in lag_features:
                row[f"lag_{lag}"] = hist[-lag] if len(hist) >= lag else 0.0

            for window in rolling_windows:
                vals = hist[-window:] if len(hist) >= window else hist
                row[f"rolling_mean_{window}"] = float(np.mean(vals)) if len(vals) > 0 else 0.0
                row[f"rolling_std_{window}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

            row["cluster_dow_mean"] = _lookup_cluster_dow_mean(cluster_value, date.dayofweek)
            row["cluster_month_mean"] = _lookup_cluster_month_mean(cluster_value, date.month)
            row["household_minus_cluster_dow_mean"] = hh_mean - row["cluster_dow_mean"]
            row["household_minus_cluster_month_mean"] = hh_mean - row["cluster_month_mean"]

            day_rows.append(row)

        day_df = pd.DataFrame(day_rows)

        for label in cluster_labels:
            mask = day_df["cluster"] == label
            if mask.sum() == 0:
                continue

            X = day_df.loc[mask, feature_cols].values
            preds = np.clip(
                cluster_models.get(label, global_model).predict(X),
                0,
                None,
            )

            out = day_df.loc[mask, ["household_id", "date"]].copy()
            out["predicted"] = preds
            out["cluster"] = label
            out["model_used"] = "cluster_lgbm"
            results.append(out)

            for hh, pred in zip(out["household_id"].values, preds):
                history[hh].append(float(pred))

        unmapped = ~day_df["cluster"].isin(cluster_labels)
        if unmapped.sum() > 0:
            X = day_df.loc[unmapped, feature_cols].values
            preds = np.clip(global_model.predict(X), 0, None)

            out = day_df.loc[unmapped, ["household_id", "date"]].copy()
            out["predicted"] = preds
            out["cluster"] = -99
            out["model_used"] = "global_fallback"
            results.append(out)

            for hh, pred in zip(out["household_id"].values, preds):
                history[hh].append(float(pred))

    all_preds = pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["household_id", "date", "predicted", "cluster", "model_used"]
    )
    all_preds = all_preds.sort_values(["household_id", "date"]).reset_index(drop=True)
    return all_preds