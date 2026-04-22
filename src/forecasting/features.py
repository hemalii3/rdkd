"""
features.py
-----------
Feature engineering for household energy forecasting.

Converts wide-format (households × dates) DataFrames into a long-format
feature matrix ready for LightGBM.

All lag and rolling features use shift(1) so they only ever look at *past*
values — no same-day leakage into the target.

The stack-then-split pattern ensures January 2024 lag features can look
back into December 2023, which is intentional and not data leakage
(we are not using 2024 targets; only 2023 consumption values as lookback).
"""

import pandas as pd
import numpy as np


def wide_to_long(wide_df: pd.DataFrame, value_name: str = "consumption") -> pd.DataFrame:
    """Convert wide (households × dates) DataFrame to long format.

    Parameters
    ----------
    wide_df    : DataFrame with household IDs as index, date strings as columns.
    value_name : Name for the value column in the output.

    Returns
    -------
    DataFrame with columns [household_id, date, <value_name>],
    sorted by (household_id, date).
    """
    df_reset = wide_df.reset_index()
    id_col = df_reset.columns[0]
    long_df = df_reset.melt(id_vars=id_col, var_name="date", value_name=value_name)
    long_df = long_df.rename(columns={id_col: "household_id"})
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df = long_df.sort_values(["household_id", "date"]).reset_index(drop=True)
    return long_df


def build_features(
    long_df: pd.DataFrame,
    lag_features: list,
    rolling_windows: list,
    household_means: pd.Series = None,
) -> pd.DataFrame:
    """Build lag, rolling, and calendar features for a long-format DataFrame.

    Parameters
    ----------
    long_df         : Long DataFrame with columns [household_id, date, consumption].
    lag_features    : List of lag days, e.g. [1, 7, 14, 30].
    rolling_windows : Window sizes for rolling mean/std, e.g. [7, 14, 30].
    household_means : Series (household_id -> mean consumption) computed on
                      2023 training data only. Used as a stable household signal.

    Returns
    -------
    DataFrame with original columns plus all engineered feature columns.
    """
    df = long_df.copy()
    df = df.sort_values(["household_id", "date"])

    grp = df.groupby("household_id")["consumption"]

    # Lag features
    for lag in lag_features:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Rolling mean and std (shift(1) so no same-day leakage)
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
        )

    # Calendar features
    df["day_of_week"]  = df["date"].dt.dayofweek        # 0=Mon, 6=Sun
    df["month"]        = df["date"].dt.month
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Household mean from 2023 training data (stable baseline signal)
    if household_means is not None:
        df["household_mean"] = df["household_id"].map(household_means)

    return df


def prepare_features(
    train_wide: pd.DataFrame,
    test_wide: pd.DataFrame,
    regular_ids,
    lag_features: list,
    rolling_windows: list,
) -> tuple:
    """Full feature preparation pipeline for regular households.

    Stacks 2023 + 2024 before computing lags so that January 2024 can look
    back into December 2023. Splits back by year before returning.

    Parameters
    ----------
    train_wide      : Wide 2023 DataFrame (households × 365 dates).
    test_wide       : Wide 2024 DataFrame (households × 366 dates).
    regular_ids     : Index of regular household IDs (non-degenerate).
    lag_features    : List of lag days.
    rolling_windows : List of rolling window sizes.

    Returns
    -------
    train_features_clean : 2023 feature rows with NaN lags dropped.
    test_features        : 2024 feature rows (no NaNs, via train history).
    feature_cols         : List of feature column names.
    household_means      : Series of per-household 2023 mean (for reuse).
    """
    reg_in_train = regular_ids.intersection(train_wide.index)
    reg_in_test  = regular_ids.intersection(test_wide.index)

    train_reg = train_wide.loc[reg_in_train]
    test_reg  = test_wide.loc[reg_in_test]

    # Household means computed exclusively from 2023
    household_means = train_wide.mean(axis=1)

    train_long = wide_to_long(train_reg)
    test_long  = wide_to_long(test_reg)

    # Stack so 2024 lags can look back into 2023
    combined = pd.concat([train_long, test_long], ignore_index=True)
    combined = combined.sort_values(["household_id", "date"]).reset_index(drop=True)

    combined_feats = build_features(
        combined,
        lag_features=lag_features,
        rolling_windows=rolling_windows,
        household_means=household_means,
    )

    is_2023 = combined_feats["date"].dt.year == 2023
    is_2024 = combined_feats["date"].dt.year == 2024

    train_features = combined_feats[is_2023].copy()
    test_features  = combined_feats[is_2024].copy()

    feature_cols = [
        c for c in train_features.columns
        if c not in ["household_id", "date", "consumption"]
    ]

    # Drop rows where any lag is NaN (first ~30 days per household in train only)
    lag_cols = [f"lag_{l}" for l in lag_features]
    train_features_clean = train_features.dropna(subset=lag_cols).copy()

    # Fill any residual NaNs in test (should be zero after stacking 2023+2024)
    nan_count = test_features[feature_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"  Warning: {nan_count} NaN values in test features — filling with 0")
        test_features[feature_cols] = test_features[feature_cols].fillna(0)

    return train_features_clean, test_features, feature_cols, household_means
