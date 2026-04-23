from typing import Optional
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


def build_cluster_profile_features(
    train_wide: pd.DataFrame,
    cluster_series: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build cluster-level seasonal priors from 2023 training data only.

    Parameters
    ----------
    train_wide : DataFrame
        Wide 2023 training panel (households × dates).
    cluster_series : pd.Series
        household_id -> cluster label. Only nonnegative labels are used to build
        cluster priors.

    Returns
    -------
    cluster_dow_profile : DataFrame
        Index = cluster, columns = cluster_dow_0 ... cluster_dow_6
    cluster_month_profile : DataFrame
        Index = cluster, columns = cluster_month_1 ... cluster_month_12
    """
    cluster_series = pd.to_numeric(cluster_series, errors="coerce")
    regular_ids = cluster_series[cluster_series >= 0].index.intersection(train_wide.index)

    if len(regular_ids) == 0:
        empty_dow = pd.DataFrame(
            columns=[f"cluster_dow_{d}" for d in range(7)],
            dtype=float,
        )
        empty_month = pd.DataFrame(
            columns=[f"cluster_month_{m}" for m in range(1, 13)],
            dtype=float,
        )
        empty_dow.index.name = "cluster"
        empty_month.index.name = "cluster"
        return empty_dow, empty_month

    train_long = wide_to_long(train_wide.loc[regular_ids])
    train_long["cluster"] = train_long["household_id"].map(cluster_series).astype(int)

    train_long["day_of_week"] = train_long["date"].dt.dayofweek
    train_long["month"] = train_long["date"].dt.month

    cluster_dow_profile = (
        train_long.groupby(["cluster", "day_of_week"])["consumption"]
        .mean()
        .unstack("day_of_week")
        .reindex(columns=list(range(7)))
        .rename(columns=lambda d: f"cluster_dow_{int(d)}")
        .sort_index()
    )

    cluster_month_profile = (
        train_long.groupby(["cluster", "month"])["consumption"]
        .mean()
        .unstack("month")
        .reindex(columns=list(range(1, 13)))
        .rename(columns=lambda m: f"cluster_month_{int(m)}")
        .sort_index()
    )

    cluster_dow_profile.index.name = "cluster"
    cluster_month_profile.index.name = "cluster"

    return cluster_dow_profile, cluster_month_profile


def build_features(
    long_df: pd.DataFrame,
    lag_features: list[int],
    rolling_windows: list[int],
    household_means: Optional[pd.Series] = None,
    cluster_series: Optional[pd.Series] = None,
    cluster_dow_profile: Optional[pd.DataFrame] = None,
    cluster_month_profile: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build lag, rolling, calendar, and cluster-prior features for long-format data.

    Parameters
    ----------
    long_df : DataFrame
        Long DataFrame with columns [household_id, date, consumption].
    lag_features : list[int]
        Lag days, e.g. [1, 7, 14, 30].
    rolling_windows : list[int]
        Window sizes for rolling mean/std, e.g. [7, 14, 30].
    household_means : Series | None
        Series mapping household_id -> mean consumption computed on 2023 only.
    cluster_series : Series | None
        Series mapping household_id -> fixed cluster label.
    cluster_dow_profile : DataFrame | None
        Cluster average consumption by day of week, computed from 2023 only.
    cluster_month_profile : DataFrame | None
        Cluster average consumption by month, computed from 2023 only.

    Returns
    -------
    DataFrame with original columns plus engineered feature columns.
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
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Cyclic calendar encoding
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366.0)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366.0)
    df["week_of_year_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 53.0)
    df["week_of_year_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 53.0)

    # Household mean from 2023 training data only
    if household_means is None:
        household_means = long_df.groupby("household_id")["consumption"].mean()

    df["household_mean"] = df["household_id"].map(household_means)

    # Cluster-derived priors from 2023 only
    if cluster_series is not None:
        cluster_numeric = pd.to_numeric(cluster_series, errors="coerce")
        df["cluster"] = df["household_id"].map(cluster_numeric)
    else:
        df["cluster"] = np.nan

    if cluster_dow_profile is not None and not cluster_dow_profile.empty:
        dow_lookup = cluster_dow_profile.reset_index().melt(
            id_vars="cluster",
            var_name="cluster_dow_col",
            value_name="cluster_dow_mean",
        )
        dow_lookup["day_of_week"] = (
            dow_lookup["cluster_dow_col"].str.replace("cluster_dow_", "", regex=False).astype(int)
        )
        dow_lookup = dow_lookup[["cluster", "day_of_week", "cluster_dow_mean"]]

        df = df.merge(
            dow_lookup,
            on=["cluster", "day_of_week"],
            how="left",
        )
    else:
        df["cluster_dow_mean"] = np.nan

    if cluster_month_profile is not None and not cluster_month_profile.empty:
        month_lookup = cluster_month_profile.reset_index().melt(
            id_vars="cluster",
            var_name="cluster_month_col",
            value_name="cluster_month_mean",
        )
        month_lookup["month"] = (
            month_lookup["cluster_month_col"].str.replace("cluster_month_", "", regex=False).astype(int)
        )
        month_lookup = month_lookup[["cluster", "month", "cluster_month_mean"]]

        df = df.merge(
            month_lookup,
            on=["cluster", "month"],
            how="left",
        )
    else:
        df["cluster_month_mean"] = np.nan

    # Relative-to-prior features
    df["household_minus_cluster_dow_mean"] = df["household_mean"] - df["cluster_dow_mean"]
    df["household_minus_cluster_month_mean"] = df["household_mean"] - df["cluster_month_mean"]

    return df


def prepare_features(
    train_wide: pd.DataFrame,
    regular_ids,
    lag_features: list,
    rolling_windows: list,
    cluster_series: Optional[pd.Series] = None,
) -> tuple:
    """Prepare 2023 training features only, with no 2024 input at all.

    Parameters
    ----------
    train_wide : DataFrame
        Wide 2023 DataFrame (households × 365 dates).
    regular_ids : Index
        Index of regular household IDs (non-degenerate).
    lag_features : list
        Lag days.
    rolling_windows : list
        Rolling window sizes.
    cluster_series : Series | None
        Optional household_id -> cluster mapping from fixed clustering output.

    Returns
    -------
    train_features_clean : DataFrame
        2023 feature rows with NaN lag rows dropped.
    feature_cols : list
        List of feature column names.
    household_means : Series
        Per-household 2023 means.
    cluster_dow_profile : DataFrame
        Cluster average day-of-week profile from 2023 only.
    cluster_month_profile : DataFrame
        Cluster average month profile from 2023 only.
    """
    reg_in_train = regular_ids.intersection(train_wide.index)
    train_reg = train_wide.loc[reg_in_train]

    # Household means computed exclusively from 2023 training data
    household_means = train_wide.mean(axis=1)

    if cluster_series is not None:
        cluster_dow_profile, cluster_month_profile = build_cluster_profile_features(
            train_wide=train_wide,
            cluster_series=cluster_series,
        )
    else:
        cluster_dow_profile = pd.DataFrame()
        cluster_month_profile = pd.DataFrame()

    train_long = wide_to_long(train_reg)

    train_features = build_features(
        train_long,
        lag_features=lag_features,
        rolling_windows=rolling_windows,
        household_means=household_means,
        cluster_series=cluster_series,
        cluster_dow_profile=cluster_dow_profile,
        cluster_month_profile=cluster_month_profile,
    )

    feature_cols = [
        c for c in train_features.columns
        if c not in ["household_id", "date", "consumption", "cluster"]
    ]

    # Drop rows where lag features are unavailable
    lag_cols = [f"lag_{l}" for l in lag_features]
    train_features_clean = train_features.dropna(subset=lag_cols).copy()

    return (
        train_features_clean,
        feature_cols,
        household_means,
        cluster_dow_profile,
        cluster_month_profile,
    )