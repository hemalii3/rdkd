"""EDA analysis functions for time series data.

This module provides helper functions for exploratory data analysis of
household energy consumption time series, with special focus on
clustering-oriented diagnostics for Phase 1 of the project.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from src.preprocessing.data_cleaning import DataCleaner, z_normalize_rows
from src.utils.config import (
    RANDOM_SEED,
    ZERO_HEAVY_THRESHOLD_PCT,
    NEAR_CONSTANT_STD_THRESHOLD,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Existing descriptive helpers
# =============================================================================

def compute_household_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for each household.

    Args:
        data: DataFrame with households as rows, days as columns.

    Returns:
        DataFrame with household-level descriptive statistics.
    """
    stats_df = pd.DataFrame(index=data.index)

    stats_df["mean"] = data.mean(axis=1)
    stats_df["median"] = data.median(axis=1)
    stats_df["std"] = data.std(axis=1)
    stats_df["min"] = data.min(axis=1)
    stats_df["max"] = data.max(axis=1)
    stats_df["range"] = stats_df["max"] - stats_df["min"]
    stats_df["q25"] = data.quantile(0.25, axis=1)
    stats_df["q75"] = data.quantile(0.75, axis=1)
    
    stats_df["cv"] = np.where(
        stats_df["mean"].abs() < 1e-8,
        np.nan,
        stats_df["std"] / stats_df["mean"].abs(),
    )
    
    # Skewness and kurtosis
    stats_df["skew"] = data.apply(
        lambda x: float(stats.skew(np.asarray(x, dtype=float))),
        axis=1,
    )
    stats_df["kurtosis"] = data.apply(
        lambda x: float(stats.kurtosis(np.asarray(x, dtype=float))),
        axis=1,
    )

    stats_df["zero_days"] = (data == 0).sum(axis=1)
    stats_df["zero_pct"] = (stats_df["zero_days"] / data.shape[1]) * 100

    logger.info("Computed descriptive statistics for %d households", len(stats_df))
    return stats_df


def detect_outlier_households(
    stats_df: pd.DataFrame, 
    method: str = 'iqr',
    threshold: float = 3.0
) -> Tuple[List, List]:
    if method == 'iqr':
        Q1 = stats_df['mean'].quantile(0.25)
        Q3 = stats_df['mean'].quantile(0.75)
        IQR = Q3 - Q1
        
        high_outliers = stats_df[stats_df['mean'] > Q3 + threshold * IQR].index.tolist()
        low_outliers = stats_df[stats_df['mean'] < Q1 - threshold * IQR].index.tolist()
    
    elif method == 'zscore':
        mean_series = pd.to_numeric(stats_df['mean'], errors='coerce')
        mean_values = mean_series.to_numpy(dtype=float)
        mean_center = np.nanmean(mean_values)
        std_scale = np.nanstd(mean_values)
        if not np.isfinite(std_scale) or std_scale < 1e-12:
            z_scores = np.zeros_like(mean_values, dtype=float)
        else:
            z_scores = np.abs((mean_values - mean_center) / std_scale)
        z_scores = np.nan_to_num(z_scores, nan=0.0)
        z_scores = np.asarray(z_scores, dtype=float)

        median_consumption = float(mean_series.median())

        high_mask = (z_scores > threshold) & (mean_values > median_consumption)
        low_mask = (z_scores > threshold) & (mean_values < median_consumption)

        high_outliers = stats_df.index[high_mask].tolist()
        low_outliers = stats_df.index[low_mask].tolist()
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")
    
    logger.info(
        "Detected %d high outliers and %d low outliers",
        len(high_outliers),
        len(low_outliers),
    )
    return high_outliers, low_outliers


def aggregate_by_day_of_week(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate consumption by day of week."""
    dates = pd.to_datetime(data.columns)
    day_idx = dates.dayofweek.to_numpy()

    result = pd.DataFrame(index=data.index)
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for i, name in enumerate(day_names):
        cols = data.columns[day_idx == i]
        result[name] = data[cols].mean(axis=1) if len(cols) else np.nan

    return result


def aggregate_by_month(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate consumption by month for each household."""
    dates = pd.to_datetime(data.columns)
    
    month_data = {}
    for month_num in range(1, 13):
        month_cols = [col for col, date in zip(data.columns, dates) if date.month == month_num]
        if month_cols:  
            month_data[month_num] = data[month_cols].mean(axis=1)
    
    month_df = pd.DataFrame(month_data, index=data.index)
    month_name_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    month_df = month_df.rename(columns=month_name_map)
    return month_df


def compute_autocorrelation(
    data: pd.DataFrame,
    household_id: Any,
    lags: Sequence[int] = (1, 7, 30)
) -> Dict[int, float]:
    """Compute autocorrelation for specific lags for a household."""
    series = np.asarray(data.loc[household_id].to_numpy(dtype=float), dtype=float)
    acf_values: Dict[int, float] = {}

    for lag in lags:
        if lag >= len(series):
            acf_values[lag] = float("nan")
            continue

        x1 = np.asarray(series[:-lag], dtype=float)
        x2 = np.asarray(series[lag:], dtype=float)

        if np.std(x1) < 1e-10 or np.std(x2) < 1e-10:
            acf_values[lag] = float("nan")
        else:
            acf_values[lag] = float(np.corrcoef(x1, x2)[0, 1])

    return acf_values


def categorize_households(stats_df: pd.DataFrame) -> pd.Series:
    """Create simple descriptive household categories for EDA only.

    These are heuristic labels for visualization and should not be treated
    as clustering outputs or as evidence for the final k selection.
    """
    # Force numeric typing for compared columns
    mean_col = pd.to_numeric(stats_df["mean"], errors="coerce")
    cv_col = pd.to_numeric(stats_df["cv"], errors="coerce")
    zero_pct_col = pd.to_numeric(stats_df["zero_pct"], errors="coerce")

    categories = pd.Series(index=stats_df.index, dtype="string")

    # Define thresholds based on quantiles
    low_threshold = float(mean_col.quantile(0.33))
    high_threshold = float(mean_col.quantile(0.67))
    high_cv_threshold = float(cv_col.quantile(0.75))
    high_zero_threshold = ZERO_HEAVY_THRESHOLD_PCT

    for hh_id in stats_df.index:
        mean_consumption = float(mean_col.loc[hh_id])
        cv = float(cv_col.loc[hh_id])
        zero_pct = float(zero_pct_col.loc[hh_id])

        if pd.isna(mean_consumption) or pd.isna(cv) or pd.isna(zero_pct):
            categories.loc[hh_id] = "Unclassified"
            continue

        if zero_pct >= high_zero_threshold:
            categories.loc[hh_id] = "Intermittent/Vacant"
        elif cv > high_cv_threshold:
            if mean_consumption > high_threshold:
                categories.loc[hh_id] = "High Variable"
            elif mean_consumption < low_threshold:
                categories.loc[hh_id] = "Low Variable"
            else:
                categories.loc[hh_id] = "Medium Variable"
        else:
            if mean_consumption > high_threshold:
                categories.loc[hh_id] = "High Stable"
            elif mean_consumption < low_threshold:
                categories.loc[hh_id] = "Low Stable"
            else:
                categories.loc[hh_id] = "Medium Stable"

    logger.info(f"Categorized households: {categories.value_counts().to_dict()}")
    return categories


def detect_normalized_trend(data: pd.DataFrame, household_id: int) -> Dict[str, float]:
    """Detect trend after z-normalizing the household time series."""
    row = data.loc[household_id]
    if isinstance(row, pd.DataFrame):  # duplicate index safety
        row = row.iloc[0]

    series = np.asarray(row.to_numpy(dtype=np.float64), dtype=np.float64).ravel()
    series_std = float(np.std(series))

    if series_std < 1e-10:
        return {
            "trend_slope_normalized": 0.0,
            "r_squared": 0.0,
            "p_value": 1.0,
        }

    series = (series - float(np.mean(series))) / (series_std + 1e-10)
    x = np.arange(series.size, dtype=np.float64)

    slope, _intercept = np.polyfit(x, series, 1)

    if np.std(x) < 1e-12 or np.std(series) < 1e-12:
        r_value = 0.0
    else:
        r_value = float(np.corrcoef(x, series)[0, 1])

    n = int(series.size)
    if n > 2 and abs(r_value) < 1.0:
        t_stat = r_value * np.sqrt((n - 2) / max(1e-12, 1.0 - r_value ** 2))
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 2))
    else:
        p_value = 1.0

    return {
        "trend_slope_normalized": float(slope),
        "r_squared": float(r_value ** 2),
        "p_value": p_value,
    }

def summarize_weekly_pattern_strength(data: pd.DataFrame) -> pd.Series:
    """Summarize weekly seasonality strength using lag-7 autocorrelation."""
    values = data.to_numpy(dtype=float)
    result = []

    for row in values:
        if len(row) <= 7:
            result.append(float("nan"))
            continue

        x1 = np.asarray(row[:-7], dtype=float)
        x2 = np.asarray(row[7:], dtype=float)

        if np.std(x1) < 1e-10 or np.std(x2) < 1e-10:
            result.append(float("nan"))
        else:
            result.append(float(np.corrcoef(x1, x2)[0, 1]))

    return pd.Series(result, index=data.index, name="autocorr_lag_7")

def summarize_monthly_variation_strength(data: pd.DataFrame) -> pd.Series:
    """Summarize annual seasonality strength using variation across monthly means."""
    monthly = aggregate_by_month(data)
    return monthly.std(axis=1).rename("monthly_profile_std")

# =============================================================================

def compute_peak_month(data: pd.DataFrame) -> pd.Series:
    """Return the peak month (1-12) based on highest average monthly consumption."""
    monthly = aggregate_by_month(data)

    month_num_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }

    peak_month_name = monthly.idxmax(axis=1)
    peak_month = peak_month_name.map(month_num_map).astype("Int64")
    peak_month.name = "peak_month"
    return peak_month


def build_household_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Build one diagnostic row per household for clustering-oriented EDA.

    Includes scale, variability, zero behaviour, temporal structure,
    seasonality summaries, and simple degeneracy flags.
    """
    diagnostics = compute_household_statistics(data).copy()

    # Autocorrelation summaries
    acf_1 = {}
    acf_30 = {}
    trend_slope_normalized = {}

    for household_id in data.index:
        acf = compute_autocorrelation(data, household_id, lags=(1, 7, 30))
        acf_1[household_id] = acf[1]
        acf_30[household_id] = acf[30]

        trend = detect_normalized_trend(data, household_id)
        trend_slope_normalized[household_id] = trend["trend_slope_normalized"]

    diagnostics["autocorr_lag_1"] = pd.Series(acf_1)
    diagnostics["autocorr_lag_7"] = summarize_weekly_pattern_strength(data)
    diagnostics["autocorr_lag_30"] = pd.Series(acf_30)
    diagnostics["monthly_profile_std"] = summarize_monthly_variation_strength(data)
    diagnostics["trend_slope_normalized"] = pd.Series(trend_slope_normalized)

    # Seasonal summaries
    dates = pd.to_datetime(data.columns)
    winter_cols = [col for col, dt in zip(data.columns, dates) if dt.month in [12, 1, 2]]
    summer_cols = [col for col, dt in zip(data.columns, dates) if dt.month in [6, 7, 8]]

    diagnostics["winter_mean"] = data[winter_cols].mean(axis=1) if winter_cols else 0.0
    diagnostics["summer_mean"] = data[summer_cols].mean(axis=1) if summer_cols else 0.0
    diagnostics["winter_summer_contrast"] = (
        (diagnostics["winter_mean"] - diagnostics["summer_mean"]) /
        (diagnostics["winter_mean"] + diagnostics["summer_mean"] + 1e-6)
    )

    diagnostics["peak_month_eda"] = compute_peak_month(data)

    cleaner = DataCleaner(random_seed=RANDOM_SEED)
    flags = cleaner.flag_degenerate_households(data)

    diagnostics["all_zero_flag"] = flags["all_zero_flag"]
    diagnostics["near_constant_flag"] = flags["near_constant_flag"]
    diagnostics["zero_heavy_flag"] = flags["zero_heavy_flag"]
    diagnostics["regime_label"] = flags["regime_label"]
    diagnostics["degenerate_flag"] = flags["degenerate_flag"]
    
    logger.info("Built household diagnostics for %d households", len(diagnostics))
    return diagnostics


def split_households_for_clustering(
    data: pd.DataFrame,
) -> Tuple[pd.Index, pd.Index]:
    """Split households into main-clustering and special-regime subsets."""
    cleaner = DataCleaner(random_seed=RANDOM_SEED)
    _, _, flags = cleaner.split_regular_and_degenerate_households(data)
    main_ids = flags.index[~flags["excluded_from_main_fit"]]
    special_ids = flags.index[flags["excluded_from_main_fit"]]
    return main_ids, special_ids


def summarize_phase1_findings(data: pd.DataFrame) -> Dict[str, object]:
    """Produce a compact summary of clustering-relevant findings."""
    diagnostics = build_household_diagnostics(data)

    summary = {
        "n_households": int(len(diagnostics)),
        "n_degenerate": int(diagnostics["degenerate_flag"].sum()),
        "degenerate_breakdown_exclusive": (
            diagnostics.loc[diagnostics["degenerate_flag"], "regime_label"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "mean_household_consumption_median": float(diagnostics["mean"].median()),
        "mean_household_consumption_iqr": [
            float(diagnostics["mean"].quantile(0.25)),
            float(diagnostics["mean"].quantile(0.75)),
        ],
        "weekly_acf_median": float(diagnostics["autocorr_lag_7"].median()),
        "peak_month_mode": (
            diagnostics["peak_month_eda"].mode().iloc[0]
            if not diagnostics["peak_month_eda"].mode().empty else None
        ),
        "n_all_zero": int((diagnostics["regime_label"] == "all_zero").sum()),
        "n_near_constant": int((diagnostics["regime_label"] == "near_constant").sum()),
        "n_zero_heavy": int((diagnostics["regime_label"] == "zero_heavy").sum()),
    }

    logger.info(
        "Phase 1 summary: households=%d, degenerate=%d, breakdown=%s",
        summary["n_households"],
        summary["n_degenerate"],
        summary["degenerate_breakdown_exclusive"],
    )
    return summary


def sample_households_by_regime(
    diagnostics: pd.DataFrame,
    n_per_group: int = 5,
    random_state: int = RANDOM_SEED,
) -> Dict[str, List[Any]]:
    """Sample representative household IDs from different descriptive regimes."""
    rng = np.random.default_rng(random_state)
    sampled: Dict[str, List[Any]] = {}

    candidate_groups = {
        "all_zero": diagnostics.index[diagnostics["all_zero_flag"]].tolist(),
        "near_constant": diagnostics.index[
            diagnostics["near_constant_flag"] & ~diagnostics["all_zero_flag"]
        ].tolist(),
        "zero_heavy": diagnostics.index[
            diagnostics["zero_heavy_flag"]
            & ~diagnostics["all_zero_flag"]
            & ~diagnostics["near_constant_flag"]
        ].tolist(),
        "high_mean": diagnostics.index[
            diagnostics["mean"] >= diagnostics["mean"].quantile(0.90)
        ].tolist(),
        "high_weekly_structure": diagnostics.index[
            diagnostics["autocorr_lag_7"] >= diagnostics["autocorr_lag_7"].quantile(0.90)
        ].tolist(),
        "high_annual_variation": diagnostics.index[
            diagnostics["monthly_profile_std"] >= diagnostics["monthly_profile_std"].quantile(0.90)
        ].tolist(),
    }

    for group_name, members in candidate_groups.items():
        if len(members) == 0:
            sampled[group_name] = []
            continue

        sample_size = min(n_per_group, len(members))
        chosen = rng.choice(members, size=sample_size, replace=False)
        sampled[group_name] = chosen.tolist()

    return sampled


def build_phase1_clustering_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Main Phase 1 diagnostic table used before running clustering."""
    return build_household_diagnostics(data)

def compute_rowwise_norm_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize row-wise normalized profile behaviour for shape-based methods.

    These summaries are only for method-selection diagnostics, not the final
    feature matrix for feature-based clustering.
    """
    normalized = z_normalize_rows(data)
    summary = pd.DataFrame(index=data.index)
    summary["normalized_range"] = normalized.max(axis=1) - normalized.min(axis=1)
    summary["normalized_abs_diff_mean"] = normalized.diff(axis=1).abs().mean(axis=1)
    summary["normalized_autocorr_lag_7"] = summarize_weekly_pattern_strength(normalized)
    return summary

def build_method_selection_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Build a compact summary to justify clustering-method choice."""
    diagnostics = build_phase1_clustering_summary(data)
    norm_summary = compute_rowwise_norm_summary(data)
    output = diagnostics.join(norm_summary, how="left")
    return output


if __name__ == "__main__":
    from src.utils.data_loader import load_train_data

    print("Testing EDA analysis functions...")
    data = load_train_data()

    print("\n1. Computing descriptive statistics...")
    stats_df = compute_household_statistics(data)
    print(stats_df.head())

    print("\n2. Building diagnostics...")
    diagnostics = build_phase1_clustering_summary(data)
    print(diagnostics.head())

    print("\n3. Phase 1 summary...")
    summary = summarize_phase1_findings(data)
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}: <dict with {len(value)} entries>")
        else:
            print(f"{key}: {value}")

    print("\n4. Sampled households by regime...")
    sampled = sample_households_by_regime(diagnostics)
    print(sampled)

    print("\n EDA analysis module checks passed!")