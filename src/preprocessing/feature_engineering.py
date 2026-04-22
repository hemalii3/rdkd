"""Feature engineering module for feature-based household clustering.

This module is intentionally focused on Method 1 of the project:
feature-based clustering followed by Ward hierarchical clustering.

It extracts a compact, interpretable set of household-level features from
the 2023 daily energy consumption panel. k-Shape and DTW-based methods
should operate on sequence data directly and should not use this module
as their main input pipeline.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple, TypedDict
import numpy as np
import pandas as pd

from src.preprocessing.data_cleaning import z_normalize_rows
from src.utils.config import FEATURE_INCLUDE_NORMALIZED_SHAPE_BLOCK

logger = logging.getLogger(__name__)


class FeatureMetadata(TypedDict):
    """Metadata returned with the compact feature set."""

    feature_names: List[str]
    feature_blocks: Dict[str, List[str]]
    n_features: int
    notes: Dict[str, str]


# =============================================================================
# Internal helpers
# =============================================================================

def _safe_autocorr(series: np.ndarray, lag: int) -> float:
    """Compute lagged autocorrelation safely for a 1D numeric series."""
    if lag <= 0 or lag >= len(series):
        return float("nan")

    x1 = np.asarray(series[:-lag], dtype=float)
    x2 = np.asarray(series[lag:], dtype=float)

    if np.std(x1) < 1e-10 or np.std(x2) < 1e-10:
        return float("nan")

    return float(np.corrcoef(x1, x2)[0, 1])


def _safe_linear_slope(y: np.ndarray) -> float:
    """Compute linear trend slope safely."""
    y = np.asarray(y, dtype=float)
    if np.std(y) < 1e-10:
        return 0.0

    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def _safe_normalized_slope(y: np.ndarray) -> float:
    """Compute slope after z-normalizing the input series."""
    y = np.asarray(y, dtype=float)
    std = float(np.std(y))
    if std < 1e-10:
        return 0.0
    y_norm = (y - float(np.mean(y))) / (std + 1e-10)
    return _safe_linear_slope(y_norm)


def _monthly_mean_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Return monthly household means with Jan-Dec columns."""
    dates = pd.to_datetime(data.columns)
    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }

    monthly = pd.DataFrame(index=data.index)
    for month_num, month_name in month_map.items():
        cols = [col for col, dt in zip(data.columns, dates) if dt.month == month_num]
        monthly[month_name] = data[cols].mean(axis=1) if cols else np.nan

    return monthly


def _dayofweek_mean_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Return mean consumption by day of week for each household."""
    dates = pd.to_datetime(data.columns)
    day_name_map = {
        0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu",
        4: "Fri", 5: "Sat", 6: "Sun",
    }

    dow = pd.DataFrame(index=data.index)
    for day_num, day_name in day_name_map.items():
        cols = [col for col, dt in zip(data.columns, dates) if dt.dayofweek == day_num]
        dow[day_name] = data[cols].mean(axis=1) if cols else np.nan

    return dow


# =============================================================================
# Feature engineering class
# =============================================================================

class FeatureEngineer:
    """Feature extractor for Method 1: feature-based clustering."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.feature_names: List[str] = []
        self.feature_blocks: Dict[str, List[str]] = {}

    # -------------------------------------------------------------------------
    # Feature blocks
    # -------------------------------------------------------------------------

    def extract_level_dispersion_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract level and dispersion features.

        These features capture household scale, variability, and sparsity.
        """
        features = pd.DataFrame(index=data.index)

        features["mean"] = data.mean(axis=1)
        features["std"] = data.std(axis=1)

        features["cv"] = np.where(
            features["mean"].abs() < 1e-8,
            np.nan,
            features["std"] / features["mean"].abs(),
        )

        features["zero_pct"] = ((data == 0).sum(axis=1) / data.shape[1]) * 100

        logger.info(
            "Extracted %d level/dispersion features",
            len(features.columns),
        )
        return features

    def extract_weekly_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract weekly-structure features.

        Includes mean consumption by weekday, weekday/weekend contrast,
        and lag-7 autocorrelation as a measure of weekly persistence.
        """
        features = pd.DataFrame(index=data.index)

        dow = _dayofweek_mean_matrix(data)
        dow = dow.rename(columns={col: f"dow_{col}" for col in dow.columns})
        features = pd.concat([features, dow], axis=1)

        weekday_mean = dow[[f"dow_{d}" for d in ["Mon", "Tue", "Wed", "Thu", "Fri"]]].mean(axis=1)
        weekend_mean = dow[[f"dow_{d}" for d in ["Sat", "Sun"]]].mean(axis=1)

        features["weekend_weekday_contrast"] = (
            (weekend_mean - weekday_mean) /
            (weekend_mean + weekday_mean + 1e-6)
        )

        features["weekday_range"] = dow.max(axis=1) - dow.min(axis=1)

        weekly_total = dow.sum(axis=1) + 1e-6
        features["weekend_share"] = (
            dow[["dow_Sat", "dow_Sun"]].sum(axis=1) / weekly_total
        )

        features["weekday_std"] = dow.std(axis=1)

        features["autocorr_lag_7"] = data.apply(
            lambda row: _safe_autocorr(np.asarray(row.values, dtype=float), lag=7),
            axis=1,
        )

        logger.info("Extracted %d weekly features", len(features.columns))
        return features

    
    def extract_energy_profile_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract energy-domain profile features for customer grouping.

        These features summarize base load, seasonal concentration, peak timing,
        and burstiness in a scale-aware way.
        """
        features = pd.DataFrame(index=data.index)

        monthly = _monthly_mean_matrix(data)
        annual_mean = data.mean(axis=1) + 1e-6

        # Base load vs seasonal load
        monthly_min = monthly.min(axis=1)
        monthly_max = monthly.max(axis=1)

        features["base_load_ratio"] = monthly_min / annual_mean
        features["seasonal_peak_ratio"] = monthly_max / annual_mean
        features["seasonal_range_ratio"] = (monthly_max - monthly_min) / annual_mean

        # Seasonal energy shares
        dates = pd.to_datetime(data.columns)
        winter_cols = [col for col, dt in zip(data.columns, dates) if dt.month in [12, 1, 2]]
        summer_cols = [col for col, dt in zip(data.columns, dates) if dt.month in [6, 7, 8]]

        annual_total = data.sum(axis=1) + 1e-6
        winter_total = data[winter_cols].sum(axis=1) if winter_cols else pd.Series(0.0, index=data.index)
        summer_total = data[summer_cols].sum(axis=1) if summer_cols else pd.Series(0.0, index=data.index)

        features["winter_share"] = winter_total / annual_total
        features["summer_share"] = summer_total / annual_total

        # Concentration of monthly demand
        monthly_shares = monthly.div(monthly.sum(axis=1) + 1e-6, axis=0)
        top3_share = np.sort(monthly_shares.to_numpy(dtype=float), axis=1)[:, -3:].sum(axis=1)
        features["top3_month_share"] = top3_share

        monthly_entropy = -(monthly_shares * np.log(monthly_shares + 1e-10)).sum(axis=1)
        features["monthly_entropy"] = monthly_entropy

        # Peak / trough timing (cyclic encoding)
        peak_month_idx = monthly.to_numpy(dtype=float).argmax(axis=1) + 1
        trough_month_idx = monthly.to_numpy(dtype=float).argmin(axis=1) + 1

        features["peak_month_sin"] = np.sin(2 * np.pi * peak_month_idx / 12.0)
        features["peak_month_cos"] = np.cos(2 * np.pi * peak_month_idx / 12.0)
        features["trough_month_sin"] = np.sin(2 * np.pi * trough_month_idx / 12.0)
        features["trough_month_cos"] = np.cos(2 * np.pi * trough_month_idx / 12.0)

        logger.info("Extracted %d energy-profile features", len(features.columns))
        return features
    
    def extract_normalized_shape_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Shape-focused features derived from row-wise z-normalized series.
        These features are intended to capture annual and weekly behaviour
        without reintroducing raw consumption magnitude.
        """
        normalized_df = z_normalize_rows(data)

        norm_monthly = _monthly_mean_matrix(normalized_df).rename(
            columns=lambda c: f"norm_month_{c}"
        )
        norm_dow = _dayofweek_mean_matrix(normalized_df).rename(
            columns=lambda c: f"norm_dow_{c}"
        )

        features = pd.concat([norm_monthly, norm_dow], axis=1)

        # Existing summary features
        features["norm_monthly_profile_std"] = norm_monthly.std(axis=1)
        features["norm_monthly_profile_range"] = norm_monthly.max(axis=1) - norm_monthly.min(axis=1)

        features["norm_winter_mean"] = norm_monthly[
            ["norm_month_Dec", "norm_month_Jan", "norm_month_Feb"]
        ].mean(axis=1)
        features["norm_summer_mean"] = norm_monthly[
            ["norm_month_Jun", "norm_month_Jul", "norm_month_Aug"]
        ].mean(axis=1)
        features["norm_winter_summer_diff"] = (
            features["norm_winter_mean"] - features["norm_summer_mean"]
        )

        features["norm_weekday_mean"] = norm_dow[
            ["norm_dow_Mon", "norm_dow_Tue", "norm_dow_Wed", "norm_dow_Thu", "norm_dow_Fri"]
        ].mean(axis=1)
        features["norm_weekend_mean"] = norm_dow[
            ["norm_dow_Sat", "norm_dow_Sun"]
        ].mean(axis=1)
        features["norm_weekend_weekday_diff"] = (
            features["norm_weekend_mean"] - features["norm_weekday_mean"]
        )

        # quarterly normalized seasonality
        features["norm_q1_mean"] = norm_monthly[
            ["norm_month_Jan", "norm_month_Feb", "norm_month_Mar"]
        ].mean(axis=1)
        features["norm_q2_mean"] = norm_monthly[
            ["norm_month_Apr", "norm_month_May", "norm_month_Jun"]
        ].mean(axis=1)
        features["norm_q3_mean"] = norm_monthly[
            ["norm_month_Jul", "norm_month_Aug", "norm_month_Sep"]
        ].mean(axis=1)
        features["norm_q4_mean"] = norm_monthly[
            ["norm_month_Oct", "norm_month_Nov", "norm_month_Dec"]
        ].mean(axis=1)

        # seasonal transition features
        features["norm_spring_minus_winter"] = features["norm_q2_mean"] - features["norm_q1_mean"]
        features["norm_summer_minus_spring"] = features["norm_q3_mean"] - features["norm_q2_mean"]
        features["norm_autumn_minus_summer"] = features["norm_q4_mean"] - features["norm_q3_mean"]
        features["norm_winter_minus_autumn"] = features["norm_q1_mean"] - features["norm_q4_mean"]

        # normalized peak / trough timing (cyclic)
        peak_month_idx = norm_monthly.to_numpy(dtype=float).argmax(axis=1) + 1
        trough_month_idx = norm_monthly.to_numpy(dtype=float).argmin(axis=1) + 1

        features["norm_peak_month_sin"] = np.sin(2 * np.pi * peak_month_idx / 12.0)
        features["norm_peak_month_cos"] = np.cos(2 * np.pi * peak_month_idx / 12.0)
        features["norm_trough_month_sin"] = np.sin(2 * np.pi * trough_month_idx / 12.0)
        features["norm_trough_month_cos"] = np.cos(2 * np.pi * trough_month_idx / 12.0)

        # concentration / smoothness of normalized annual shape
        abs_norm_monthly = norm_monthly.abs()  # keep pandas DataFrame type
        row_sums = abs_norm_monthly.sum(axis=1).replace(0, np.nan)
        norm_month_probs = abs_norm_monthly.div(row_sums, axis=0).fillna(0.0)

        log_probs = norm_month_probs.clip(lower=1e-10).apply(np.log)
        features["norm_month_entropy"] = -(norm_month_probs * log_probs).sum(axis=1)

        norm_month_diff = norm_monthly.diff(axis=1)
        features["norm_month_diff_abs_mean"] = norm_month_diff.abs().mean(axis=1)
        features["norm_month_diff_std"] = norm_month_diff.std(axis=1)

        # normalized weekly roughness
        features["norm_dow_range"] = norm_dow.max(axis=1) - norm_dow.min(axis=1)
        norm_dow_diff = norm_dow.diff(axis=1)
        features["norm_dow_diff_abs_mean"] = norm_dow_diff.abs().mean(axis=1)

        logger.info("Extracted %d normalized-shape features", len(features.columns))
        return features

    def extract_persistence_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract persistence and volatility features.

        These features summarize short-term dependence and roughness.
        """
        features = pd.DataFrame(index=data.index)

        features["autocorr_lag_1"] = data.apply(
            lambda row: _safe_autocorr(np.asarray(row.values, dtype=float), lag=1),
            axis=1,
        )
        features["autocorr_lag_30"] = data.apply(
            lambda row: _safe_autocorr(np.asarray(row.values, dtype=float), lag=30),
            axis=1,
        )

        rolling_std_7 = data.T.rolling(window=7).std().T.mean(axis=1)
        rolling_std_30 = data.T.rolling(window=30).std().T.mean(axis=1)
        features["rolling_std_7"] = rolling_std_7
        features["rolling_std_30"] = rolling_std_30

        diffs = data.diff(axis=1)
        features["diff_mean"] = diffs.mean(axis=1)
        features["diff_std"] = diffs.std(axis=1)
        features["abs_diff_mean"] = diffs.abs().mean(axis=1)
        features["diff_max"] = diffs.max(axis=1)
        features["diff_min"] = diffs.min(axis=1)

        q50 = data.quantile(0.50, axis=1) + 1e-6
        q95 = data.quantile(0.95, axis=1)
        q99 = data.quantile(0.99, axis=1)
        max_val = data.max(axis=1)

        features["p95_to_median_ratio"] = q95 / q50
        features["p99_to_median_ratio"] = q99 / q50
        features["max_to_p95_ratio"] = max_val / (q95 + 1e-6)

        dates = pd.to_datetime(data.columns)
        winter_cols = [col for col, dt in zip(data.columns, dates) if dt.month in [12, 1, 2]]
        summer_cols = [col for col, dt in zip(data.columns, dates) if dt.month in [6, 7, 8]]

        annual_mean = data.mean(axis=1) + 1e-6

        if winter_cols:
            features["winter_std_norm"] = data[winter_cols].std(axis=1) / annual_mean
        else:
            features["winter_std_norm"] = 0.0

        if summer_cols:
            features["summer_std_norm"] = data[summer_cols].std(axis=1) / annual_mean
        else:
            features["summer_std_norm"] = 0.0

        features["winter_summer_volatility_ratio"] = (
            features["winter_std_norm"] / (features["summer_std_norm"] + 1e-6)
        )

        logger.info(
            "Extracted %d persistence/volatility features",
            len(features.columns),
        )
        return features
    
    

    def extract_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract simple trend features."""
        features = pd.DataFrame(index=data.index)

        features["trend_slope"] = data.apply(
            lambda row: _safe_linear_slope(np.asarray(row.values, dtype=float)),
            axis=1,
        )
        features["trend_slope_normalized"] = data.apply(
            lambda row: _safe_normalized_slope(np.asarray(row.values, dtype=float)),
            axis=1,
        )

        logger.info("Extracted %d trend features", len(features.columns))
        return features
    
    def _select_normalized_shape_block(self, shape_features: pd.DataFrame) -> pd.DataFrame:
        """Return the standardized normalized-shape feature subset used in Method 1."""
        selected_columns = [
            "norm_dow_Mon",
            "norm_dow_Tue",
            "norm_dow_Wed",
            "norm_dow_Thu",
            "norm_dow_Fri",
            "norm_dow_Sat",
            "norm_dow_Sun",

            "norm_month_Jan",
            "norm_month_Feb",
            "norm_month_Mar",
            "norm_month_Apr",
            "norm_month_May",
            "norm_month_Jun",
            "norm_month_Jul",
            "norm_month_Aug",
            "norm_month_Sep",
            "norm_month_Oct",
            "norm_month_Nov",
            "norm_month_Dec",

            "norm_monthly_profile_std",
            "norm_monthly_profile_range",
            "norm_winter_summer_diff",
            "norm_weekend_weekday_diff",
            "norm_month_entropy",
            "norm_dow_range",
        ]
        return shape_features[selected_columns].copy()

    # -------------------------------------------------------------------------
    # Main Method 1 pipeline
    # -------------------------------------------------------------------------

    def extract_final_feature_ward_features(
        self,
        data: pd.DataFrame,
        return_metadata: bool = False,
    ) -> pd.DataFrame | Tuple[pd.DataFrame, FeatureMetadata]:
        """Extract the final shape-first feature set for Ward clustering."""

        if not FEATURE_INCLUDE_NORMALIZED_SHAPE_BLOCK:
            raise ValueError(
                "extract_final_feature_ward_features() requires FEATURE_INCLUDE_NORMALIZED_SHAPE_BLOCK=True."
            )

        # Main behavioral block
        shape_normalized = self._select_normalized_shape_block(
            self.extract_normalized_shape_features(data)
        )

        # Weekly block: keep only shape-safe / dependence-oriented signals
        weekly_all = self.extract_weekly_features(data)
        weekly = pd.DataFrame(index=data.index)
        weekly["autocorr_lag_7"] = weekly_all["autocorr_lag_7"]

        # Persistence / roughness block
        persistence_all = self.extract_persistence_volatility_features(data)
        persistence = persistence_all[
            [
                "autocorr_lag_30",
                "diff_std",
            ]
        ].copy()

        # Trend block: normalized only
        trend_all = self.extract_trend_features(data)
        trend = trend_all[
            [
                "trend_slope_normalized",
            ]
        ].copy()

        energy_profile = pd.DataFrame(index=data.index)

        features = pd.concat(
            [
                weekly,
                shape_normalized,
                persistence,
                trend,
                #energy_profile,
            ],
            axis=1,
        )

        self.feature_names = features.columns.tolist()
        self.feature_blocks = {
            "weekly": weekly.columns.tolist(),
            "shape_normalized": shape_normalized.columns.tolist(),
            "persistence": persistence.columns.tolist(),
            "trend": trend.columns.tolist(),
            #"energy_profile": energy_profile.columns.tolist(),
        }

        notes = {
            "weekly": "Weekly dependence features that do not directly encode raw level.",
            "shape_normalized": "Main behavior block: normalized annual and weekly shape descriptors.",
            "persistence": "Roughness and short/medium-range dependence features.",
            "trend": "Normalized trend direction over the year.",
            #"energy_profile": "Small secondary raw-profile block kept for interpretability.",
        }

        metadata: FeatureMetadata = {
            "feature_names": self.feature_names,
            "feature_blocks": self.feature_blocks,
            "n_features": len(self.feature_names),
            "notes": notes,
        }

        logger.info(
            "Extracted final Ward feature set with %d features across %d blocks",
            len(self.feature_names),
            len(self.feature_blocks),
        )

        if return_metadata:
            return features, metadata
        return features
    # -------------------------------------------------------------------------
    # Interpretation helpers
    # -------------------------------------------------------------------------

    def get_feature_block_map(self) -> Dict[str, List[str]]:
        """Return the feature-block mapping from the latest extraction."""
        return self.feature_blocks

    def summarize_feature_groups(self) -> pd.DataFrame:
        """Summarize feature blocks from the latest extraction."""
        if not self.feature_blocks:
            raise ValueError(
                "No features have been extracted yet. "
            )

        rows = []
        for block_name, block_features in self.feature_blocks.items():
            rows.append(
                {
                    "block": block_name,
                    "n_features": len(block_features),
                    "features": ", ".join(block_features),
                }
            )

        return pd.DataFrame(rows)
