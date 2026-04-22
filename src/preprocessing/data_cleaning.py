"""Data cleaning and clustering-specific preparation utilities.

This module provides functions for handling missing values, outliers,
and method-specific preprocessing for clustering household energy
consumption time series.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.config import (
    KSHAPE_NORMALIZATION,
    MISSING_VALUE_STRATEGY,
    NEGATIVE_VALUE_ACTION,
    NEAR_CONSTANT_STD_THRESHOLD,
    ZERO_HEAVY_THRESHOLD_PCT,
    EXCLUDE_ZERO_HEAVY_FROM_MAIN_CLUSTERING,
)

logger = logging.getLogger(__name__)

def z_normalize_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Perform row-wise z-normalization for shape comparison."""
    values = data.to_numpy(dtype=float)
    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, keepdims=True)
    normalized = np.where(stds < 1e-10, 0.0, (values - means) / (stds + 1e-10))
    return pd.DataFrame(normalized, index=data.index, columns=data.columns)

class DataCleaner:
    """Comprehensive data cleaning pipeline for time series data."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def handle_missing_values(
        self, 
        data: pd.DataFrame, 
        strategy: str = MISSING_VALUE_STRATEGY
    ) -> Tuple[pd.DataFrame, Dict]:
        missing_before = data.isna().sum().sum()
        
        if missing_before == 0:
            logger.info("No missing values detected")
            return data.copy(), {'missing_before': 0, 'missing_after': 0, 'strategy': 'none'}
        
        data_clean = data.copy()
        
        if strategy == "forward_fill":
            data_clean = data_clean.ffill(axis=1).bfill(axis=1)

        elif strategy == "interpolate":
            data_clean = data_clean.interpolate(method="linear", axis=1).bfill(axis=1).ffill(axis=1)

        elif strategy == "mean":
            household_means = data_clean.mean(axis=1)
            for idx in data_clean.index:
                data_clean.loc[idx] = data_clean.loc[idx].fillna(household_means[idx])

        elif strategy == "drop":
            raise ValueError(
                "Row-dropping is not allowed in the clustering pipeline because "
                "household identity must be preserved across train/test evaluation."
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        missing_after = data_clean.isna().sum().sum()
        
        summary = {
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'strategy': strategy,
            'households_dropped': len(data) - len(data_clean) if strategy == 'drop' else 0
        }
        
        logger.info(
            "Missing values handled: %d -> %d using %s",
            missing_before,
            missing_after,
            strategy,
        )
        return data_clean, summary
    
    def handle_negatives(
        self,
        data: pd.DataFrame,
        action: str = NEGATIVE_VALUE_ACTION,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Handle negative consumption values."""
        negatives_before = (data < 0).sum().sum()
        data_clean = data.copy()
        
        if negatives_before == 0:
            return data_clean, {
                "negatives_before": 0,
                "negatives_after": 0,
                "action": "none",
            }

        if action == "set_zero":
            data_clean[data_clean < 0] = 0
        elif action == "abs":
            data_clean = data_clean.abs()
        elif action == "remove":
            data_clean[data_clean < 0] = np.nan
            data_clean = data_clean.interpolate(method="linear", axis=1).bfill(axis=1).ffill(axis=1)
        else:
            raise ValueError(f"Unknown negative action: {action}")

        negatives_after = int((data_clean < 0).sum().sum())

        summary = {
            "negatives_before": negatives_before,
            "negatives_after": negatives_after,
            "action": action,
        }

        logger.info(
            "Negative values handled: %d -> %d using %s",
            negatives_before,
            negatives_after,
            action,
        )
        return data_clean, summary
    
    def normalize_data(
        self,
        data: pd.DataFrame,
        method: str = "standard",
        axis: int = 1,
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """Normalize/standardize time series data."""
        values = data.to_numpy(dtype=float)

        if method == "standard":
            if axis == 1:
                normalized_df = z_normalize_rows(data)
                logger.info("Data normalized using %s method (axis=%d)", method, axis)
                return normalized_df, {"method": method, "axis": axis}
            else:
                means = values.mean(axis=0, keepdims=True)
                stds = values.std(axis=0, keepdims=True)
                normalized = (values - means) / (stds + 1e-10)

        elif method == "minmax":
            if axis == 1:
                mins = values.min(axis=1, keepdims=True)
                maxs = values.max(axis=1, keepdims=True)
                normalized = (values - mins) / (maxs - mins + 1e-10)
            else:
                mins = values.min(axis=0, keepdims=True)
                maxs = values.max(axis=0, keepdims=True)
                normalized = (values - mins) / (maxs - mins + 1e-10)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        normalized_df = pd.DataFrame(normalized, index=data.index, columns=data.columns)
        logger.info("Data normalized using %s method (axis=%d)", method, axis)
        return normalized_df, {"method": method, "axis": axis}

    def summarize_cleaning_changes(self, original: pd.DataFrame, cleaned: pd.DataFrame) -> Dict:
        """Summarize changes between original and cleaned data."""
        return {
            "shape_before": original.shape,
            "shape_after": cleaned.shape,
            "missing_before": int(original.isna().sum().sum()),
            "missing_after": int(cleaned.isna().sum().sum()),
            "negative_before": int((original < 0).sum().sum()),
            "negative_after": int((cleaned < 0).sum().sum()),
            "zero_before": int((original == 0).sum().sum()),
            "zero_after": int((cleaned == 0).sum().sum()),
        }
    
    # -------------------------------------------------------------------------

    def flag_degenerate_households(
        self,
        data: pd.DataFrame,
        zero_heavy_threshold_pct: float = ZERO_HEAVY_THRESHOLD_PCT,
        near_constant_std_threshold: float = NEAR_CONSTANT_STD_THRESHOLD,
    ) -> pd.DataFrame:
        """Flag household regimes that can distort clustering.

        Returns a DataFrame indexed by household ID.
        """
        row_sum_abs = data.abs().sum(axis=1)
        row_std = data.std(axis=1)
        zero_days = (data == 0).sum(axis=1)
        zero_pct = (zero_days / data.shape[1]) * 100

        flags = pd.DataFrame(index=data.index)
        flags["all_zero_flag"] = row_sum_abs <= 1e-12
        flags["near_constant_flag"] = (
            (row_std <= near_constant_std_threshold) &
            (~flags["all_zero_flag"])
        )
        flags["zero_heavy_flag"] = (
            (zero_pct >= zero_heavy_threshold_pct) &
            (~flags["all_zero_flag"]) &
            (~flags["near_constant_flag"])
        )
        flags["regime_label"] = "regular"
        flags.loc[flags["zero_heavy_flag"], "regime_label"] = "zero_heavy"
        flags.loc[flags["near_constant_flag"], "regime_label"] = "near_constant"
        flags.loc[flags["all_zero_flag"], "regime_label"] = "all_zero"
        flags["degenerate_flag"] = (
            flags["all_zero_flag"] |
            flags["near_constant_flag"] |
            flags["zero_heavy_flag"]
        )
        flags["std"] = row_std
        flags["zero_days"] = zero_days
        flags["zero_pct"] = zero_pct

        logger.info(
            "Degenerate household flags: all_zero=%d, near_constant=%d, zero_heavy=%d",
            int(flags["all_zero_flag"].sum()),
            int(flags["near_constant_flag"].sum()),
            int(flags["zero_heavy_flag"].sum()),
        )
        return flags
    
    def split_regular_and_degenerate_households(
        self,
        data: pd.DataFrame,
        zero_heavy_threshold_pct: float = ZERO_HEAVY_THRESHOLD_PCT,
        near_constant_std_threshold: float = NEAR_CONSTANT_STD_THRESHOLD,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into regular and degenerate households.

        Returns:
            regular_data: households to be used in the main clustering fit
            excluded_data: households to keep separate
            flags: household-level regime flags
        """
        flags = self.flag_degenerate_households(
            data,
            zero_heavy_threshold_pct=zero_heavy_threshold_pct,
            near_constant_std_threshold=near_constant_std_threshold,
        )

        exclude_mask = (
            flags["all_zero_flag"] |
            flags["near_constant_flag"] |
            (
                flags["zero_heavy_flag"] &
                EXCLUDE_ZERO_HEAVY_FROM_MAIN_CLUSTERING
            )
        )
        flags["excluded_from_main_fit"] = exclude_mask
        regular_mask = ~exclude_mask
        regular_data = data.loc[regular_mask].copy()
        excluded_data = data.loc[flags["excluded_from_main_fit"]].copy()

        logger.info(
            "Split households into regular=%d and excluded=%d",
            len(regular_data),
            len(excluded_data),
        )
        return regular_data, excluded_data, flags

    def prepare_regular_subset(
        self,
        data: pd.DataFrame,
        transform: str = "none",
        normalize_method: str | None = None,
        normalize_axis: int = 1,
        negative_action: str = NEGATIVE_VALUE_ACTION,
        missing_strategy: str = MISSING_VALUE_STRATEGY,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Prepare a clustering-ready subset while preserving the full cleaned panel.

        Steps:
        1. handle negative values on the full panel
        2. handle missing values on the full panel
        3. flag and split regular vs degenerate households
        4. optionally transform / normalize only the regular subset used for fitting

        Returns:
            prepared_regular:
                Regular-household subset after optional transform/normalization.
            cleaned_full:
                Fully cleaned panel before degenerate splitting. Useful when a caller
                may choose to keep degenerate households in scope.
            excluded_data:
                Cleaned degenerate-household subset.
            flags:
                Household-level regime flags aligned to cleaned_full index.
            summary:
                Cleaning and screening summary for reporting/debugging.
        """

        cleaned_full, neg_summary = self.handle_negatives(
            data,
            action=negative_action,
        )
        cleaned_full, miss_summary = self.handle_missing_values(
            cleaned_full,
            strategy=missing_strategy,
        )

        regular_data, excluded_data, flags = self.split_regular_and_degenerate_households(
            cleaned_full
        )

        prepared_regular = regular_data.copy()

        if transform != "none":
            prepared_regular = self.transform_series(
                prepared_regular,
                method=transform,
            )

        normalization_summary = None
        if normalize_method is not None:
            prepared_regular, normalization_summary = self.normalize_data(
                prepared_regular,
                method=normalize_method,
                axis=normalize_axis,
            )

        summary = {
            "negatives": neg_summary,
            "missing": miss_summary,
            "degenerate_flags": {
                "all_zero": int(flags["all_zero_flag"].sum()),
                "near_constant": int(flags["near_constant_flag"].sum()),
                "zero_heavy": int(flags["zero_heavy_flag"].sum()),
                "degenerate": int(flags["degenerate_flag"].sum()),
            },
            "n_cleaned_total": int(len(cleaned_full)),
            "n_regular_for_fit": int(len(prepared_regular)),
            "n_degenerate_excluded": int(len(excluded_data)),
            "transform": transform,
            "normalization": normalization_summary,
        }
        return prepared_regular, cleaned_full, excluded_data, flags, summary

    def transform_series(self, data: pd.DataFrame, method: str = "none") -> pd.DataFrame:
        """Apply an optional transform to the series values."""
        if method == "none":
            return data.copy()
        if method == "log1p":
            clipped = data.clip(lower=0)
            transformed = clipped.transform(np.log1p)  # keeps DataFrame type
            return transformed
        raise ValueError(f"Unsupported transform method: {method}")

    def prepare_for_feature_clustering(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        prepared_regular, cleaned_full, excluded_data, flags, base_summary = (
            self.prepare_regular_subset(
                data,
                transform="none",
                normalize_method=None,
                negative_action=NEGATIVE_VALUE_ACTION,
                missing_strategy=MISSING_VALUE_STRATEGY,
            )
        )

        data_for_fit = prepared_regular

        summary = {
            "intended_method": "feature_ward",
            **base_summary,
            "excluded_special_regimes_from_fit": {
                "all_zero": True,
                "near_constant": True,
                "zero_heavy": bool(EXCLUDE_ZERO_HEAVY_FROM_MAIN_CLUSTERING),
            },
            "scope_note": (
                "Feature-Ward and k-Shape are fit on the same screened regular-household "
                "subset so k-selection compares methods on the same data."
            ),
            "outlier_handling_applied": False,
            "outlier_policy_note": (
                "Outliers are retained by design because consumption spikes may reflect "
                "real household behaviour that is relevant for clustering."
            ),
            "output_semantics": (
                "Returns cleaned household time-series rows for feature extraction. "
                "This is not yet the final Ward clustering matrix."
            ),
        }

        logger.info(
            "Prepared data for feature-based clustering (fit rows=%d, excluded degenerate=%d)",
            len(data_for_fit),
            len(excluded_data),
        )
        details = {
            "degenerate_ids": excluded_data.index.tolist(),
            "degenerate_flag_table": flags.loc[excluded_data.index].copy(),
        }
        return data_for_fit, summary, details

    def prepare_for_kshape(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Prepare 2023 data for k-Shape clustering.

        k-Shape is shape-based, so the main requirement is row-wise
        standardization. Constant or all-zero households are not informative
        for k-Shape and are tracked explicitly.
        """
        prepared, _, excluded_data, flags, base_summary = self.prepare_regular_subset(
            data,
            transform="none",
            normalize_method="standard",
            normalize_axis=1,
        )

        if KSHAPE_NORMALIZATION != "zscore_rowwise":
            raise ValueError(
                f"Unsupported KSHAPE_NORMALIZATION setting: {KSHAPE_NORMALIZATION}"
            )

        excluded_ids = excluded_data.index.tolist()

        summary = {
            "intended_method": "kshape",
            **base_summary,
            "excluded_from_kshape_fit_count": int(len(excluded_ids)),
            "excluded_from_kshape_fit_ids": excluded_ids,
            "excluded_degenerate_from_fit": True,
            "fit_subset_shape": prepared.shape,
            "outlier_handling_applied": False,
            "outlier_policy_note": "Outliers are retained by design because consumption spikes may be real behaviour relevant for clustering.",
        }
        details = {
            "degenerate_ids": excluded_ids,
            "degenerate_flag_table": flags.loc[excluded_data.index].copy(),
        }
        return prepared, summary, details

    
    def prepare_for_clustering(
        self,
        data: pd.DataFrame,
        method: str,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Dispatch to the correct method-specific preparation routine."""
        if method == "feature_ward":
            return self.prepare_for_feature_clustering(data)
        if method == "kshape":
            return self.prepare_for_kshape(data)
        raise ValueError(f"Unknown clustering preparation method: {method}")

