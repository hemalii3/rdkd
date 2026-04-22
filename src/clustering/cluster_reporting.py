"""Cluster reporting and interpretation utilities.

This module builds human-readable summaries of clustering outputs for the
two-method experiment. It is designed to sit on top of:

- clustering.py
- eda_analysis.py
- feature_engineering.py (for Method 1 feature interpretation)

Main use cases:
- summarize cluster sizes
- summarize raw and normalized cluster profiles
- summarize monthly and weekly cluster behaviour
- extract representative households
- summarize cluster feature means
- build report-ready tables for comparing clustering methods
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

from src.preprocessing.eda_analysis import (
    aggregate_by_day_of_week,
    aggregate_by_month,
    build_phase1_clustering_summary,
)
from src.preprocessing.data_cleaning import z_normalize_rows
from src.utils.config import LEVEL_DOMINANCE_WARNING_THRESHOLD

if TYPE_CHECKING:
    from src.clustering.clustering import ClusteringArtifacts

logger = logging.getLogger(__name__)


# =============================================================================
# Small helpers
# =============================================================================

def _valid_cluster_labels(labels: pd.Series) -> pd.Series:
    """Keep only nonnegative cluster labels."""
    numeric = pd.to_numeric(labels, errors="coerce")
    valid_mask = numeric >= 0
    return labels.loc[valid_mask]


def _attach_labels(data: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Attach cluster labels to a DataFrame by index alignment."""
    df = data.copy()
    df["cluster"] = pd.to_numeric(labels.reindex(df.index), errors="coerce")
    df = df[df["cluster"].notna()].copy()
    df["cluster"] = df["cluster"].astype(int)
    return df


def _mean_profile_by_cluster(data: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Return mean row profile by cluster."""
    df = _attach_labels(data, labels)
    return df.groupby("cluster").mean(numeric_only=True)


# =============================================================================
# Main reporting class
# =============================================================================

class ClusterReporter:
    """Interpret and summarize clustering results."""

    # -------------------------------------------------------------------------
    # Core size / assignment summaries
    # -------------------------------------------------------------------------

    @staticmethod
    def cluster_size_summary(result: ClusteringArtifacts) -> pd.DataFrame:
        """Return cluster-size summary excluding special/fallback labels."""
        cluster_sizes = result.cluster_sizes.copy()
        if not cluster_sizes.empty:
            cluster_sizes = cluster_sizes[cluster_sizes["cluster"] >= 0].copy()
        return cluster_sizes

    @staticmethod
    def assignment_summary(result: ClusteringArtifacts) -> Dict[str, object]:
        """Return compact assignment diagnostics."""
        labels = result.labels.copy()
        numeric = pd.to_numeric(labels, errors="coerce")
        valid = numeric.loc[numeric >= 0].astype(int)

        n_total = int(len(labels))
        n_valid = int(len(valid))
        n_special = int((numeric < 0).fillna(False).sum())
        n_unassigned = int(numeric.isna().sum())

        counts = valid.value_counts()
        return {
            "method": result.method,
            "k_requested": int(result.k),
            "n_total_labels": n_total,
            "n_valid_labels": n_valid,
            "n_special_regime_labels": n_special,
            "n_regular_clusters_found": int(counts.shape[0]),
            "n_unassigned_households": n_unassigned,
            "min_cluster_size": int(counts.min()) if not counts.empty else None,
            "max_cluster_size": int(counts.max()) if not counts.empty else None,
        }
    
    @staticmethod
    def special_regime_assignment_summary(result: ClusteringArtifacts) -> Dict[str, object]:
        labels = result.labels.copy()
        numeric = pd.to_numeric(labels, errors="coerce")
        special = numeric[numeric < 0].astype(int)
        assigned_mask = numeric.notna()

        return {
            "special_regime_household_ids_sample": labels.index[numeric < 0].tolist()[:20],
            "n_special_regime_households": int(special.shape[0]),
            "pct_special_regime_households_all_rows": float((numeric < 0).mean() * 100),
            "pct_special_regime_households_assigned_only": float(
                (numeric[assigned_mask] < 0).mean() * 100
            ) if assigned_mask.any() else 0.0,
            "special_label_counts": special.value_counts().sort_index().to_dict(),
            "special_label_map": result.additional_outputs.get("special_label_map"),
        }
    
    @staticmethod
    def level_dominance_summary_by_cluster(
        data_2023: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        """Compare raw-level differences and normalized-shape differences by cluster."""
        valid = _valid_cluster_labels(labels)
        raw = data_2023.loc[valid.index]
        normalized = z_normalize_rows(raw)

        raw_means = _mean_profile_by_cluster(raw, valid).mean(axis=1)
        raw_stds = _mean_profile_by_cluster(raw, valid).std(axis=1)
        norm_means = _mean_profile_by_cluster(normalized, valid).mean(axis=1)
        norm_stds = _mean_profile_by_cluster(normalized, valid).std(axis=1)

        return pd.DataFrame({
            "cluster_raw_profile_mean": raw_means,
            "cluster_raw_profile_std": raw_stds,
            "cluster_normalized_profile_mean": norm_means,
            "cluster_normalized_profile_std": norm_stds,
        })
    
    @staticmethod
    def household_mean_summary_by_cluster(
        data_2023: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        """Summarize household-average consumption by cluster."""
        valid = _valid_cluster_labels(labels)
        hh_mean = data_2023.loc[valid.index].mean(axis=1).to_frame("household_mean")
        df = _attach_labels(hh_mean, valid)
        return df.groupby("cluster")["household_mean"].agg(["mean", "median", "std", "min", "max"])
    
    
    # -------------------------------------------------------------------------
    # Raw / normalized profile summaries
    # -------------------------------------------------------------------------

    @staticmethod
    def mean_raw_profile_by_cluster(
        data_2023: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        """Return mean raw daily profile per cluster."""
        valid = _valid_cluster_labels(labels)
        return _mean_profile_by_cluster(data_2023, valid)

    @staticmethod
    def mean_normalized_profile_by_cluster(
        data_2023: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        """Return mean row-z-normalized daily profile per cluster."""
        valid = _valid_cluster_labels(labels)
        normalized = z_normalize_rows(data_2023.loc[valid.index])
        return _mean_profile_by_cluster(normalized, valid)

    # -------------------------------------------------------------------------
    # Weekly / monthly summaries
    # -------------------------------------------------------------------------

    @staticmethod
    def mean_weekly_profile_by_cluster(
        data_2023: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        """Return mean day-of-week profile per cluster."""
        valid = _valid_cluster_labels(labels)
        dow = aggregate_by_day_of_week(data_2023.loc[valid.index])
        return _mean_profile_by_cluster(dow, valid)

    @staticmethod
    def mean_monthly_profile_by_cluster(
        data_2023: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        """Return mean monthly profile per cluster."""
        valid = _valid_cluster_labels(labels)
        monthly = aggregate_by_month(data_2023.loc[valid.index])
        return _mean_profile_by_cluster(monthly, valid)

    # -------------------------------------------------------------------------
    # Diagnostics / EDA summaries by cluster
    # -------------------------------------------------------------------------

    @staticmethod
    def diagnostic_summary_by_cluster(
        data_2023: pd.DataFrame,
        labels: pd.Series,
    ) -> pd.DataFrame:
        """Return cluster means for the Phase 1 household diagnostics."""
        valid = _valid_cluster_labels(labels)
        diagnostics = build_phase1_clustering_summary(data_2023.loc[valid.index])
        return _mean_profile_by_cluster(diagnostics, valid)

    # -------------------------------------------------------------------------
    # Method 1 feature summaries
    # -------------------------------------------------------------------------

    @staticmethod
    def feature_summary_by_cluster(
        labels: pd.Series,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return mean feature values by cluster."""
        valid = _valid_cluster_labels(labels)
        df = _attach_labels(features.loc[valid.index], valid)
        return df.groupby("cluster").mean(numeric_only=True)

    @staticmethod
    def top_feature_differences_by_cluster(
        labels: pd.Series,
        features: pd.DataFrame,
        top_n: int = 10,
    ) -> Dict[int, pd.DataFrame]:
        """Return the most distinctive features for each cluster.

        Distinctiveness is defined as the absolute deviation of a cluster mean
        from the overall mean, standardized by the overall feature std.
        """
        valid = _valid_cluster_labels(labels)
        X = features.loc[valid.index].copy()
        overall_mean = X.mean(axis=0)
        overall_std = X.std(axis=0).replace(0, np.nan)

        cluster_means = ClusterReporter.feature_summary_by_cluster(valid, X)
        output: Dict[int, pd.DataFrame] = {}

        for cluster_id in cluster_means.index:
            z = (cluster_means.loc[cluster_id] - overall_mean) / overall_std
            table = pd.DataFrame({
                "cluster_mean": cluster_means.loc[cluster_id],
                "overall_mean": overall_mean,
                "standardized_difference": z,
                "abs_standardized_difference": z.abs(),
            }).sort_values("abs_standardized_difference", ascending=False)

            table["feature_family"] = np.select(
                [
                    table.index.isin(["mean", "std", "cv", "zero_pct"]),
                    table.index.str.startswith("dow_"),
                    table.index.str.startswith("month_"),
                    table.index.str.startswith("norm_dow_"),
                    table.index.str.startswith("norm_month_"),
                    table.index.isin([
                        "weekend_weekday_contrast",
                        "winter_summer_contrast",
                        "seasonal_amplitude",
                        "monthly_profile_std",
                        "norm_monthly_profile_std",
                        "norm_monthly_profile_range",
                        "norm_winter_summer_diff",
                        "norm_weekend_weekday_diff",
                        "weekday_range",
                    ]),
                    table.index.isin([
                        "base_load_ratio",
                        "seasonal_peak_ratio",
                        "seasonal_range_ratio",
                        "winter_share",
                        "summer_share",
                        "top3_month_share",
                        "monthly_entropy",
                        "peak_month_sin",
                        "peak_month_cos",
                        "trough_month_sin",
                        "trough_month_cos",
                    ]),
                    table.index.isin([
                        "autocorr_lag_1",
                        "autocorr_lag_7",
                        "autocorr_lag_30",
                        "rolling_std_7",
                        "rolling_std_30",
                        "diff_mean",
                        "diff_std",
                        "abs_diff_mean",
                        "diff_max",
                        "diff_min",
                        "p95_to_median_ratio",
                        "p99_to_median_ratio",
                        "max_to_p95_ratio",
                        "winter_std_norm",
                        "summer_std_norm",
                        "winter_summer_volatility_ratio",
                        "trend_slope",
                        "trend_slope_normalized",
                    ]),
                ],
                [
                    "raw_level",
                    "raw_weekly_profile",
                    "raw_monthly_profile",
                    "normalized_weekly_shape",
                    "normalized_monthly_shape",
                    "shape_summary",
                    "energy_profile",
                    "persistence_trend",
                ],
                default="other",
            )

            output[int(cluster_id)] = table.head(top_n)

        return output

    # -------------------------------------------------------------------------
    # Representative households
    # -------------------------------------------------------------------------

    @staticmethod
    def representative_households_by_raw_distance(
        data_2023: pd.DataFrame,
        labels: pd.Series,
        n_per_cluster: int = 3,
    ) -> Dict[int, List[Any]]:
        """Find representative households by Euclidean distance to cluster raw mean profile.

        This is useful for level-sensitive interpretation, but it should be read
        alongside normalized-distance representatives to distinguish scale from shape.
        """
        valid = _valid_cluster_labels(labels)
        representatives: Dict[int, List[Any]] = {}

        for cluster_id in sorted(valid.unique()):
            cluster_ids = valid.index[valid == cluster_id]
            cluster_data = data_2023.loc[cluster_ids]
            centroid = cluster_data.mean(axis=0).to_numpy(dtype=float)

            distances = {}
            for household_id in cluster_ids:
                series = cluster_data.loc[household_id].to_numpy(dtype=float)
                dist = float(np.linalg.norm(series - centroid))
                distances[household_id] = dist

            closest = sorted(distances.items(), key=lambda x: x[1])[:n_per_cluster]
            representatives[int(cluster_id)] = [hh for hh, _ in closest]

        return representatives

    @staticmethod
    def representative_households_by_normalized_distance(
        data_2023: pd.DataFrame,
        labels: pd.Series,
        n_per_cluster: int = 3,
    ) -> Dict[int, List[Any]]:
        """Find representative households by distance to cluster normalized mean profile."""
        valid = _valid_cluster_labels(labels)
        normalized = z_normalize_rows(data_2023.loc[valid.index])
        representatives: Dict[int, List[Any]] = {}

        for cluster_id in sorted(valid.unique()):
            cluster_ids = valid.index[valid == cluster_id]
            cluster_data = normalized.loc[cluster_ids]
            centroid = cluster_data.mean(axis=0).to_numpy(dtype=float)

            distances = {}
            for household_id in cluster_ids:
                series = cluster_data.loc[household_id].to_numpy(dtype=float)
                dist = float(np.linalg.norm(series - centroid))
                distances[household_id] = dist

            closest = sorted(distances.items(), key=lambda x: x[1])[:n_per_cluster]
            representatives[int(cluster_id)] = [hh for hh, _ in closest]

        return representatives

    @staticmethod
    def medoid_households_from_result(
        result: ClusteringArtifacts,
    ) -> Optional[List[int]]:
        """Return medoid household IDs when available."""
        medoid_ids = result.additional_outputs.get("medoid_household_ids")
        if medoid_ids is None:
            return None
        if isinstance(medoid_ids, (str, bytes)):
            logger.warning("medoid_household_ids is not an iterable of IDs")
            return None
        if not isinstance(medoid_ids, Iterable):
            logger.warning("medoid_household_ids is not iterable")
            return None
        return list(medoid_ids)


    
    # -------------------------------------------------------------------------
    # Cluster label export table
    # -------------------------------------------------------------------------

    @staticmethod
    def cluster_assignment_table(result: ClusteringArtifacts) -> pd.DataFrame:
        """Return household -> cluster assignment table."""
        return pd.DataFrame({
            "household_id": result.labels.index,
            "cluster": result.labels.values,
        }).sort_values("household_id").reset_index(drop=True)
    
    # -------------------------------------------------------------------------
    # Forecasting handoff table
    # -------------------------------------------------------------------------
    @staticmethod
    def forecasting_ready_handoff_table(labels: pd.Series) -> pd.DataFrame:
        """
        Return a forecasting-ready household -> cluster table.
        Requires every household to have a regular nonnegative cluster ID.
        """
        numeric = pd.to_numeric(labels, errors="coerce")

        if numeric.isna().any():
            raise ValueError("Labels contain non-numeric values.")
        if (numeric < 0).any():
            raise ValueError(
                "forecasting_ready_handoff_table requires fully assigned nonnegative cluster labels."
            )

        labels = numeric.astype(int).sort_index()

        return pd.DataFrame({
            "household_id": labels.index,
            "cluster": labels.values,
        }).reset_index(drop=True)
    
    # -------------------------------------------------------------------------
    # Cluster meaningfulness diagnostics
    # -------------------------------------------------------------------------
    
    @staticmethod
    def cluster_meaningfulness_diagnostics(
        data_2023: pd.DataFrame,
        labels: pd.Series,
    ) -> Dict[str, object]:
        """Diagnose whether clustering appears overly driven by raw consumption level.

        The main idea is to compare:
        - how much clusters differ in average household consumption level
        - how much clusters differ in normalized profile shape

        A high level-to-shape ratio suggests the clustering may mainly reflect
        magnitude separation rather than genuinely different temporal behaviour.
        """
        valid = _valid_cluster_labels(labels)

        if valid.empty:
            return {
                "n_valid_clusters": 0,
                "cluster_mean_spread": None,
                "normalized_shape_separation": None,
                "level_to_shape_ratio": None,
                "level_dominance_warning": None,
            }

        hh_mean_summary = ClusterReporter.household_mean_summary_by_cluster(
            data_2023=data_2023,
            labels=valid,
        )

        normalized_profiles = ClusterReporter.mean_normalized_profile_by_cluster(
            data_2023=data_2023,
            labels=valid,
        )

        cluster_mean_spread = float(
            hh_mean_summary["mean"].max() - hh_mean_summary["mean"].min()
        )

        if normalized_profiles.shape[0] < 2:
            normalized_shape_separation = None
        else:
            profile_values = normalized_profiles.to_numpy(dtype=float)
            pairwise_distances = []
            for i in range(profile_values.shape[0]):
                for j in range(i + 1, profile_values.shape[0]):
                    pairwise_distances.append(
                        float(np.linalg.norm(profile_values[i] - profile_values[j]))
                    )
            normalized_shape_separation = (
                float(np.mean(pairwise_distances)) if pairwise_distances else None
            )

        level_to_shape_ratio = (
            None
            if normalized_shape_separation is None
            else cluster_mean_spread / (normalized_shape_separation + 1e-6)
        )

        return {
            "n_valid_clusters": int(valid.nunique()),
            "cluster_mean_spread": cluster_mean_spread,
            "normalized_shape_separation": normalized_shape_separation,
            "level_to_shape_ratio": level_to_shape_ratio,
            "level_dominance_warning": (
                None
                if level_to_shape_ratio is None
                else bool(level_to_shape_ratio > LEVEL_DOMINANCE_WARNING_THRESHOLD)
            ),
        }

    # -------------------------------------------------------------------------
    # Full report bundle for one result
    # -------------------------------------------------------------------------
    # This is a notebook/reporting convenience bundle.
    # Core pipeline decisions for Task 1 should rely primarily on:
    # - cluster_sizes
    # - household_mean_summary_by_cluster
    # - level_dominance_summary_by_cluster
    # - cluster_meaningfulness_diagnostics
    
    def build_report_bundle(
        self,
        result: ClusteringArtifacts,
        data_2023: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        n_representatives: int = 3,
        include_profiles: bool = False,
        include_diagnostics: bool = False,
        include_feature_tables: bool = False,
    ) -> Dict[str, object]:
        """Build a full interpretation bundle for one clustering result."""
        labels = result.labels.copy()
        valid = _valid_cluster_labels(labels)

        bundle: Dict[str, object] = {
            "assignment_summary": self.assignment_summary(result),
            "cluster_sizes": self.cluster_size_summary(result),
            "household_mean_summary_by_cluster": self.household_mean_summary_by_cluster(data_2023, valid),
            "cluster_meaningfulness_diagnostics": self.cluster_meaningfulness_diagnostics(data_2023, valid),
            "special_regime_assignment_summary": self.special_regime_assignment_summary(result),
        }

        if include_profiles:
            bundle["mean_weekly_profile_by_cluster"] = self.mean_weekly_profile_by_cluster(data_2023, valid)
            bundle["mean_monthly_profile_by_cluster"] = self.mean_monthly_profile_by_cluster(data_2023, valid)

        if include_diagnostics:
            bundle["level_dominance_summary_by_cluster"] = self.level_dominance_summary_by_cluster(data_2023, valid)
            bundle["diagnostic_summary_by_cluster"] = self.diagnostic_summary_by_cluster(data_2023, valid)
            bundle["representative_households_raw"] = self.representative_households_by_raw_distance(
                data_2023, valid, n_per_cluster=n_representatives
            )
            bundle["representative_households_normalized"] = self.representative_households_by_normalized_distance(
                data_2023, valid, n_per_cluster=n_representatives
            )

        if features is not None and include_feature_tables:
            feature_subset = features.loc[valid.index]
            bundle["feature_summary_by_cluster"] = self.feature_summary_by_cluster(valid, feature_subset)
            bundle["top_feature_differences_by_cluster"] = self.top_feature_differences_by_cluster(
                valid, feature_subset
            )

        medoids = self.medoid_households_from_result(result)
        if medoids is not None:
            bundle["medoid_households"] = medoids

        return bundle