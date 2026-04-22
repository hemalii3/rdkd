"""Clustering module for the two-method household clustering experiment.
Design goals:
- use only 2023-prepared inputs
- return labels and metadata in a consistent structure
- support k-search and downstream comparison
- preserve interpretability artifacts for later reporting

This module focuses on clustering only. Cluster stability and forecasting
evaluation should be implemented in separate modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pyexpat import features
from typing import Any, Dict, List, Literal, Mapping, Optional, cast

import numpy as np
import pandas as pd

from src.preprocessing.data_cleaning import z_normalize_rows

KShape: Any | None = None
try:
    from tslearn.clustering import KShape as _KShape
    KShape = _KShape
except Exception:
    KShape = None


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.utils.config import (
    FEATURE_WARD_LINKAGE,
    MIN_CLUSTER_SIZE,
    KEEP_SPECIAL_CLUSTERS_FOR_FORECASTING,
    MIN_SPECIAL_CLUSTER_SIZE_FOR_FORECASTING,
    RANDOM_SEED,
    DEGENERATE_SPECIAL_LABEL,
    SPECIAL_CLUSTER_LABELS,
)

from src.preprocessing.data_cleaning import DataCleaner
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.feature_selection import FeatureSelector
from src.clustering.cluster_reporting import ClusterReporter

logger = logging.getLogger(__name__)


TSLEARN_KSHAPE_AVAILABLE = KShape is not None

# =============================================================================
# Data structure for results
# =============================================================================

@dataclass
class ClusteringArtifacts:
    """Container for clustering outputs."""
    method: str
    k: int
    labels: pd.Series
    cluster_sizes: pd.DataFrame
    metrics: Dict[str, Optional[float]]
    fitted_model: object
    prepared_input: pd.DataFrame
    additional_outputs: Dict[str, object]


# =============================================================================
# Helper utilities
# =============================================================================

def _build_cluster_size_table(labels: pd.Series) -> pd.DataFrame:
    """Build a cluster-size summary table using only real cluster labels."""
    numeric = pd.to_numeric(labels, errors="coerce")
    valid_mask = numeric >= 0
    valid_labels = labels.loc[valid_mask]

    if valid_labels.empty:
        return pd.DataFrame(columns=["cluster", "n_households", "pct_households"])

    counts = valid_labels.value_counts().sort_index()

    n_households = counts.to_numpy(dtype=np.int64)
    total = float(n_households.sum())
    pct_households = (n_households.astype(np.float64) / total) * 100.0

    table = pd.DataFrame(
        {
            "cluster": counts.index.to_numpy(dtype=np.int64),
            "n_households": n_households,
            "pct_households": pct_households,
        }
    )
    return table

def _compute_balance_metric(cluster_sizes: pd.DataFrame) -> float:
    """Simple balance metric: min cluster size / max cluster size."""
    if cluster_sizes.empty:
        return float("nan")
    min_size = float(cluster_sizes["n_households"].min())
    max_size = float(cluster_sizes["n_households"].max())
    if max_size <= 0:
        return float("nan")
    return min_size / max_size

def _count_selected_features_by_family(columns: List[str]) -> Dict[str, int]:
    """Count selected Method 1 features by broad interpretation family."""
    counts = {
        "raw_level_features": 0,
        "raw_profile_features": 0,
        "normalized_shape_features": 0,
        "energy_profile_features": 0,
        "persistence_trend_features": 0,
        "other_features": 0,
    }

    for col in columns:
        if col in {"mean", "std", "cv", "zero_pct"}:
            counts["raw_level_features"] += 1
        elif col.startswith("dow_") or col.startswith("month_") or col in {
            "weekend_weekday_contrast",
            "winter_summer_contrast",
            "seasonal_amplitude",
            "monthly_profile_std",
            "weekday_range",
        }:
            counts["raw_profile_features"] += 1
        elif col.startswith("norm_dow_") or col.startswith("norm_month_") or col in {
            "norm_monthly_profile_std",
            "norm_monthly_profile_range",
            "norm_winter_summer_diff",
            "norm_weekend_weekday_diff",
        }:
            counts["normalized_shape_features"] += 1
        elif col in {
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
        }:
            counts["energy_profile_features"] += 1
        elif col in {
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
        }:
            counts["persistence_trend_features"] += 1
        else:
            counts["other_features"] += 1

    return counts


LinkageType = Literal["ward", "complete", "average", "single"]


def _normalize_linkage(linkage: str) -> LinkageType:
    """Validate linkage string and narrow it to the supported Literal type."""
    normalized = linkage.strip().lower()
    allowed: tuple[LinkageType, ...] = ("ward", "complete", "average", "single")
    if normalized not in allowed:
        raise ValueError(f"Invalid linkage: {linkage}. Must be one of {allowed}.")
    return cast(LinkageType, normalized)


def _safe_vector_metrics(
    X: pd.DataFrame,
    labels: pd.Series,
) -> Dict[str, Optional[float]]:
    """Compute vector-space clustering metrics safely."""
    unique_labels = np.unique(labels.to_numpy())
    if len(unique_labels) < 2:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }

    try:
        sil = float(silhouette_score(X, labels))
    except Exception:
        sil = None

    try:
        db = float(davies_bouldin_score(X, labels))
    except Exception:
        db = None

    try:
        ch = float(calinski_harabasz_score(X, labels))
    except Exception:
        ch = None

    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
    }



def _reshape_for_tslearn(data: pd.DataFrame) -> np.ndarray:
    """Convert (n_households, n_days) DataFrame to tslearn shape."""
    arr = data.to_numpy(dtype=float)
    return arr[:, :, np.newaxis]

def _build_special_label_series(
    excluded_ids: List[int],
    degenerate_flag_table: Optional[pd.DataFrame],
) -> pd.Series:
    """Build a consistent special-label Series for excluded households."""
    if not excluded_ids:
        return pd.Series(dtype="int64", name="cluster")

    if degenerate_flag_table is None or degenerate_flag_table.empty:
        return pd.Series(DEGENERATE_SPECIAL_LABEL, index=excluded_ids, name="cluster", dtype="int64")

    special_labels = pd.Series(index=degenerate_flag_table.index, dtype="float64", name="cluster")

    all_zero_mask = degenerate_flag_table["all_zero_flag"]
    near_constant_mask = degenerate_flag_table["near_constant_flag"] & ~all_zero_mask
    zero_heavy_mask = (
        degenerate_flag_table["zero_heavy_flag"]
        & ~all_zero_mask
        & ~near_constant_mask
    )

    special_labels.loc[degenerate_flag_table.index[all_zero_mask]] = -1
    special_labels.loc[degenerate_flag_table.index[near_constant_mask]] = -2
    special_labels.loc[degenerate_flag_table.index[zero_heavy_mask]] = -3

    unassigned = special_labels.isna()
    if unassigned.any():
        unknown_ids = special_labels.index[unassigned].tolist()
        raise ValueError(
            f"Found degenerate households that do not match any special regime rule: {unknown_ids[:10]}"
        )

    return special_labels.astype(int)



# =============================================================================
# Main clustering class
# =============================================================================

class HouseholdClusterer:
    """Common clustering interface for both methods."""

    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        self.cleaner = DataCleaner(random_seed=random_seed)
        self.feature_engineer = FeatureEngineer(random_seed=random_seed)

    # -------------------------------------------------------------------------
    # Method 1: feature-based Ward clustering
    # -------------------------------------------------------------------------
    def prepare_feature_ward_input(
        self,
        data_2023: pd.DataFrame,
    ) -> Dict[str, object]:
        """Prepare Method 1 input once so multiple k values can reuse it."""
        prepared_data, prep_summary, prep_details = self.cleaner.prepare_for_feature_clustering(data_2023)

        # Use the shape-first feature set for the current Task 1 pipeline to reduce
        # dominance from raw level/profile magnitude and focus Ward clustering more on
        # behavioral pattern structure.
        result = self.feature_engineer.extract_final_feature_ward_features(
            prepared_data,
            return_metadata=True,
        )
        features, feature_metadata = cast(
            tuple[pd.DataFrame, dict[str, object]],
            result,
        )

        selector = FeatureSelector(random_seed=self.random_seed)
        prepared = selector.prepare_feature_method_input(
            features=features,
            feature_block_map=cast(dict[str, list[str]], feature_metadata["feature_blocks"]),
        )

        return {
            "prep_summary": prep_summary,
            "prep_details": prep_details,
            "feature_metadata": feature_metadata,
            "prepared": prepared,
            "original_features": features,
        }
    
    def run_feature_ward_on_prepared(
        self,
        prepared_bundle: Mapping[str, Any],
        k: int,
        linkage: str = FEATURE_WARD_LINKAGE,
    ) -> ClusteringArtifacts:
        """Run Ward clustering for one k using already-prepared Method 1 input."""
        prep_summary = cast(Mapping[str, Any], prepared_bundle["prep_summary"])
        prep_details = cast(Mapping[str, Any], prepared_bundle["prep_details"])
        feature_metadata = prepared_bundle["feature_metadata"]
        prepared = cast(Mapping[str, Any], prepared_bundle["prepared"])
        original_features = cast(pd.DataFrame, prepared_bundle["original_features"])

        X = cast(pd.DataFrame, prepared["clustering_input"])

        linkage_value = _normalize_linkage(linkage)
        if linkage_value != "ward":
            raise ValueError(
                "Method 1 is defined as Ward hierarchical clustering; "
                f"got linkage='{linkage_value}'."
            )

        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        raw_labels = model.fit_predict(X)
        labels = pd.Series(raw_labels, index=X.index, name="cluster")

        excluded_ids = prep_details.get("degenerate_ids", [])
        degenerate_flag_table = prep_details.get("degenerate_flag_table")

        if excluded_ids:
            special_labels = _build_special_label_series(
                excluded_ids=excluded_ids,
                degenerate_flag_table=degenerate_flag_table,
            )
            labels = pd.concat([labels, special_labels]).sort_index()

        cluster_sizes = _build_cluster_size_table(labels)
        min_cluster_size = (
            int(cluster_sizes["n_households"].min()) if not cluster_sizes.empty else None
        )
        small_cluster_warning = (
            min_cluster_size is not None and min_cluster_size < MIN_CLUSTER_SIZE
        )
        if small_cluster_warning:
            logger.warning(
                "feature_ward produced a very small cluster (min size=%d) at k=%d",
                min_cluster_size,
                k,
            )

        metrics = _safe_vector_metrics(X, pd.Series(raw_labels, index=X.index))
        metrics["cluster_balance"] = _compute_balance_metric(cluster_sizes)

        return ClusteringArtifacts(
            method="feature_ward",
            k=k,
            labels=labels,
            cluster_sizes=cluster_sizes,
            metrics=metrics,
            fitted_model=model,
            prepared_input=X,
            additional_outputs={
                "prep_summary": prep_summary,
                "feature_metadata": feature_metadata,
                "original_features": original_features,
                "filtered_features": prepared["filtered_features"],
                "selector_summary": prepared["summary"],
                "selected_feature_names": prepared["selected_feature_names"],
                "selected_feature_family_counts": _count_selected_features_by_family(
                    prepared["selected_feature_names"]
                ),
                "pca_model": prepared["pca_model"],
                "fit_household_ids": X.index.tolist(),
                "excluded_ids": excluded_ids,
                "method_role": "primary_candidate",
                "evaluation_scope": "full_data_regular_households",
                "small_cluster_warning": small_cluster_warning,
                "special_label_map": SPECIAL_CLUSTER_LABELS,
            },
        )
    # -------------------------------------------------------------------------
    # Method 2: k-Shape
    # -------------------------------------------------------------------------

    def prepare_kshape_input(
        self,
        data_2023: pd.DataFrame,
        max_households: Optional[int] = None,
    ) -> Dict[str, object]:
        """Prepare reusable input for k-Shape across multiple k values."""
        prepared_data, prep_summary, prep_details = self.cleaner.prepare_for_kshape(data_2023)
        excluded_ids = prep_details.get("degenerate_ids", [])
        all_regular_ids = prepared_data.index.tolist()
        fit_data = prepared_data.copy()
        sampled_out_ids: List[int] = []

        was_sampled = False
        n_fit_available = int(fit_data.shape[0])

        if max_households is not None and fit_data.shape[0] > max_households:
            fit_data = fit_data.sample(
                n=max_households,
                random_state=self.random_seed,
            )
            fit_index_set = set(fit_data.index)
            sampled_out_ids = [hh_id for hh_id in all_regular_ids if hh_id not in fit_index_set]
            was_sampled = True

        return {
            "fit_data": fit_data,
            "prep_summary": prep_summary,
            "excluded_ids": excluded_ids,
            "max_households": max_households,
            "prep_details": prep_details,
            "n_fit_available": n_fit_available,
            "n_fit_used": int(fit_data.shape[0]),
            "was_sampled": was_sampled,
            "sampled_out_ids": sampled_out_ids,
        }
    

    def run_kshape_on_prepared(
        self,
        kshape_bundle: Dict[str, object],
        k: int,
        n_init: int = 5,
        verbose: bool = False,
    ) -> ClusteringArtifacts:
        """Run k-Shape clustering using already-prepared input."""
        if not TSLEARN_KSHAPE_AVAILABLE:
            raise ImportError(
                "tslearn is required for k-Shape clustering. "
                "Install with: pip install tslearn"
            )

        fit_data = cast(pd.DataFrame, kshape_bundle["fit_data"])
        prep_summary = cast(Dict[str, Any], kshape_bundle["prep_summary"])
        excluded_ids = cast(List[int], kshape_bundle["excluded_ids"])

        min_required = max(k * 10, k + 1)
        if fit_data.shape[0] < min_required:
            raise ValueError(
                f"Not enough households available for a stable k-Shape fit: "
                f"{fit_data.shape[0]} rows for k={k}; require at least {min_required}."
            )

        X_ts = _reshape_for_tslearn(fit_data)

        if KShape is None:
            raise ImportError(
                "tslearn is required for k-Shape clustering. Install with: pip install tslearn"
            )

        model = KShape(
            n_clusters=k,
            random_state=self.random_seed,
            n_init=n_init,
            verbose=verbose,
        )

        raw_labels = model.fit_predict(X_ts)
        labels = pd.Series(raw_labels, index=fit_data.index, name="cluster")

        sampled_out_ids = cast(List[int], kshape_bundle.get("sampled_out_ids", []))
        if sampled_out_ids:
            unsampled_series = pd.Series(
                np.nan,
                index=sampled_out_ids,
                name="cluster",
                dtype="float64",
            )
            labels = pd.concat([labels, unsampled_series]).sort_index()

        prep_details = cast(Dict[str, Any], kshape_bundle["prep_details"])
        degenerate_flag_table = prep_details.get("degenerate_flag_table")

        if excluded_ids:
            excluded_series = _build_special_label_series(
                excluded_ids=excluded_ids,
                degenerate_flag_table=degenerate_flag_table,
            )
            labels = pd.concat([labels, excluded_series]).sort_index()

        cluster_sizes = _build_cluster_size_table(labels)
        fit_labels = pd.Series(raw_labels, index=fit_data.index, name="cluster")
        vector_metrics = _safe_vector_metrics(fit_data, fit_labels)

        metrics = {
            "silhouette": vector_metrics["silhouette"],
            "davies_bouldin": vector_metrics["davies_bouldin"],
            "calinski_harabasz": vector_metrics["calinski_harabasz"],
            "cluster_balance": _compute_balance_metric(cluster_sizes),
        }

        return ClusteringArtifacts(
            method="kshape",
            k=k,
            labels=labels,
            cluster_sizes=cluster_sizes,
            metrics=metrics,
            fitted_model=model,
            prepared_input=fit_data,
            additional_outputs={
                "prep_summary": prep_summary,
                "excluded_ids": excluded_ids,
                "cluster_centers_": getattr(model, "cluster_centers_", None),
                "special_label_map": SPECIAL_CLUSTER_LABELS,
                "n_fit_available": kshape_bundle.get("n_fit_available"),
                "n_fit_used": kshape_bundle.get("n_fit_used"),
                "was_sampled": kshape_bundle.get("was_sampled"),
                "sampled_out_regular_ids": sampled_out_ids,
                "method_role": "sequence_benchmark",
                "evaluation_scope": (
                    "sampled_regular_households"
                    if kshape_bundle.get("was_sampled")
                    else "full_data_regular_households"
                ),
                "metric_note": (
                    "Vector-space internal metrics are reported on the prepared standardized "
                    "input for reference only; they are not directly equivalent to the "
                    "shape-based k-Shape objective."
                ),
            },
        )
    
    
    def run_final_clustering(
        self,
        data_2023: pd.DataFrame,
        k: int,
    ) -> ClusteringArtifacts:
        return self.run_feature_ward_on_prepared(
            prepared_bundle=self.prepare_feature_ward_input(data_2023),
            k=k,
        )
            

    def assign_special_regime_households(
        self,
        result: ClusteringArtifacts,
        data_2023: pd.DataFrame,
        strategy: str = "nearest_cluster_profile",
    ) -> pd.Series:
        """
        Convert diagnostic labels into forecasting-ready labels by assigning
        special-regime households to regular nonnegative clusters.

        Diagnostic labels in `result.labels` are preserved elsewhere; this function
        returns a full household-level label Series with only nonnegative cluster IDs.
        """
        if strategy != "nearest_cluster_profile":
            raise ValueError(f"Unsupported assignment strategy: {strategy}")

        labels = result.labels.copy()
        numeric = pd.to_numeric(labels, errors="coerce")
        

        if KEEP_SPECIAL_CLUSTERS_FOR_FORECASTING:
            numeric_labels = pd.to_numeric(labels, errors="raise").astype(int)
            special_counts = numeric_labels[numeric_labels < 0].value_counts()
            keep_special = special_counts[
                special_counts >= MIN_SPECIAL_CLUSTER_SIZE_FOR_FORECASTING
            ].index.tolist()

            regular_labels_only = numeric_labels[numeric_labels >= 0]
            next_cluster_id = (
                int(regular_labels_only.max()) + 1
                if not regular_labels_only.empty
                else 0
            )
            
            remapped_labels = numeric_labels.copy()

            for old_special_label in sorted(keep_special):
                remapped_labels.loc[remapped_labels == old_special_label] = next_cluster_id
                next_cluster_id += 1

            remaining_special_ids = remapped_labels.index[remapped_labels < 0]

            if len(remaining_special_ids) == 0:
                return remapped_labels.astype(int)
        else:
            remapped_labels = pd.to_numeric(labels, errors="raise").astype(int)
            remaining_special_ids = remapped_labels.index[remapped_labels < 0]

        if remapped_labels.isna().any():
            raise ValueError(
                "assign_special_regime_households() requires a complete clustering result "
                "with no unassigned households. Found NaN labels, likely from sampled benchmarks."
            )

        regular_ids = remapped_labels.index[remapped_labels >= 0]
        special_ids = remaining_special_ids

        if len(special_ids) == 0:
            return remapped_labels.astype(int)

        reporter = ClusterReporter()
        regular_labels = remapped_labels.loc[regular_ids].astype(int)

        cluster_profiles = reporter.mean_normalized_profile_by_cluster(
            data_2023=data_2023.loc[regular_ids],
            labels=regular_labels,
        )

        special_data = data_2023.loc[special_ids]
        special_normalized = z_normalize_rows(special_data)

        assigned = {}
        for hh_id in special_normalized.index:
            row = special_normalized.loc[hh_id].to_numpy(dtype=float)

            best_cluster = None
            best_distance = None

            for cluster_id in cluster_profiles.index:
                centroid = cluster_profiles.loc[cluster_id].to_numpy(dtype=float)
                dist = float(np.linalg.norm(row - centroid))
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    best_cluster = int(cluster_id)

            if best_cluster is None:
                raise ValueError(f"Could not assign special-regime household {hh_id}.")

            assigned[hh_id] = best_cluster

        forecasting_labels = remapped_labels.copy()

        for hh_id, cluster_id in assigned.items():
            forecasting_labels.loc[hh_id] = cluster_id

        return pd.to_numeric(forecasting_labels, errors="raise").astype(int)
    

    def order_final_clusters_by_profile(
        self,
        result: ClusteringArtifacts,
        data_2023: pd.DataFrame,
    ) -> ClusteringArtifacts:
        """
        Relabel final nonnegative clusters into a stable interpretable order
        for whatever k was selected as the final solution.

        This function does NOT assign semantic names. It only remaps the fitted
        cluster IDs into an ordered label set 0..k-1 using profile summaries.

        Ordering rule:
            1. stronger winter-vs-summer contrast first
            2. stronger weekend lift first
            3. stronger weekly routine first
            4. larger normalized-profile range first
            5. higher overall consumption level first

        Special negative labels (-1, -2, -3, etc.) are preserved unchanged.
        """
        reporter = ClusterReporter()

        labels = result.labels.copy()
        numeric = pd.to_numeric(labels, errors="coerce")

        if numeric.isna().any():
            raise ValueError(
                "order_final_clusters_by_profile() requires a complete clustering result "
                "with no unassigned households. Sampled benchmark results cannot be relabeled "
                "as final clusters."
            )
        valid_mask = numeric >= 0

        if valid_mask.sum() == 0:
            raise ValueError("No nonnegative cluster labels available to relabel.")

        valid_labels = numeric.loc[valid_mask].astype(int)

        monthly = reporter.mean_monthly_profile_by_cluster(data_2023, valid_labels)
        weekly = reporter.mean_weekly_profile_by_cluster(data_2023, valid_labels)
        hh_mean = reporter.household_mean_summary_by_cluster(data_2023, valid_labels)

        winter_mean = monthly[["Dec", "Jan", "Feb"]].mean(axis=1)
        summer_mean = monthly[["Jun", "Jul", "Aug"]].mean(axis=1)
        winter_minus_summer = winter_mean - summer_mean

        overall_level = hh_mean["mean"]

        weekend_mean = weekly[["Sat", "Sun"]].mean(axis=1)
        weekday_mean = weekly[["Mon", "Tue", "Wed", "Thu", "Fri"]].mean(axis=1)
        weekend_lift = weekend_mean - weekday_mean

        weekly_routine_strength = (weekly.max(axis=1) - weekly.min(axis=1))

        normalized_profiles = reporter.mean_normalized_profile_by_cluster(data_2023, valid_labels)
        normalized_profile_range = normalized_profiles.max(axis=1) - normalized_profiles.min(axis=1)

        ordering_table = pd.DataFrame(
            {
                "winter_minus_summer": winter_minus_summer,
                "weekend_lift": weekend_lift,
                "weekly_routine_strength": weekly_routine_strength,
                "normalized_profile_range": normalized_profile_range,
                "overall_level": overall_level,
            }
        ).sort_values(
            by=[
                "winter_minus_summer",
                "weekend_lift",
                "weekly_routine_strength",
                "normalized_profile_range",
            ],
            ascending=[False, False, False, False],
        )

        ordered_old_labels = ordering_table.index.tolist()
        relabel_map = {old_label: new_label for new_label, old_label in enumerate(ordered_old_labels)}

        relabeled = labels.copy()
        for old_label, new_label in relabel_map.items():
            relabeled.loc[numeric == old_label] = new_label

        relabeled = pd.to_numeric(relabeled, errors="coerce").astype(int)
        cluster_sizes = _build_cluster_size_table(relabeled)

        updated = ClusteringArtifacts(
            method=result.method,
            k=result.k,
            labels=relabeled,
            cluster_sizes=cluster_sizes,
            metrics=result.metrics.copy(),
            fitted_model=result.fitted_model,
            prepared_input=result.prepared_input,
            additional_outputs={
                **result.additional_outputs,
                "cluster_order_relabel_map": relabel_map,
                "cluster_label_map": result.additional_outputs.get("cluster_label_map"),
                "cluster_ordering_rule": {
                    1: "winter_minus_summer descending",
                    2: "weekend_lift descending",
                    3: "weekly_routine_strength descending",
                    4: "normalized_profile_range descending",
                    5: "overall_level descending",
                },
            },
        )
        return updated
    

    def prepare_clustering_handoff(
        self,
        data_2023: pd.DataFrame,
        result: Optional[ClusteringArtifacts] = None,
    ) -> Dict[str, object]:
        if result is None:
            raise ValueError(
                "prepare_clustering_handoff() requires an explicit clustering result "
                "because run_final_clustering() now requires a chosen k."
            )
        
        numeric_labels = pd.to_numeric(result.labels, errors="coerce")
        if numeric_labels.isna().any():
            raise ValueError(
                "prepare_clustering_handoff() requires a full-household clustering result "
                "with no unassigned households. Sampled benchmark results are not valid "
                "for forecasting handoff."
            )
        
        if "cluster_order_relabel_map" not in result.additional_outputs:
            result = self.order_final_clusters_by_profile(result, data_2023)
            
        reporter = ClusterReporter()

        forecasting_labels = self.assign_special_regime_households(
            result=result,
            data_2023=data_2023,
        )

        forecasting_cluster_sizes = (
            forecasting_labels.value_counts()
            .sort_index()
            .rename_axis("cluster")
            .reset_index(name="n_households")
        )
        forecasting_cluster_sizes["pct_households"] = (
            forecasting_cluster_sizes["n_households"] / forecasting_cluster_sizes["n_households"].sum() * 100.0
        )

        return {
            "forecasting_handoff_table": reporter.forecasting_ready_handoff_table(forecasting_labels),
            "cluster_sizes": forecasting_cluster_sizes,
            "diagnostic_cluster_sizes": reporter.cluster_size_summary(result),
            "monthly_profiles": reporter.mean_monthly_profile_by_cluster(data_2023, forecasting_labels),
            "weekly_profiles": reporter.mean_weekly_profile_by_cluster(data_2023, forecasting_labels),
            "cluster_label_map": result.additional_outputs.get("cluster_label_map"),
            "special_regime_summary": reporter.special_regime_assignment_summary(result),
            "special_regime_assignment_strategy": "nearest_cluster_profile",
        }
