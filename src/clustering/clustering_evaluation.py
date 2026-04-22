"""Evaluation utilities for the clustering experiment.

This module evaluates clustering results in a method-aware way and helps
choose k using a combination of:

- internal validity metrics
- cluster size balance
- minimum cluster size
- stability under resampling

It is designed to sit on top of `clustering.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.clustering.clustering import ClusteringArtifacts, HouseholdClusterer
from src.clustering.cluster_reporting import ClusterReporter
from src.utils.config import (
    K_SEARCH_VALUES,
    MIN_CLUSTER_SIZE,
    STABILITY_N_RUNS,
    STABILITY_SAMPLE_FRAC,
    USE_STABILITY_FOR_K_SELECTION,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Result containers
# =============================================================================

@dataclass
class StabilityArtifacts:
    """Container for resampling-based stability outputs."""
    method: str
    k: int
    scores: List[float]
    mean_score: Optional[float]
    std_score: Optional[float]
    successful_runs: int
    failed_runs: int
    details: Dict[str, object]


# =============================================================================
# Core helpers
# =============================================================================

def _normalize_higher_better(series: pd.Series) -> pd.Series:
    """Min-max normalize a metric where larger is better."""
    x = pd.to_numeric(series, errors="coerce")
    valid = x.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)

    xmin = valid.min()
    xmax = valid.max()
    if np.isclose(xmin, xmax):
        return pd.Series(1.0, index=series.index)

    result = (x - xmin) / (xmax - xmin)
    return result


def _normalize_lower_better(series: pd.Series) -> pd.Series:
    """Min-max normalize a metric where smaller is better."""
    x = pd.to_numeric(series, errors="coerce")
    valid = x.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)

    xmin = valid.min()
    xmax = valid.max()
    if np.isclose(xmin, xmax):
        return pd.Series(1.0, index=series.index)

    result = 1.0 - ((x - xmin) / (xmax - xmin))
    return result


def _extract_nonnegative_clusters(labels: pd.Series) -> pd.Series:
    """Drop special/fallback cluster labels such as -1 before evaluation."""
    numeric_labels = pd.to_numeric(labels, errors="coerce")
    valid_mask = numeric_labels >= 0
    return labels.loc[valid_mask]


# =============================================================================
# Evaluation class
# =============================================================================

class ClusteringEvaluator:
    """Evaluate clustering results and support k-selection."""

    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        self.clusterer = HouseholdClusterer(random_seed=random_seed)
        self.reporter = ClusterReporter()

    # -------------------------------------------------------------------------
    # Single-result evaluation
    # -------------------------------------------------------------------------

    def evaluate_single_result(
        self,
        result: ClusteringArtifacts,
        data_2023: Optional[pd.DataFrame] = None,
        min_cluster_size: int = MIN_CLUSTER_SIZE,
    ) -> Dict[str, object]:
        """Evaluate one clustering result in a method-agnostic way."""
        labels = _extract_nonnegative_clusters(result.labels)
        cluster_sizes = result.cluster_sizes.copy()

        # result.cluster_sizes should already exclude negative labels, but enforce it defensively
        if not cluster_sizes.empty:
            cluster_sizes = cluster_sizes[cluster_sizes["cluster"] >= 0].copy()

        has_small_clusters = False
        smallest_cluster = None
        largest_cluster = None

        if not cluster_sizes.empty:
            smallest_cluster = int(cluster_sizes["n_households"].min())
            largest_cluster = int(cluster_sizes["n_households"].max())
            has_small_clusters = smallest_cluster < min_cluster_size

        meaningfulness = {
            "cluster_mean_spread": None,
            "normalized_shape_separation": None,
            "level_to_shape_ratio": None,
            "level_dominance_warning": None,
        }

        if data_2023 is not None:
            meaningfulness = self.reporter.cluster_meaningfulness_diagnostics(
                data_2023=data_2023,
                labels=result.labels,
            )

        evaluation = {
            "method": result.method,
            "k": result.k,
            "n_households_regular_clusters": int(len(labels)),
            "n_clusters_found": int(cluster_sizes.shape[0]),
            "smallest_cluster": smallest_cluster,
            "largest_cluster": largest_cluster,
            "has_small_clusters": has_small_clusters,
            "cluster_balance": result.metrics.get("cluster_balance"),
            "silhouette": result.metrics.get("silhouette"),
            "davies_bouldin": result.metrics.get("davies_bouldin"),
            "calinski_harabasz": result.metrics.get("calinski_harabasz"),
            "cluster_mean_spread": meaningfulness.get("cluster_mean_spread"),
            "normalized_shape_separation": meaningfulness.get("normalized_shape_separation"),
            "level_to_shape_ratio": meaningfulness.get("level_to_shape_ratio"),
            "level_dominance_warning": meaningfulness.get("level_dominance_warning"),
        }

        return evaluation

    # -------------------------------------------------------------------------
    # Stability analysis
    # -------------------------------------------------------------------------

    def _run_method_for_stability(
        self,
        method: str,
        sampled_data: pd.DataFrame,
        k: int,
        **kwargs,
    ) -> ClusteringArtifacts:
        """Run one clustering method on one sampled subset for stability estimation."""
        if method == "feature_ward":
            prepared_bundle = self.clusterer.prepare_feature_ward_input(
                data_2023=sampled_data,
            )
            return self.clusterer.run_feature_ward_on_prepared(
                prepared_bundle=prepared_bundle,
                k=k,
            )

        if method == "kshape":
            kshape_bundle = self.clusterer.prepare_kshape_input(
                data_2023=sampled_data,
                max_households=kwargs.get("max_households"),
            )
            return self.clusterer.run_kshape_on_prepared(
                kshape_bundle=kshape_bundle,
                k=k,
                n_init=kwargs.get("n_init", 3),
                verbose=kwargs.get("verbose", False),
            )

        raise ValueError(f"Unknown clustering method: {method}")
    
    def estimate_stability(
        self,
        method: str,
        data_2023: pd.DataFrame,
        k: int,
        n_runs: int = STABILITY_N_RUNS,
        sample_frac: float = STABILITY_SAMPLE_FRAC,
        **kwargs,
    ) -> StabilityArtifacts:
        rng = np.random.default_rng(self.random_seed)
        all_ids = np.asarray(data_2023.index)
        n_households = len(all_ids)
        sample_size = max(k + 1, int(sample_frac * n_households))

        scores: List[float] = []
        failed_runs = 0

        sampled_label_runs: List[pd.Series] = []
        successful_run_count = 0

        for run_idx in range(n_runs):
            try:
                sampled_ids = rng.choice(all_ids, size=sample_size, replace=False)
                sampled_data = data_2023.loc[sampled_ids]

                sampled_result = self._run_method_for_stability(
                    method=method,
                    sampled_data=sampled_data,
                    k=k,
                    **kwargs,
                )
                sampled_labels = _extract_nonnegative_clusters(sampled_result.labels)
                sampled_label_runs.append(sampled_labels)
                successful_run_count += 1

            except Exception as exc:
                failed_runs += 1
                logger.exception(
                    "Stability run %d failed for %s k=%d: %s",
                    run_idx + 1,
                    method,
                    k,
                    exc,
                )

        for i in range(len(sampled_label_runs)):
            for j in range(i + 1, len(sampled_label_runs)):
                ids_i = sampled_label_runs[i].index
                ids_j = sampled_label_runs[j].index
                overlap_ids = ids_i.intersection(ids_j)
                if len(overlap_ids) < k:
                    continue
                ari = adjusted_rand_score(
                    sampled_label_runs[i].loc[overlap_ids].to_numpy(),
                    sampled_label_runs[j].loc[overlap_ids].to_numpy(),
                )
                scores.append(float(ari))
        
        n_pairwise_comparisons = len(scores)
        successful_runs = successful_run_count
        mean_score = float(np.mean(scores)) if scores else None
        std_score = float(np.std(scores)) if scores else None

        return StabilityArtifacts(
            method=method,
            k=k,
            scores=scores,
            mean_score=mean_score,
            std_score=std_score,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            details={
                "sample_frac": sample_frac,
                "n_runs": n_runs,
                "sample_size": int(sample_size),
                "n_pairwise_comparisons": n_pairwise_comparisons,
                "scope_note": (
                    "Stability is estimated on repeated subsamples of the original 2023 data. "
                    "For methods with internal exclusions or benchmark sampling, stability is conditional "
                    "on those method-specific preprocessing rules."
                ),
            },
        )

    # -------------------------------------------------------------------------
    # Ranking / selecting k
    # -------------------------------------------------------------------------

    def rank_k_candidates(
        self,
        evaluation_df: pd.DataFrame,
        min_cluster_size: int = MIN_CLUSTER_SIZE,
    ) -> pd.DataFrame:
        """Rank k candidates using a transparent composite score.

        Scoring logic:
        - higher silhouette is better
        - lower Davies-Bouldin is better
        - higher Calinski-Harabasz is better
        - higher cluster balance is better
        - higher stability is better
        - candidates with very small clusters are penalized
        - candidates flagged as overly level-dominated are penalized

        Important:
        This ranking is most appropriate within a single method.
        Cross-method comparison should rely mainly on forecasting utility and interpretation,
        because internal metrics are not fully comparable across different clustering objectives.
        """
        if "method" in evaluation_df.columns and evaluation_df["method"].nunique(dropna=True) > 1:
            logger.warning(
                "rank_k_candidates() received multiple methods. "
                "This ranking is intended primarily within a single method."
            )
        ranked = evaluation_df.copy()

        if "error" in ranked.columns:
            ranked = ranked[ranked["error"].isna()].copy()

        if ranked.empty:
            return ranked

        # Validity metrics
        ranked["score_silhouette"] = _normalize_higher_better(ranked["silhouette"])
        ranked["score_davies_bouldin"] = _normalize_lower_better(ranked["davies_bouldin"])
        ranked["score_calinski_harabasz"] = _normalize_higher_better(ranked["calinski_harabasz"])
        ranked["score_balance"] = _normalize_higher_better(ranked["cluster_balance"])
        ranked["score_stability"] = _normalize_higher_better(ranked["stability_mean"])

        # Penalty for small clusters
        ranked["size_penalty"] = np.where(
            pd.to_numeric(ranked["smallest_cluster"], errors="coerce") < min_cluster_size,
            1.0,
            0.0,
        )

        ranked["singleton_penalty"] = np.where(
            pd.to_numeric(ranked["smallest_cluster"], errors="coerce") <= 1,
            1.0,
            0.0,
        )

        ranked["extreme_imbalance_penalty"] = np.where(
            pd.to_numeric(ranked["cluster_balance"], errors="coerce") < 0.10,
            1.0,
            0.0,
        )

        ranked["level_dominance_penalty"] = np.where(
            ranked["level_dominance_warning"].fillna(False),
            1.0,
            0.0,
        )

        # Composite score
        component_cols = [
            "score_silhouette",
            "score_davies_bouldin",
            "score_calinski_harabasz",
            "score_balance",
            "score_stability",
        ]

        ranked["composite_score_raw"] = ranked[component_cols].mean(axis=1, skipna=True)
        ranked["n_available_components"] = ranked[component_cols].notna().sum(axis=1)
        ranked["low_evidence_penalty"] = np.where(
            ranked["n_available_components"] < 3,
            0.15,
            0.0,
        )
        
        ranked["composite_score"] = (
            ranked["composite_score_raw"]
            - 0.25 * ranked["size_penalty"]
            - 0.35 * ranked["singleton_penalty"]
            - 0.25 * ranked["extreme_imbalance_penalty"]
            - 0.20 * ranked["level_dominance_penalty"]
            - ranked["low_evidence_penalty"]
        )

        ranked = ranked.sort_values(
            by=["composite_score", "n_available_components", "k"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        return ranked

    def select_best_k(
        self,
        ranked_df: pd.DataFrame,
    ) -> Dict[str, object]:
        """Select the top-ranked k candidate from a vetted shortlist.

        This function assumes the caller has already filtered out structurally weak
        solutions using shortlist_reasonable_k(). It should not be used on the raw
        ranked table for final Task 1 selection.
        """
        if ranked_df.empty:
            raise ValueError("ranked_df is empty")
        
        working = ranked_df.copy()

        if "error" in working.columns:
            working = working[working["error"].isna()].copy()

        working = working[
            working["k"].notna()
            & working["method"].notna()
            & working["n_clusters_found"].notna()
        ].copy()

        if working.empty:
            raise ValueError("No valid k candidates available after filtering failed rows.")
        
        if "smallest_cluster" in working.columns:
            if pd.to_numeric(working.iloc[0]["smallest_cluster"], errors="coerce") <= 1:
                logger.warning(
                    "Top-ranked candidate includes a singleton cluster. "
                    "This should normally be rejected before final Task 1 selection."
                )
        
        if "cluster_balance" in working.columns:
            top_balance = pd.to_numeric(working.iloc[0]["cluster_balance"], errors="coerce")
            if pd.notna(top_balance) and top_balance < 0.10:
                logger.warning(
                    "Top-ranked candidate has extreme imbalance (cluster_balance < 0.10). "
                    "This should normally be rejected before final Task 1 selection."
                )
                
        top = working.iloc[0].to_dict()

        if top.get("cluster_balance") is not None and float(top["cluster_balance"]) < 0.15:
            logger.warning(
                "Selected top-ranked k=%s for method=%s still shows weak balance (balance=%s). "
                "Confirm this choice with cluster-profile interpretability checks before finalizing.",
                top.get("k"),
                top.get("method"),
                top.get("cluster_balance"),
            )
        return {
            "selected_k": int(top["k"]),
            "method": top["method"],
            "composite_score": top.get("composite_score"),
            "selection_row": top,
        }

    
    def shortlist_reasonable_k(
        self,
        ranked_df: pd.DataFrame,
        min_cluster_size: int = MIN_CLUSTER_SIZE,
        min_balance: float = 0.20,
    ) -> pd.DataFrame:
        """Keep only k values that are structurally reasonable before final selection.

        This is the main guardrail against choosing a k that looks good on a
        composite score but produces weakly supported or highly imbalanced clusters.
        """
        df = ranked_df.copy()

        smallest = pd.to_numeric(df["smallest_cluster"], errors="coerce")
        balance = pd.to_numeric(df["cluster_balance"], errors="coerce")

        mask = (
            smallest.ge(min_cluster_size)
            & balance.ge(min_balance)
            & pd.to_numeric(df["n_clusters_found"], errors="coerce").eq(
                pd.to_numeric(df["k"], errors="coerce")
            )
        )

        shortlisted = df.loc[mask].copy()
        return shortlisted.sort_values(
            by=["composite_score", "k"],
            ascending=[False, True],
        ).reset_index(drop=True)


    # -------------------------------------------------------------------------
    # Cluster interpretation helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def cluster_size_diagnostics(labels: pd.Series) -> Dict[str, object]:
        """Return compact cluster-size diagnostics."""
        labels = _extract_nonnegative_clusters(labels)
        counts = labels.value_counts().sort_index()

        if counts.empty:
            return {
                "n_clusters": 0,
                "min_size": None,
                "max_size": None,
                "median_size": None,
            }

        return {
            "n_clusters": int(counts.shape[0]),
            "min_size": int(counts.min()),
            "max_size": int(counts.max()),
            "median_size": float(counts.median()),
        }
