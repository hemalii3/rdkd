"""Feature selection and dimensionality reduction for Method 1 clustering.

This module is intentionally focused on the feature-based clustering pipeline.
It prepares household-level feature tables for Ward hierarchical clustering by:

1. removing low-variance features
2. removing highly correlated features
3. sanitizing non-finite values
4. scaling retained features
5. optionally applying PCA
6. preserving interpretable feature tables for later cluster explanation

t-SNE and UMAP are kept as optional visualization helpers only.
"""

from __future__ import annotations

import logging
from typing import cast
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.config import (
    FEATURE_CLUSTERING_SCALE_METHOD,
    FEATURE_LOW_VARIANCE_THRESHOLD,
    FEATURE_PCA_VARIANCE_THRESHOLD,
    FEATURE_CORRELATION_THRESHOLD,
    FEATURE_USE_PCA,
    RANDOM_SEED,
    USE_FEATURE_BLOCK_WEIGHTS,
    FEATURE_BLOCK_WEIGHTS,
)

logger = logging.getLogger(__name__)

class FeatureSelector:
    """Prepare Method 1 features for clustering."""

    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.selected_feature_names: List[str] = []
        self.low_variance_removed_: List[str] = []
        self.correlated_removed_: List[str] = []
        self.nan_imputed_columns_: List[str] = []
        self.inf_replaced_columns_: List[str] = []

        self.scaler = None
        self.pca_model = None

        self.filtered_features_: Optional[pd.DataFrame] = None
        self.feature_block_map_: Optional[Dict[str, List[str]]] = None
        self.scaled_features_: Optional[pd.DataFrame] = None
        self.weighted_features_: Optional[pd.DataFrame] = None
        self.clustering_input_: Optional[pd.DataFrame] = None
        self.fill_values_: Optional[pd.Series] = None

    # -------------------------------------------------------------------------
    # Basic utilities
    # -------------------------------------------------------------------------

    def sanitize_features(
        self,
        features: pd.DataFrame,
        fill_strategy: str = "median",
        fill_values: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """Replace inf values and impute NaNs before scaling / PCA."""
        X = features.copy()

        inf_mask = np.isinf(X.to_numpy(dtype=float))
        col_has_inf = np.asarray(inf_mask.any(axis=0), dtype=bool).reshape(-1)
        inf_columns = [col for col, has_inf in zip(X.columns, col_has_inf) if has_inf]

        if bool(inf_mask.any()):
            X = X.replace([np.inf, -np.inf], np.nan)

        nan_columns_before = X.columns[X.isna().any(axis=0)].tolist()

        if fill_values is None:
            if fill_strategy == "median":
                fill_values = X.median(axis=0, numeric_only=True)
            elif fill_strategy == "mean":
                fill_values = X.mean(axis=0, numeric_only=True)
            elif fill_strategy == "zero":
                fill_values = pd.Series(0.0, index=X.columns)
            else:
                raise ValueError(f"Unknown fill_strategy: {fill_strategy}")

        fill_values = fill_values.reindex(X.columns).fillna(0.0)
        X = X.fillna(fill_values)
        X = X.fillna(0.0)

        if X.isna().sum().sum() > 0:
            raise ValueError("sanitize_features failed to remove all NaNs.")

        self.nan_imputed_columns_ = nan_columns_before
        self.inf_replaced_columns_ = inf_columns

        summary = {
            "nan_imputed_columns": nan_columns_before,
            "n_nan_imputed_columns": len(nan_columns_before),
            "inf_replaced_columns": inf_columns,
            "n_inf_replaced_columns": len(inf_columns),
            "fill_strategy": fill_strategy,
        }

        logger.info(
            "Sanitized features: %d columns had NaNs, %d columns had infs",
            len(nan_columns_before),
            len(inf_columns),
        )
        return X, summary

    def select_by_variance(
        self,
        features: pd.DataFrame,
        threshold: float = FEATURE_LOW_VARIANCE_THRESHOLD,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove near-constant features using a variance threshold."""
        variances = features.var()
        to_keep = variances[variances > threshold].index.tolist()
        to_remove = variances[variances <= threshold].index.tolist()

        filtered = features[to_keep].copy()

        self.low_variance_removed_ = to_remove
        logger.info(
            "Removed %d low-variance features (threshold=%g). Remaining=%d",
            len(to_remove),
            threshold,
            filtered.shape[1],
        )
        return filtered, to_remove
    
    def remove_highly_correlated_block_aware(
        self,
        features: pd.DataFrame,
        threshold: float = FEATURE_CORRELATION_THRESHOLD,
        protected_prefixes: Optional[List[str]] = None,
        protected_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features while preserving interpretable profile blocks.

        Protected prefixes are intended for structured profile blocks such as
        full weekday or month-level profiles, where correlation is expected and
        interpretability is more important than aggressive pruning.
        """
        if protected_prefixes is None:
            protected_prefixes = ["dow_", "month_"]
        if protected_columns is None:
            protected_columns = []

        protected_set = set(protected_columns)
        for col in features.columns:
            if any(col.startswith(prefix) for prefix in protected_prefixes):
                protected_set.add(col)

        corr_matrix = features.corr().abs()

        to_remove = set()
        columns = features.columns.tolist()

        for i, col_i in enumerate(columns):
            if col_i in to_remove:
                continue
            for j in range(i + 1, len(columns)):
                col_j = columns[j]
                if col_j in to_remove:
                    continue

                corr_ij_raw = corr_matrix.loc[col_i, col_j]
                corr_ij = pd.to_numeric(pd.Series([corr_ij_raw]), errors="coerce").iloc[0]

                if pd.isna(corr_ij) or float(corr_ij) <= threshold:
                    continue

                if col_i in protected_set and col_j in protected_set:
                    continue
                if col_i in protected_set:
                    to_remove.add(col_j)
                    continue
                if col_j in protected_set:
                    to_remove.add(col_i)
                    break

                mean_i_raw = corr_matrix[col_i].drop(labels=[col_i]).mean()
                mean_j_raw = corr_matrix[col_j].drop(labels=[col_j]).mean()

                mean_i = float(pd.to_numeric(pd.Series([mean_i_raw]), errors="coerce").iloc[0])
                mean_j = float(pd.to_numeric(pd.Series([mean_j_raw]), errors="coerce").iloc[0])

                if mean_i >= mean_j:
                    to_remove.add(col_i)
                    break
                else:
                    to_remove.add(col_j)

        to_remove = sorted(to_remove)

        filtered = features.drop(columns=to_remove).copy()
        self.correlated_removed_ = to_remove
        self.selected_feature_names = filtered.columns.tolist()

        logger.info(
            "Removed %d highly correlated features with block-aware protection. Remaining=%d",
            len(to_remove),
            filtered.shape[1],
        )
        return filtered, to_remove

    def scale_features(
        self,
        features: pd.DataFrame,
        method: str = FEATURE_CLUSTERING_SCALE_METHOD,
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, object]:
        """Scale features using StandardScaler or MinMaxScaler."""
        if method == "standard":
            scaler = self.scaler if (self.scaler is not None and not fit) else StandardScaler()
        elif method == "minmax":
            scaler = self.scaler if (self.scaler is not None and not fit) else MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        if fit:
            scaled = scaler.fit_transform(features)
            self.scaler = scaler
        else:
            if self.scaler is None:
                raise ValueError("No fitted scaler available. Run with fit=True first.")
            scaled = self.scaler.transform(features)

        scaled_df = pd.DataFrame(
            scaled,
            index=features.index,
            columns=features.columns,
        )

        logger.info(
            "Scaled %d features using %s scaling",
            features.shape[1],
            method,
        )
        return scaled_df, scaler
    
    def apply_feature_block_weights(
        self,
        scaled_features: pd.DataFrame,
        feature_block_map: Dict[str, List[str]],
        block_weights: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Apply heuristic block weights after scaling and before PCA/clustering.

        These weights are a modeling choice used to moderate the influence of
        different feature families; they are not learned from data and should be
        reported transparently in the notebook/report.
        """
        weighted = scaled_features.copy()

        for block_name, columns in feature_block_map.items():
            weight = float(block_weights.get(block_name, 1.0))
            existing_cols = [col for col in columns if col in weighted.columns]
            if existing_cols:
                weighted.loc[:, existing_cols] = weighted[existing_cols] * weight
            missing_from_scaled = [col for col in columns if col not in weighted.columns]
            if missing_from_scaled:
                logger.debug(
                    "Block '%s' has %d columns not present after filtering: %s",
                    block_name,
                    len(missing_from_scaled),
                    missing_from_scaled,
                )
        logger.info("Applied feature block weights to scaled features")
        return weighted
    
    def rebalance_feature_blocks(
        self,
        scaled_features: pd.DataFrame,
        feature_block_map: Dict[str, List[str]],
    ) -> pd.DataFrame:
        """Downweight large feature blocks so they do not dominate by column count alone.

        Each block is divided by sqrt(number of retained columns in that block).
        This keeps the effective contribution of large structured blocks closer
        to that of small blocks before heuristic block weighting is applied.
        """
        rebalanced = scaled_features.copy()

        for block_name, columns in feature_block_map.items():
            existing_cols = [col for col in columns if col in rebalanced.columns]
            if not existing_cols:
                continue

            block_size = len(existing_cols)
            rebalanced.loc[:, existing_cols] = rebalanced[existing_cols] / np.sqrt(block_size)

        logger.info("Applied block-size rebalancing to scaled features")
        return rebalanced
    
    def summarize_retained_block_counts(self) -> pd.DataFrame:
        """Return number of retained columns per feature block after filtering."""
        if self.filtered_features_ is None or self.feature_block_map_ is None:
            raise ValueError("No fitted feature pipeline/block map available.")

        kept = set(self.filtered_features_.columns)
        rows = []
        for block_name, cols in self.feature_block_map_.items():
            retained = [col for col in cols if col in kept]
            rows.append(
                {
                    "block": block_name,
                    "n_original": len(cols),
                    "n_retained": len(retained),
                    "retained_columns": ", ".join(retained),
                }
            )
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # PCA
    # -------------------------------------------------------------------------

    def apply_pca(
        self,
        features_scaled: pd.DataFrame,
        n_components: Optional[int] = None,
        variance_threshold: float = FEATURE_PCA_VARIANCE_THRESHOLD,
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, PCA]:
        """Apply PCA to a scaled feature matrix."""
        if fit:
            if n_components is None:
                pca_temp = PCA(random_state=self.random_seed)
                pca_temp.fit(features_scaled)
                cumulative = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = int(np.argmax(cumulative >= variance_threshold) + 1)

            pca = PCA(n_components=n_components, random_state=self.random_seed)
            transformed = pca.fit_transform(features_scaled)
            self.pca_model = pca
        else:
            if self.pca_model is None:
                raise ValueError("No fitted PCA model available. Run with fit=True first.")
            pca = self.pca_model
            transformed = pca.transform(features_scaled)

        pca_columns = [f"PC{i+1}" for i in range(transformed.shape[1])]
        pca_df = pd.DataFrame(
            transformed,
            index=features_scaled.index,
            columns=pca_columns,
        )

        explained_variance = float(pca.explained_variance_ratio_.sum())
        logger.info(
            "Applied PCA: %d -> %d dimensions (explained variance = %.2f%%)",
            features_scaled.shape[1],
            pca_df.shape[1],
            100 * explained_variance,
        )
        return pca_df, pca

    def get_pca_loadings(self) -> pd.DataFrame:
        """Return PCA loading matrix for the last fitted PCA model."""
        if self.pca_model is None or self.filtered_features_ is None:
            raise ValueError(
                "PCA model or filtered features not available. "
                "Run prepare_feature_method_input() first."
            )

        loadings = pd.DataFrame(
            self.pca_model.components_.T,
            index=self.filtered_features_.columns,
            columns=[f"PC{i+1}" for i in range(self.pca_model.components_.shape[0])],
        )
        return loadings

    def get_top_pca_features(self, top_n: int = 10) -> Dict[str, pd.Series]:
        """Return top absolute-loading features for each principal component."""
        loadings = self.get_pca_loadings()
        top_features: Dict[str, pd.Series] = {}

        for pc in loadings.columns:
            top_features[pc] = loadings[pc].abs().sort_values(ascending=False).head(top_n)

        return top_features
    
    # -------------------------------------------------------------------------
    # Main Method 1 preparation pipeline
    # -------------------------------------------------------------------------

    def prepare_feature_method_input(
        self,
        features: pd.DataFrame,
        feature_block_map: Optional[Dict[str, List[str]]] = None,
        variance_threshold: float = FEATURE_LOW_VARIANCE_THRESHOLD,
        correlation_threshold: float = FEATURE_CORRELATION_THRESHOLD,
        scale_method: str = FEATURE_CLUSTERING_SCALE_METHOD,
        use_pca: Optional[bool] = None,
        pca_variance_threshold: float = FEATURE_PCA_VARIANCE_THRESHOLD,
        pca_n_components: Optional[int] = None,
        nan_fill_strategy: str = "median",
        protected_prefixes: Optional[List[str]] = None,
        protected_columns: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """Prepare feature table for Method 1 clustering.

        This implementation is currently specialized for the shape-first Ward
        pipeline used in Task 1. In particular, normalized weekday/month blocks
        are protected during correlation pruning for interpretability reasons.
        """
        original_features = features.copy()

        if use_pca is None:
            use_pca = FEATURE_USE_PCA

        if nan_fill_strategy == "median":
            self.fill_values_ = original_features.replace([np.inf, -np.inf], np.nan).median(axis=0, numeric_only=True)
        elif nan_fill_strategy == "mean":
            self.fill_values_ = original_features.replace([np.inf, -np.inf], np.nan).mean(axis=0, numeric_only=True)
        elif nan_fill_strategy == "zero":
            self.fill_values_ = pd.Series(0.0, index=original_features.columns)
        else:
            raise ValueError(f"Unknown nan_fill_strategy: {nan_fill_strategy}")

        sanitized_features, sanitize_summary = self.sanitize_features(
            original_features,
            fill_strategy=nan_fill_strategy,
            fill_values=self.fill_values_,
        )

        # Step 1: remove low-variance features
        filtered_features, low_var_removed = self.select_by_variance(
            sanitized_features,
            threshold=variance_threshold,
        )

        features_after_low_variance = filtered_features.copy()
        if filtered_features.shape[1] == 0:
            raise ValueError(
                "All features were removed by the low-variance filter. "
                "Lower FEATURE_LOW_VARIANCE_THRESHOLD or revise the feature set."
            )
        
        # Step 2: remove highly correlated features
        if protected_prefixes is None:
            protected_prefixes = ["norm_dow_", "norm_month_"]

        if protected_columns is None:
            protected_columns = [
                # normalized-shape summaries
                "norm_monthly_profile_std",
                "norm_monthly_profile_range",
                "norm_winter_summer_diff",
                "norm_weekend_weekday_diff",
                "norm_q1_mean",
                "norm_q2_mean",
                "norm_q3_mean",
                "norm_q4_mean",
                "norm_spring_minus_winter",
                "norm_summer_minus_spring",
                "norm_autumn_minus_summer",
                "norm_winter_minus_autumn",
                "norm_peak_month_sin",
                "norm_peak_month_cos",
                "norm_trough_month_sin",
                "norm_trough_month_cos",
                "norm_month_entropy",
                "norm_month_diff_abs_mean",
                "norm_month_diff_std",
                "norm_dow_range",
                "norm_dow_diff_abs_mean",

                # dependence / persistence
                "autocorr_lag_7",
                "autocorr_lag_1",
                "autocorr_lag_30",
                "rolling_std_30",
                "diff_std",
                "abs_diff_mean",
                "trend_slope_normalized",

                # small secondary raw-profile block
                "base_load_ratio",
                "seasonal_peak_ratio",
                "seasonal_range_ratio",
                "winter_share",
                "summer_share",
                "monthly_entropy",
            ]

        protected_columns = list(dict.fromkeys(protected_columns))

        filtered_features, correlated_removed = self.remove_highly_correlated_block_aware(
            filtered_features,
            threshold=correlation_threshold,
            protected_prefixes=protected_prefixes,
            protected_columns=protected_columns,
        )

        self.filtered_features_ = filtered_features.copy()
        if filtered_features.shape[1] == 0:
            raise ValueError(
                "All features were removed by the correlation filter. "
                "Raise FEATURE_CORRELATION_THRESHOLD or revise protected feature blocks."
            )
        self.feature_block_map_ = feature_block_map.copy() if feature_block_map is not None else None

        # Step 3: scale features
        scaled_features, scaler = self.scale_features(
            filtered_features,
            method=scale_method,
            fit=True,
        )
        self.scaled_features_ = scaled_features.copy()
        weighted_features = scaled_features.copy()

        if feature_block_map is not None:
            weighted_features = self.rebalance_feature_blocks(
                scaled_features=weighted_features,
                feature_block_map=feature_block_map,
            )

        if USE_FEATURE_BLOCK_WEIGHTS and feature_block_map is not None:
            weighted_features = self.apply_feature_block_weights(
                scaled_features=weighted_features,
                feature_block_map=feature_block_map,
                block_weights=FEATURE_BLOCK_WEIGHTS,
            )

        self.weighted_features_ = weighted_features.copy()

        # Safety check before PCA
        arr = weighted_features.to_numpy(dtype=float)
        finite_all = np.isfinite(arr).all()

        if not bool(finite_all):
            col_finite_mask = np.asarray(np.isfinite(arr).all(axis=0), dtype=bool)
            bad_cols = [col for col, is_finite in zip(scaled_features.columns, col_finite_mask) if not is_finite]
            raise ValueError(
                f"Scaled feature matrix still contains non-finite values. Bad columns: {bad_cols}"
            )
        # Step 4: optional PCA
        if use_pca:
            clustering_input, pca_model = self.apply_pca(
                weighted_features,
                n_components=pca_n_components,
                variance_threshold=pca_variance_threshold,
                fit=True,
            )
        else:
            clustering_input = weighted_features.copy()
            pca_model = None
            self.pca_model = None

        self.clustering_input_ = clustering_input.copy()

        summary = {
            "n_original_features": int(original_features.shape[1]),
            "n_after_low_variance_filter": int(features_after_low_variance.shape[1]),
            "n_after_correlation_filter": int(filtered_features.shape[1]),
            "n_final_clustering_dimensions": int(clustering_input.shape[1]),
            "low_variance_removed": low_var_removed,
            "correlated_removed": correlated_removed,
            "selected_feature_names": filtered_features.columns.tolist(),
            "scale_method": scale_method,
            "used_pca": bool(use_pca),
            "pca_explained_variance": (
                float(pca_model.explained_variance_ratio_.sum()) if pca_model is not None else None
            ),
            "pca_n_components": (
                int(pca_model.n_components_) if pca_model is not None else None
            ),
            "sanitize_summary": sanitize_summary,
            "used_feature_block_weights": bool(USE_FEATURE_BLOCK_WEIGHTS and feature_block_map is not None),
            "retained_feature_blocks": (
                self.get_retained_feature_blocks() if feature_block_map is not None else None
            ),
            "feature_block_weights": FEATURE_BLOCK_WEIGHTS if (USE_FEATURE_BLOCK_WEIGHTS and feature_block_map is not None) else None,
            "retained_block_counts": (
                self.summarize_retained_block_counts().to_dict(orient="records")
                if feature_block_map is not None else None
            ),
        }

        logger.info(
            "Prepared Method 1 clustering input: original=%d, selected=%d, final_dims=%d",
            original_features.shape[1],
            filtered_features.shape[1],
            clustering_input.shape[1],
        )

        return {
            "filtered_features": filtered_features,
            "clustering_input": clustering_input,
            "selected_feature_names": filtered_features.columns.tolist(),
            "scaler": scaler,
            "pca_model": pca_model,
            "summary": summary,
        }


    # -------------------------------------------------------------------------
    # Transform-only utilities for inference / reuse
    # -------------------------------------------------------------------------

    def transform_new_features(
        self,
        features: pd.DataFrame,
        apply_pca: Optional[bool] = None,
    ) -> Dict[str, object]:
        """Transform a new feature table using the fitted Method 1 pipeline."""
        if self.filtered_features_ is None:
            raise ValueError(
                "No fitted feature pipeline available. "
                "Run prepare_feature_method_input() first."
            )

        if self.scaler is None:
            raise ValueError("No fitted scaler available.")

        features_sanitized, _ = self.sanitize_features(
            features,
            fill_values=self.fill_values_,
        )

        selected_columns = self.filtered_features_.columns.tolist()
        missing_cols = [col for col in selected_columns if col not in features_sanitized.columns]
        if missing_cols:
            raise ValueError(
                f"New feature table is missing required columns: {missing_cols}"
            )

        features_selected = features_sanitized[selected_columns].copy()
        if isinstance(self.scaler, StandardScaler):
            fitted_scale_method = "standard"
        elif isinstance(self.scaler, MinMaxScaler):
            fitted_scale_method = "minmax"
        else:
            raise ValueError(f"Unsupported fitted scaler type: {type(self.scaler).__name__}")

        scaled_features, _ = self.scale_features(
            features_selected,
            method=fitted_scale_method,
            fit=False,
        )

        weighted_features = scaled_features.copy()
        if self.feature_block_map_ is not None:
            weighted_features = self.rebalance_feature_blocks(
                scaled_features=weighted_features,
                feature_block_map=self.feature_block_map_,
            )

        if USE_FEATURE_BLOCK_WEIGHTS and self.feature_block_map_ is not None:
            weighted_features = self.apply_feature_block_weights(
                scaled_features=weighted_features,
                feature_block_map=self.feature_block_map_,
                block_weights=FEATURE_BLOCK_WEIGHTS,
            )

        if apply_pca is None:
            apply_pca = self.pca_model is not None

        if apply_pca:
            clustering_input, _ = self.apply_pca(
                weighted_features,
                fit=False,
            )
        else:
            clustering_input = weighted_features.copy()

        return {
            "filtered_features": features_selected,
            "scaled_features": scaled_features,
            "weighted_features": weighted_features,
            "clustering_input": clustering_input,
        }

    # -------------------------------------------------------------------------
    # Reporting helpers
    # -------------------------------------------------------------------------

    def summarize_selection(self) -> pd.DataFrame:
        """Return a compact summary of the fitted selection pipeline."""
        rows = [
            {"stage": "nan_imputed_columns", "count": len(self.nan_imputed_columns_)},
            {"stage": "inf_replaced_columns", "count": len(self.inf_replaced_columns_)},
            {"stage": "low_variance_removed", "count": len(self.low_variance_removed_)},
            {"stage": "correlated_removed", "count": len(self.correlated_removed_)},
            {
                "stage": "selected_features",
                "count": len(self.selected_feature_names),
            },
            {
                "stage": "pca_components",
                "count": (
                    int(self.pca_model.n_components_)
                    if self.pca_model is not None else len(self.selected_feature_names)
                ),
            },
        ]
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Optional visualization helpers
    # -------------------------------------------------------------------------

    def apply_tsne(
        self,
        features: pd.DataFrame,
        n_components: int = 2,
        perplexity: float = 30.0,
    ) -> pd.DataFrame:
        """Apply t-SNE for visualization only."""
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=self.random_seed,
            n_jobs=-1,
        )
        transformed = tsne.fit_transform(features_scaled)

        tsne_columns = [f"tSNE{i+1}" for i in range(n_components)]
        tsne_df = pd.DataFrame(
            transformed,
            index=features.index,
            columns=tsne_columns,
        )

        logger.info(
            "Applied t-SNE for visualization: %d -> %d dims",
            features.shape[1],
            n_components,
        )
        return tsne_df

    def get_retained_feature_blocks(self) -> Dict[str, List[str]]:
        """Return retained columns by feature block after filtering."""
        if self.filtered_features_ is None or self.feature_block_map_ is None:
            raise ValueError("No fitted feature pipeline/block map available.")

        kept = set(self.filtered_features_.columns)
        retained = {}
        for block_name, cols in self.feature_block_map_.items():
            retained[block_name] = [col for col in cols if col in kept]
        return retained
