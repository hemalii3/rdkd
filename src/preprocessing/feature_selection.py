"""Feature Selection and Dimensionality Reduction Module.

This module provides functions for feature selection, correlation analysis,
and dimensionality reduction (PCA, t-SNE, UMAP).
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")


class FeatureSelector:
    """Feature selection and dimensionality reduction for clustering."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize feature selector.
        
        Args:
            random_seed: Random seed for reproducibility.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.selected_features = None
        self.pca_model = None
        self.scaler = None
    
    def calculate_correlations(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature correlation matrix.
        
        Args:
            features: DataFrame with features.
        
        Returns:
            Correlation matrix.
        """
        corr_matrix = features.corr()
        logger.info(f"Calculated correlations for {len(features.columns)} features")
        return corr_matrix
    
    def remove_highly_correlated(
        self, 
        features: pd.DataFrame, 
        threshold: float = 0.9
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with high correlation.
        
        Args:
            features: DataFrame with features.
            threshold: Correlation threshold (default: 0.9).
        
        Returns:
            Tuple of (filtered_features, removed_feature_names).
        """
        corr_matrix = features.corr().abs()
        
        # Find features to remove
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_corr = corr_matrix.where(upper_triangle)
        
        to_remove = [column for column in upper_corr.columns 
                    if any(upper_corr[column] > threshold)]
        
        # Keep features
        features_filtered = features.drop(columns=to_remove)
        
        self.selected_features = features_filtered.columns.tolist()
        
        logger.info(f"Removed {len(to_remove)} highly correlated features (threshold={threshold})")
        logger.info(f"Remaining features: {len(features_filtered.columns)}")
        
        return features_filtered, to_remove
    
    def select_by_variance(
        self,
        features: pd.DataFrame,
        threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove low-variance features.
        
        Args:
            features: DataFrame with features.
            threshold: Minimum variance threshold.
        
        Returns:
            Tuple of (filtered_features, removed_feature_names).
        """
        variances = features.var()
        to_keep = variances[variances > threshold].index.tolist()
        to_remove = variances[variances <= threshold].index.tolist()
        
        features_filtered = features[to_keep]
        
        logger.info(f"Removed {len(to_remove)} low-variance features (threshold={threshold})")
        logger.info(f"Remaining features: {len(features_filtered.columns)}")
        
        return features_filtered, to_remove
    
    def apply_pca(
        self,
        features: pd.DataFrame,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, PCA]:
        """Apply PCA for dimensionality reduction.
        
        Args:
            features: DataFrame with features.
            n_components: Number of components (None = auto-determine by variance).
            variance_threshold: Cumulative variance to explain (if n_components=None).
        
        Returns:
            Tuple of (pca_features, pca_model).
        """
        # Standardize features first
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        if n_components is None:
            # Determine n_components by variance threshold
            pca_temp = PCA(random_state=self.random_seed)
            pca_temp.fit(features_scaled)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=self.random_seed)
        features_pca = pca.fit_transform(features_scaled)
        
        # Create DataFrame
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        features_pca_df = pd.DataFrame(
            features_pca,
            index=features.index,
            columns=pca_columns
        )
        
        self.pca_model = pca
        self.scaler = scaler
        
        explained_var = pca.explained_variance_ratio_.sum()
        
        logger.info(f"PCA: {features.shape[1]} → {n_components} components")
        logger.info(f"Explained variance: {explained_var:.2%}")
        
        return features_pca_df, pca
    
    def apply_tsne(
        self,
        features: pd.DataFrame,
        n_components: int = 2,
        perplexity: float = 30.0
    ) -> pd.DataFrame:
        """Apply t-SNE for 2D visualization.
        
        Args:
            features: DataFrame with features.
            n_components: Number of dimensions (typically 2 or 3).
            perplexity: t-SNE perplexity parameter.
        
        Returns:
            DataFrame with t-SNE components.
        """
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=self.random_seed,
            n_jobs=-1
        )
        features_tsne = tsne.fit_transform(features_scaled)
        
        # Create DataFrame
        tsne_columns = [f'tSNE{i+1}' for i in range(n_components)]
        features_tsne_df = pd.DataFrame(
            features_tsne,
            index=features.index,
            columns=tsne_columns
        )
        
        logger.info(f"t-SNE: {features.shape[1]} → {n_components} components")
        
        return features_tsne_df
    
    def apply_umap(
        self,
        features: pd.DataFrame,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1
    ) -> pd.DataFrame:
        """Apply UMAP for 2D visualization.
        
        Args:
            features: DataFrame with features.
            n_components: Number of dimensions (typically 2 or 3).
            n_neighbors: Number of neighbors for UMAP.
            min_dist: Minimum distance for UMAP.
        
        Returns:
            DataFrame with UMAP components.
        """
        if not UMAP_AVAILABLE:
            logger.error("UMAP not available. Install with: pip install umap-learn")
            raise ImportError("UMAP not installed")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply UMAP
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=self.random_seed
        )
        features_umap = umap_model.fit_transform(features_scaled)
        
        # Create DataFrame
        umap_columns = [f'UMAP{i+1}' for i in range(n_components)]
        features_umap_df = pd.DataFrame(
            features_umap,
            index=features.index,
            columns=umap_columns
        )
        
        logger.info(f"UMAP: {features.shape[1]} → {n_components} components")
        
        return features_umap_df
    
    def get_feature_importance_pca(self, feature_names: List[str], top_n: int = 10) -> pd.DataFrame:
        """Get feature importance from PCA loadings.
        
        Args:
            feature_names: List of original feature names.
            top_n: Number of top features to return per component.
        
        Returns:
            DataFrame with feature importance.
        """
        if self.pca_model is None:
            raise ValueError("PCA model not fitted. Run apply_pca() first.")
        
        # Get loadings (components)
        loadings = self.pca_model.components_
        
        # Create DataFrame
        loading_df = pd.DataFrame(
            loadings.T,
            columns=[f'PC{i+1}' for i in range(loadings.shape[0])],
            index=feature_names
        )
        
        # Get top features for each PC
        importance_summary = {}
        for pc in loading_df.columns:
            abs_loadings = loading_df[pc].abs().sort_values(ascending=False)
            importance_summary[pc] = abs_loadings.head(top_n)
        
        logger.info(f"Computed feature importance for {loadings.shape[0]} PCs")
        
        return loading_df, importance_summary


def scale_features(features: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, object]:
    """Scale features using StandardScaler or MinMaxScaler.
    
    Args:
        features: DataFrame with features.
        method: 'standard' or 'minmax'.
    
    Returns:
        Tuple of (scaled_features, scaler).
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(
        features_scaled,
        index=features.index,
        columns=features.columns
    )
    
    logger.info(f"Features scaled using {method} scaler")
    return features_scaled_df, scaler


if __name__ == "__main__":
    # Test the module
    print("Testing Feature Selection Module...")
    
    # Create sample features
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    # Create some correlated features
    base_features = np.random.randn(n_samples, 10)
    correlated_features = base_features + np.random.randn(n_samples, 10) * 0.1
    random_features = np.random.randn(n_samples, 30)
    
    all_features = np.hstack([base_features, correlated_features, random_features])
    feature_names = [f'feature_{i}' for i in range(n_features)]
    features_df = pd.DataFrame(all_features, columns=feature_names)
    
    selector = FeatureSelector(random_seed=42)
    
    print("\n1. Testing correlation calculation...")
    corr = selector.calculate_correlations(features_df)
    print(f"Correlation matrix shape: {corr.shape}")
    
    print("\n2. Testing high correlation removal...")
    features_filtered, removed = selector.remove_highly_correlated(features_df, threshold=0.9)
    print(f"Original: {features_df.shape[1]} features")
    print(f"Filtered: {features_filtered.shape[1]} features")
    print(f"Removed: {len(removed)} features")
    
    print("\n3. Testing PCA...")
    features_pca, pca_model = selector.apply_pca(features_filtered, variance_threshold=0.95)
    print(f"PCA shape: {features_pca.shape}")
    print(f"Explained variance: {pca_model.explained_variance_ratio_.sum():.2%}")
    
    print("\n4. Testing t-SNE...")
    features_tsne = selector.apply_tsne(features_filtered, n_components=2)
    print(f"t-SNE shape: {features_tsne.shape}")
    
    print("\n5. Testing feature scaling...")
    features_scaled, scaler = scale_features(features_filtered, method='standard')
    print(f"Scaled features shape: {features_scaled.shape}")
    print(f"Mean: {features_scaled.mean().mean():.6f}")
    print(f"Std: {features_scaled.std().mean():.6f}")
    
    print("\n✅ All tests passed!")
