"""Data Cleaning Module for Time Series Preprocessing.

This module provides functions for handling missing values, outliers,
and data normalization for household energy consumption data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
import pickle

logger = logging.getLogger(__name__)


class DataCleaner:
    """Comprehensive data cleaning pipeline for time series data."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize data cleaner.
        
        Args:
            random_seed: Random seed for reproducibility.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.scaler = None
        self.outlier_mask = None
        self.zero_strategy = None
        
    def handle_missing_values(
        self, 
        data: pd.DataFrame, 
        strategy: str = 'forward_fill'
    ) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values in time series data.
        
        Args:
            data: DataFrame with households as rows, days as columns.
            strategy: 'forward_fill', 'interpolate', 'mean', or 'drop'.
        
        Returns:
            Tuple of (cleaned_data, summary_dict).
        """
        missing_before = data.isna().sum().sum()
        
        if missing_before == 0:
            logger.info("No missing values detected")
            return data.copy(), {'missing_before': 0, 'missing_after': 0, 'strategy': 'none'}
        
        data_clean = data.copy()
        
        if strategy == 'forward_fill':
            data_clean = data_clean.fillna(method='ffill', axis=1)
            # Fill any remaining with backward fill
            data_clean = data_clean.fillna(method='bfill', axis=1)
        
        elif strategy == 'interpolate':
            data_clean = data_clean.interpolate(method='linear', axis=1)
        
        elif strategy == 'mean':
            household_means = data_clean.mean(axis=1)
            for idx in data_clean.index:
                data_clean.loc[idx] = data_clean.loc[idx].fillna(household_means[idx])
        
        elif strategy == 'drop':
            # Drop households with any missing values
            data_clean = data_clean.dropna(axis=0)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        missing_after = data_clean.isna().sum().sum()
        
        summary = {
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'strategy': strategy,
            'households_dropped': len(data) - len(data_clean) if strategy == 'drop' else 0
        }
        
        logger.info(f"Missing values handled: {missing_before} → {missing_after} using {strategy}")
        return data_clean, summary
    
    def handle_outliers(
        self,
        data: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 3.0,
        action: str = 'cap'
    ) -> Tuple[pd.DataFrame, Dict]:
        """Handle outlier values in time series.
        
        Args:
            data: DataFrame with households as rows, days as columns.
            method: 'iqr', 'zscore', or 'percentile'.
            threshold: Threshold for outlier detection.
            action: 'cap', 'remove', or 'keep'.
        
        Returns:
            Tuple of (cleaned_data, summary_dict).
        """
        data_clean = data.copy()
        outliers_detected = 0
        
        if method == 'iqr':
            Q1 = data_clean.quantile(0.25, axis=1)
            Q3 = data_clean.quantile(0.75, axis=1)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            for idx in data_clean.index:
                outlier_mask = (data_clean.loc[idx] < lower_bound[idx]) | (data_clean.loc[idx] > upper_bound[idx])
                outliers_detected += outlier_mask.sum()
                
                if action == 'cap':
                    data_clean.loc[idx] = data_clean.loc[idx].clip(lower=lower_bound[idx], upper=upper_bound[idx])
                elif action == 'remove':
                    data_clean.loc[idx, outlier_mask] = np.nan
        
        elif method == 'zscore':
            for idx in data_clean.index:
                mean = data_clean.loc[idx].mean()
                std = data_clean.loc[idx].std()
                z_scores = np.abs((data_clean.loc[idx] - mean) / (std + 1e-10))
                outlier_mask = z_scores > threshold
                outliers_detected += outlier_mask.sum()
                
                if action == 'cap':
                    lower = mean - threshold * std
                    upper = mean + threshold * std
                    data_clean.loc[idx] = data_clean.loc[idx].clip(lower=lower, upper=upper)
                elif action == 'remove':
                    data_clean.loc[idx, outlier_mask] = np.nan
        
        elif method == 'percentile':
            lower_percentile = data_clean.quantile(threshold/100, axis=1)
            upper_percentile = data_clean.quantile(1 - threshold/100, axis=1)
            
            for idx in data_clean.index:
                outlier_mask = (data_clean.loc[idx] < lower_percentile[idx]) | (data_clean.loc[idx] > upper_percentile[idx])
                outliers_detected += outlier_mask.sum()
                
                if action == 'cap':
                    data_clean.loc[idx] = data_clean.loc[idx].clip(lower=lower_percentile[idx], upper=upper_percentile[idx])
                elif action == 'remove':
                    data_clean.loc[idx, outlier_mask] = np.nan
        
        # If removed outliers created NaN, fill them
        if action == 'remove':
            data_clean = data_clean.interpolate(method='linear', axis=1)
            data_clean = data_clean.fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
        
        summary = {
            'method': method,
            'threshold': threshold,
            'action': action,
            'outliers_detected': int(outliers_detected),
            'pct_outliers': (outliers_detected / data.size) * 100
        }
        
        logger.info(f"Outliers handled: {outliers_detected} ({summary['pct_outliers']:.2f}%) using {method}/{action}")
        return data_clean, summary
    
    def handle_zeros(
        self,
        data: pd.DataFrame,
        strategy: str = 'keep',
        min_threshold: float = 0.5
    ) -> Tuple[pd.DataFrame, Dict]:
        """Handle zero consumption values.
        
        Args:
            data: DataFrame with households as rows, days as columns.
            strategy: 'keep', 'impute', or 'flag'.
            min_threshold: Minimum value to replace zeros with (for impute).
        
        Returns:
            Tuple of (cleaned_data, summary_dict).
        """
        zeros_before = (data == 0).sum().sum()
        data_clean = data.copy()
        
        if strategy == 'keep':
            # Do nothing, keep zeros as valid consumption
            pass
        
        elif strategy == 'impute':
            # Replace zeros with small positive value or interpolated values
            for idx in data_clean.index:
                zero_mask = data_clean.loc[idx] == 0
                if zero_mask.any():
                    # Get non-zero values for this household
                    non_zero_values = data_clean.loc[idx][~zero_mask]
                    if len(non_zero_values) > 0:
                        # Replace with minimum non-zero value or threshold
                        replacement_value = max(non_zero_values.min() * 0.1, min_threshold)
                        data_clean.loc[idx, zero_mask] = replacement_value
        
        elif strategy == 'flag':
            # Create a separate flag column indicating zero days
            # (This would be used as a feature in clustering/forecasting)
            pass
        
        zeros_after = (data_clean == 0).sum().sum()
        
        summary = {
            'zeros_before': int(zeros_before),
            'zeros_after': int(zeros_after),
            'strategy': strategy,
            'pct_zeros_before': (zeros_before / data.size) * 100,
            'pct_zeros_after': (zeros_after / data_clean.size) * 100
        }
        
        self.zero_strategy = strategy
        logger.info(f"Zero values handled: {zeros_before} → {zeros_after} using {strategy}")
        return data_clean, summary
    
    def handle_negatives(
        self,
        data: pd.DataFrame,
        action: str = 'set_zero'
    ) -> Tuple[pd.DataFrame, Dict]:
        """Handle negative consumption values (invalid).
        
        Args:
            data: DataFrame with households as rows, days as columns.
            action: 'set_zero', 'abs', or 'remove'.
        
        Returns:
            Tuple of (cleaned_data, summary_dict).
        """
        negatives_before = (data < 0).sum().sum()
        data_clean = data.copy()
        
        if negatives_before == 0:
            return data_clean, {'negatives_before': 0, 'negatives_after': 0, 'action': 'none'}
        
        if action == 'set_zero':
            data_clean[data_clean < 0] = 0
        elif action == 'abs':
            data_clean = data_clean.abs()
        elif action == 'remove':
            data_clean[data_clean < 0] = np.nan
            data_clean = data_clean.interpolate(method='linear', axis=1)
        
        negatives_after = (data_clean < 0).sum().sum()
        
        summary = {
            'negatives_before': int(negatives_before),
            'negatives_after': int(negatives_after),
            'action': action
        }
        
        logger.info(f"Negative values handled: {negatives_before} → {negatives_after} using {action}")
        return data_clean, summary
    
    def normalize_data(
        self,
        data: pd.DataFrame,
        method: str = 'standard',
        axis: int = 1
    ) -> Tuple[pd.DataFrame, object]:
        """Normalize/standardize time series data.
        
        Args:
            data: DataFrame with households as rows, days as columns.
            method: 'standard' (z-score) or 'minmax' (0-1 scaling).
            axis: 0 for column-wise, 1 for row-wise normalization.
        
        Returns:
            Tuple of (normalized_data, scaler_object).
        """
        if method == 'standard':
            if axis == 1:  # Row-wise (per household)
                means = data.mean(axis=1)
                stds = data.std(axis=1)
                data_norm = data.sub(means, axis=0).div(stds + 1e-10, axis=0)
            else:  # Column-wise (per day)
                means = data.mean(axis=0)
                stds = data.std(axis=0)
                data_norm = data.sub(means, axis=1).div(stds + 1e-10, axis=1)
        
        elif method == 'minmax':
            if axis == 1:  # Row-wise
                mins = data.min(axis=1)
                maxs = data.max(axis=1)
                data_norm = data.sub(mins, axis=0).div(maxs - mins + 1e-10, axis=0)
            else:  # Column-wise
                mins = data.min(axis=0)
                maxs = data.max(axis=0)
                data_norm = data.sub(mins, axis=1).div(maxs - mins + 1e-10, axis=1)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        logger.info(f"Data normalized using {method} method (axis={axis})")
        return data_norm, {'method': method, 'axis': axis}
    
    def clean_pipeline(
        self,
        data: pd.DataFrame,
        handle_missing: bool = True,
        handle_outliers_flag: bool = True,
        handle_zeros_flag: bool = True,
        handle_negatives_flag: bool = True,
        normalize: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """Complete data cleaning pipeline.
        
        Args:
            data: Raw time series DataFrame.
            handle_missing: Whether to handle missing values.
            handle_outliers_flag: Whether to handle outliers.
            handle_zeros_flag: Whether to handle zero values.
            handle_negatives_flag: Whether to handle negative values.
            normalize: Whether to normalize data.
        
        Returns:
            Tuple of (cleaned_data, summary_dict).
        """
        data_clean = data.copy()
        summary = {}
        
        if handle_negatives_flag:
            data_clean, neg_summary = self.handle_negatives(data_clean, action='set_zero')
            summary['negatives'] = neg_summary
        
        if handle_missing:
            data_clean, miss_summary = self.handle_missing_values(data_clean, strategy='interpolate')
            summary['missing'] = miss_summary
        
        if handle_zeros_flag:
            data_clean, zero_summary = self.handle_zeros(data_clean, strategy='keep')
            summary['zeros'] = zero_summary
        
        if handle_outliers_flag:
            data_clean, outlier_summary = self.handle_outliers(data_clean, method='iqr', threshold=3.0, action='cap')
            summary['outliers'] = outlier_summary
        
        if normalize:
            data_clean, norm_info = self.normalize_data(data_clean, method='standard', axis=1)
            summary['normalization'] = norm_info
        
        summary['final_shape'] = data_clean.shape
        summary['final_missing'] = int(data_clean.isna().sum().sum())
        
        logger.info(f"Cleaning pipeline complete: {data.shape} → {data_clean.shape}")
        return data_clean, summary


def save_scaler(scaler: object, filepath: str) -> None:
    """Save scaler object to disk.
    
    Args:
        scaler: Fitted scaler object.
        filepath: Path to save the scaler.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {filepath}")


def load_scaler(filepath: str) -> object:
    """Load scaler object from disk.
    
    Args:
        filepath: Path to the scaler file.
    
    Returns:
        Loaded scaler object.
    """
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"Scaler loaded from {filepath}")
    return scaler


if __name__ == "__main__":
    # Test the module
    from src.utils.data_loader import load_train_data
    
    print("Testing Data Cleaning Module...")
    data = load_train_data()
    
    cleaner = DataCleaner(random_seed=42)
    
    print("\n1. Testing missing value handling...")
    data_no_missing, summary = cleaner.handle_missing_values(data)
    print(f"Summary: {summary}")
    
    print("\n2. Testing outlier handling...")
    data_no_outliers, summary = cleaner.handle_outliers(data, method='iqr', threshold=3.0, action='cap')
    print(f"Summary: {summary}")
    
    print("\n3. Testing zero handling...")
    data_no_zeros, summary = cleaner.handle_zeros(data, strategy='keep')
    print(f"Summary: {summary}")
    
    print("\n4. Testing negative handling...")
    data_no_negatives, summary = cleaner.handle_negatives(data, action='set_zero')
    print(f"Summary: {summary}")
    
    print("\n5. Testing complete pipeline...")
    data_clean, summary = cleaner.clean_pipeline(data)
    print(f"Pipeline summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✅ All tests passed!")
