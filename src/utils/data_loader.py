"""Data loading utilities for the project.

This module provides functions to load and perform basic validation
on the household energy consumption datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

from src.utils.config import TRAIN_PATH, TEST_PATH, N_HOUSEHOLDS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_train_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the 2023 training dataset.
    
    Args:
        filepath: Path to the CSV file. If None, uses default from config.
    
    Returns:
        DataFrame with household IDs as index and dates as columns.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the data has unexpected shape.
    
    Example:
        >>> data = load_train_data()
        >>> print(data.shape)
        (17548, 365)
    """
    filepath = filepath or TRAIN_PATH
    
    logger.info(f"Loading training data from {filepath}")
    
    try:
        data = pd.read_csv(filepath, index_col=0)
        logger.info(f"Loaded {data.shape[0]} households with {data.shape[1]} days")
        
        # Validate shape
        if data.shape[1] != 365:
            logger.warning(f"Expected 365 days, got {data.shape[1]}")
        
        return data
    
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise


def load_test_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the 2024 testing dataset.
    
    Args:
        filepath: Path to the CSV file. If None, uses default from config.
    
    Returns:
        DataFrame with household IDs as index and dates as columns.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the data has unexpected shape.
    
    Example:
        >>> data = load_test_data()
        >>> print(data.shape)
        (17548, 366)
    """
    filepath = filepath or TEST_PATH
    
    logger.info(f"Loading test data from {filepath}")
    
    try:
        data = pd.read_csv(filepath, index_col=0)
        logger.info(f"Loaded {data.shape[0]} households with {data.shape[1]} days")
        
        # Validate shape
        if data.shape[1] != 366:
            logger.warning(f"Expected 366 days (leap year), got {data.shape[1]}")
        
        return data
    
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise


def load_both_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both training and testing datasets.
    
    Returns:
        Tuple of (train_data, test_data) DataFrames.
    
    Example:
        >>> train, test = load_both_datasets()
        >>> print(f"Train: {train.shape}, Test: {test.shape}")
        Train: (17548, 365), Test: (17548, 366)
    """
    train_data = load_train_data()
    test_data = load_test_data()
    
    # Verify household IDs match
    if not train_data.index.equals(test_data.index):
        logger.warning("Household IDs don't match between train and test!")
    
    return train_data, test_data


def validate_data(data: pd.DataFrame, year: int = 2023) -> dict:
    """Perform basic data quality checks.
    
    Args:
        data: DataFrame to validate.
        year: Year of the data (2023 or 2024).
    
    Returns:
        Dictionary with validation results.
    
    Example:
        >>> data = load_train_data()
        >>> results = validate_data(data, year=2023)
        >>> print(results['missing_values'])
    """
    results = {
        'n_households': data.shape[0],
        'n_days': data.shape[1],
        'missing_values': data.isna().sum().sum(),
        'missing_pct': (data.isna().sum().sum() / data.size) * 100,
        'zero_values': (data == 0).sum().sum(),
        'negative_values': (data < 0).sum().sum(),
        'mean_consumption': data.mean().mean(),
        'median_consumption': data.median().median(),
        'min_consumption': data.min().min(),
        'max_consumption': data.max().max(),
    }
    
    logger.info(f"Data validation for {year}:")
    logger.info(f"  - Households: {results['n_households']}")
    logger.info(f"  - Days: {results['n_days']}")
    logger.info(f"  - Missing values: {results['missing_values']} ({results['missing_pct']:.2f}%)")
    logger.info(f"  - Zero values: {results['zero_values']}")
    logger.info(f"  - Negative values: {results['negative_values']}")
    logger.info(f"  - Mean consumption: {results['mean_consumption']:.2f}")
    
    return results


def get_household_series(data: pd.DataFrame, household_id: int) -> pd.Series:
    """Extract time series for a specific household.
    
    Args:
        data: DataFrame with household data.
        household_id: ID of the household to extract.
    
    Returns:
        Series with daily consumption values.
    
    Raises:
        KeyError: If household_id not found.
    
    Example:
        >>> data = load_train_data()
        >>> series = get_household_series(data, household_id=22)
        >>> print(series.shape)
        (365,)
    """
    if household_id not in data.index:
        raise KeyError(f"Household {household_id} not found in dataset")
    
    return data.loc[household_id]


if __name__ == "__main__":
    # Test the module
    print("Testing data loading...")
    
    train, test = load_both_datasets()
    print(f"\nTrain shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    print("\nValidating training data...")
    train_validation = validate_data(train, year=2023)
    
    print("\nValidating test data...")
    test_validation = validate_data(test, year=2024)
    
    print("\nSample household (ID=22):")
    sample = get_household_series(train, 22)
    print(sample.describe())
