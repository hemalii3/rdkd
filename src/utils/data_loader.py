"""Data loading utilities for the project.

This module provides functions to load and perform basic validation
on the household energy consumption datasets.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

from src.utils.config import (
    TRAIN_PATH,
    TEST_PATH,
    TRAIN_DAYS,
    TEST_DAYS,
)

logger = logging.getLogger(__name__)

def validate_datetime_columns(
    data: pd.DataFrame,
    *,
    expected_days: Optional[int] = None,
    expected_start: Optional[str] = None,
    expected_end: Optional[str] = None,
) -> pd.DatetimeIndex:
    """Validate that columns are parseable, unique, ordered, and optionally exact."""
    try:
        dates = pd.to_datetime(data.columns)
    except Exception as e:
        raise ValueError(f"Could not parse date columns: {e}") from e

    if dates.has_duplicates:
        raise ValueError("Date columns contain duplicates.")

    if not dates.is_monotonic_increasing:
        raise ValueError("Date columns are not sorted in increasing order.")

    if expected_days is not None and len(dates) != expected_days:
        raise ValueError(f"Expected {expected_days} date columns, got {len(dates)}.")

    if expected_start is not None and dates[0] != pd.Timestamp(expected_start):
        raise ValueError(
            f"Expected first date {expected_start}, got {dates[0].date()}."
        )

    if expected_end is not None and dates[-1] != pd.Timestamp(expected_end):
        raise ValueError(
            f"Expected last date {expected_end}, got {dates[-1].date()}."
        )

    return dates
    
def load_train_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    filepath = filepath or TRAIN_PATH
    
    logger.info(f"Loading training data from {filepath}")
    
    try:
        data = pd.read_csv(filepath, index_col=0)
        logger.info("Loaded %d households with %d days", data.shape[0], data.shape[1])

        if data.shape[1] != TRAIN_DAYS:
            raise ValueError(f"Expected {TRAIN_DAYS} days, got {data.shape[1]}")

        if data.index.duplicated().any():
            n_dupes = int(data.index.duplicated().sum())
            raise ValueError(f"Found {n_dupes} duplicate household IDs in training data.")

        validate_datetime_columns(
            data,
            expected_days=TRAIN_DAYS,
            expected_start="2023-01-01",
            expected_end="2023-12-31",
        )

        return data
    
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise

def load_test_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    filepath = filepath or TEST_PATH
    
    logger.info(f"Loading test data from {filepath}")
    
    try:
        data = pd.read_csv(filepath, index_col=0)
        logger.info("Loaded %d households with %d days", data.shape[0], data.shape[1])

        if data.shape[1] != TEST_DAYS:
            raise ValueError(f"Expected {TEST_DAYS} days (leap year), got {data.shape[1]}")

        if data.index.duplicated().any():
            n_dupes = int(data.index.duplicated().sum())
            raise ValueError(f"Found {n_dupes} duplicate household IDs in test data.")

        validate_datetime_columns(
            data,
            expected_days=TEST_DAYS,
            expected_start="2024-01-01",
            expected_end="2024-12-31",
        )

        return data
    
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise


def validate_data(data: pd.DataFrame, year: int = 2023) -> dict:
    if year == 2023:
        validate_datetime_columns(
            data,
            expected_days=TRAIN_DAYS,
            expected_start="2023-01-01",
            expected_end="2023-12-31",
        )
    elif year == 2024:
        validate_datetime_columns(
            data,
            expected_days=TEST_DAYS,
            expected_start="2024-01-01",
            expected_end="2024-12-31",
        )
    else:
        raise ValueError(f"Unsupported year: {year}")
    
    results = {
        "n_households": int(data.shape[0]),
        "n_days": int(data.shape[1]),
        "missing_values": int(data.isna().sum().sum()),
        "negative_values": int((data < 0).sum().sum()),
        "zero_values": int((data == 0).sum().sum()),
        "duplicate_household_ids": int(data.index.duplicated().sum()),
        "min_consumption": float(data.min().min()),
        "max_consumption": float(data.max().max()),
    }
    
    logger.info("Data validation for %s:", year)
    logger.info("  - Households: %d", results["n_households"])
    logger.info("  - Days: %d", results["n_days"])
    logger.info("  - Missing values: %d", results["missing_values"])
    logger.info("  - Zero values: %d", results["zero_values"])
    logger.info("  - Negative values: %d", results["negative_values"])
    logger.info("  - Min consumption: %.4f", results["min_consumption"])
    logger.info("  - Max consumption: %.4f", results["max_consumption"])
    logger.info("  - Duplicate household IDs: %d", results["duplicate_household_ids"])
        
    return results

def validate_train_test_household_alignment(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Validate that 2023 and 2024 contain the same household IDs."""
    train_ids = pd.Index(train_data.index)
    test_ids = pd.Index(test_data.index)

    if not train_ids.equals(test_ids):
        missing_in_test = train_ids.difference(test_ids).tolist()
        missing_in_train = test_ids.difference(train_ids).tolist()
        raise ValueError(
            "Train/test household IDs do not align. "
            f"Missing in test: {missing_in_test[:10]}; "
            f"missing in train: {missing_in_train[:10]}"
        )

if __name__ == "__main__":
    train_data = load_train_data()
    test_data = load_test_data()
    
    train_stats = validate_data(train_data, year=2023)
    test_stats = validate_data(test_data, year=2024)
    validate_train_test_household_alignment(train_data, test_data)