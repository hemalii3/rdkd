"""EDA Analysis Functions for Time Series Data.

This module provides helper functions for exploratory data analysis
of household energy consumption time series.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def compute_household_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for each household.
    
    Args:
        data: DataFrame with households as rows, days as columns.
    
    Returns:
        DataFrame with statistics for each household.
    
    Example:
        >>> stats = compute_household_statistics(train_data)
        >>> print(stats.columns)
        ['mean', 'median', 'std', 'min', 'max', 'range', 'q25', 'q75', 'cv', 'skew', 'kurtosis']
    """
    stats_df = pd.DataFrame(index=data.index)
    
    stats_df['mean'] = data.mean(axis=1)
    stats_df['median'] = data.median(axis=1)
    stats_df['std'] = data.std(axis=1)
    stats_df['min'] = data.min(axis=1)
    stats_df['max'] = data.max(axis=1)
    stats_df['range'] = stats_df['max'] - stats_df['min']
    stats_df['q25'] = data.quantile(0.25, axis=1)
    stats_df['q75'] = data.quantile(0.75, axis=1)
    
    # Coefficient of variation (CV = std/mean)
    stats_df['cv'] = stats_df['std'] / (stats_df['mean'] + 1e-10)
    
    # Skewness and kurtosis
    stats_df['skew'] = data.apply(lambda x: stats.skew(x), axis=1)
    stats_df['kurtosis'] = data.apply(lambda x: stats.kurtosis(x), axis=1)
    
    # Zero consumption count
    stats_df['zero_days'] = (data == 0).sum(axis=1)
    stats_df['zero_pct'] = (stats_df['zero_days'] / data.shape[1]) * 100
    
    logger.info(f"Computed statistics for {len(stats_df)} households")
    return stats_df


def detect_outlier_households(
    stats_df: pd.DataFrame, 
    method: str = 'iqr',
    threshold: float = 3.0
) -> Tuple[List[int], List[int]]:
    """Identify outlier households based on consumption patterns.
    
    Args:
        stats_df: DataFrame with household statistics.
        method: 'iqr' or 'zscore' for outlier detection.
        threshold: Threshold for outlier detection (IQR multiplier or Z-score).
    
    Returns:
        Tuple of (high_outliers, low_outliers) household IDs.
    """
    if method == 'iqr':
        Q1 = stats_df['mean'].quantile(0.25)
        Q3 = stats_df['mean'].quantile(0.75)
        IQR = Q3 - Q1
        
        high_outliers = stats_df[stats_df['mean'] > Q3 + threshold * IQR].index.tolist()
        low_outliers = stats_df[stats_df['mean'] < Q1 - threshold * IQR].index.tolist()
    
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(stats_df['mean']))
        outliers_mask = z_scores > threshold
        
        # Split into high and low based on mean
        median_consumption = stats_df['mean'].median()
        high_outliers = stats_df[(outliers_mask) & (stats_df['mean'] > median_consumption)].index.tolist()
        low_outliers = stats_df[(outliers_mask) & (stats_df['mean'] < median_consumption)].index.tolist()
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")
    
    logger.info(f"Detected {len(high_outliers)} high outliers and {len(low_outliers)} low outliers")
    return high_outliers, low_outliers


def aggregate_by_day_of_week(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate consumption by day of week.
    
    Args:
        data: DataFrame with households as rows, dates as columns.
    
    Returns:
        DataFrame with 7 columns (Mon-Sun) and households as rows.
    """
    # Parse column names as dates
    dates = pd.to_datetime(data.columns)
    
    # Create DataFrame with day of week
    day_data = {}
    for day_num in range(7):
        day_cols = [col for col, date in zip(data.columns, dates) if date.dayofweek == day_num]
        day_data[day_num] = data[day_cols].mean(axis=1)
    
    dow_df = pd.DataFrame(day_data)
    dow_df.columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    return dow_df


def aggregate_by_month(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate consumption by month.
    
    Args:
        data: DataFrame with households as rows, dates as columns.
    
    Returns:
        DataFrame with 12 columns (Jan-Dec) and households as rows.
    """
    dates = pd.to_datetime(data.columns)
    
    month_data = {}
    for month_num in range(1, 13):
        month_cols = [col for col, date in zip(data.columns, dates) if date.month == month_num]
        if month_cols:  # Only if we have data for this month
            month_data[month_num] = data[month_cols].mean(axis=1)
    
    month_df = pd.DataFrame(month_data)
    month_df.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(month_df.columns)]
    
    return month_df


def detect_seasonality(data: pd.DataFrame, household_id: int) -> Dict[str, float]:
    """Analyze seasonality for a specific household.
    
    Args:
        data: DataFrame with households as rows, dates as columns.
        household_id: ID of the household to analyze.
    
    Returns:
        Dictionary with seasonality metrics.
    """
    series = data.loc[household_id]
    dates = pd.to_datetime(data.columns)
    
    # Split into seasons (Northern Hemisphere)
    winter_months = [12, 1, 2]
    spring_months = [3, 4, 5]
    summer_months = [6, 7, 8]
    fall_months = [9, 10, 11]
    
    winter_vals = [series[col] for col, date in zip(data.columns, dates) if date.month in winter_months]
    spring_vals = [series[col] for col, date in zip(data.columns, dates) if date.month in spring_months]
    summer_vals = [series[col] for col, date in zip(data.columns, dates) if date.month in summer_months]
    fall_vals = [series[col] for col, date in zip(data.columns, dates) if date.month in fall_months]
    
    seasonality_metrics = {
        'winter_mean': np.mean(winter_vals) if winter_vals else 0,
        'spring_mean': np.mean(spring_vals) if spring_vals else 0,
        'summer_mean': np.mean(summer_vals) if summer_vals else 0,
        'fall_mean': np.mean(fall_vals) if fall_vals else 0,
        'winter_summer_ratio': np.mean(winter_vals) / (np.mean(summer_vals) + 1e-10) if winter_vals and summer_vals else 1,
        'seasonal_amplitude': max([np.mean(x) for x in [winter_vals, spring_vals, summer_vals, fall_vals] if x]) - 
                             min([np.mean(x) for x in [winter_vals, spring_vals, summer_vals, fall_vals] if x])
    }
    
    return seasonality_metrics


def compute_autocorrelation(data: pd.DataFrame, household_id: int, lags: List[int] = [1, 7, 30]) -> Dict[int, float]:
    """Compute autocorrelation for specific lags.
    
    Args:
        data: DataFrame with households as rows, dates as columns.
        household_id: ID of the household to analyze.
        lags: List of lags to compute autocorrelation for.
    
    Returns:
        Dictionary mapping lag to autocorrelation value.
    """
    series = data.loc[household_id].values
    
    acf_values = {}
    for lag in lags:
        if lag < len(series):
            # Compute Pearson correlation between series and lagged series
            acf_values[lag] = np.corrcoef(series[:-lag], series[lag:])[0, 1]
        else:
            acf_values[lag] = np.nan
    
    return acf_values


def categorize_households(stats_df: pd.DataFrame) -> pd.Series:
    """Categorize households into preliminary groups based on consumption.
    
    Args:
        stats_df: DataFrame with household statistics.
    
    Returns:
        Series with category labels for each household.
    """
    categories = pd.Series(index=stats_df.index, dtype=str)
    
    # Define thresholds based on quantiles
    low_threshold = stats_df['mean'].quantile(0.33)
    high_threshold = stats_df['mean'].quantile(0.67)
    
    high_cv_threshold = stats_df['cv'].quantile(0.75)
    high_zero_threshold = 20  # More than 20% zero days
    
    for hh_id in stats_df.index:
        mean_consumption = stats_df.loc[hh_id, 'mean']
        cv = stats_df.loc[hh_id, 'cv']
        zero_pct = stats_df.loc[hh_id, 'zero_pct']
        
        # High zero consumption
        if zero_pct > high_zero_threshold:
            categories[hh_id] = 'Intermittent/Vacant'
        # High variability
        elif cv > high_cv_threshold:
            if mean_consumption > high_threshold:
                categories[hh_id] = 'High Variable'
            elif mean_consumption < low_threshold:
                categories[hh_id] = 'Low Variable'
            else:
                categories[hh_id] = 'Medium Variable'
        # Stable consumption
        else:
            if mean_consumption > high_threshold:
                categories[hh_id] = 'High Stable'
            elif mean_consumption < low_threshold:
                categories[hh_id] = 'Low Stable'
            else:
                categories[hh_id] = 'Medium Stable'
    
    logger.info(f"Categorized households: {categories.value_counts().to_dict()}")
    return categories


def detect_trend(data: pd.DataFrame, household_id: int) -> Dict[str, float]:
    """Detect linear trend in time series.
    
    Args:
        data: DataFrame with households as rows, dates as columns.
        household_id: ID of the household to analyze.
    
    Returns:
        Dictionary with trend metrics (slope, intercept, r_value).
    """
    series = data.loc[household_id].values
    x = np.arange(len(series))
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'trend_direction': 'increasing' if slope > 0.01 else ('decreasing' if slope < -0.01 else 'stable')
    }


if __name__ == "__main__":
    # Test the module
    from src.utils.data_loader import load_train_data
    
    print("Testing EDA analysis functions...")
    data = load_train_data()
    
    print("\n1. Computing household statistics...")
    stats = compute_household_statistics(data)
    print(stats.head())
    
    print("\n2. Detecting outliers...")
    high_out, low_out = detect_outlier_households(stats)
    print(f"High outliers: {len(high_out)}, Low outliers: {len(low_out)}")
    
    print("\n3. Day of week aggregation...")
    dow = aggregate_by_day_of_week(data)
    print(dow.head())
    
    print("\n4. Monthly aggregation...")
    monthly = aggregate_by_month(data)
    print(monthly.head())
    
    print("\n5. Categorizing households...")
    categories = categorize_households(stats)
    print(categories.value_counts())
