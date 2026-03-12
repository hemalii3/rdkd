"""Feature Engineering Module for Time Series Clustering.

This module extracts comprehensive features from household energy
consumption time series for clustering analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats, signal
from scipy.fft import fft
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Comprehensive feature extraction for time series clustering."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize feature engineer.
        
        Args:
            random_seed: Random seed for reproducibility.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.feature_names = []
    
    def extract_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features from time series.
        
        Args:
            data: DataFrame with households as rows, days as columns.
        
        Returns:
            DataFrame with statistical features per household.
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic statistics
        features['mean'] = data.mean(axis=1)
        features['median'] = data.median(axis=1)
        features['std'] = data.std(axis=1)
        features['min'] = data.min(axis=1)
        features['max'] = data.max(axis=1)
        features['range'] = features['max'] - features['min']
        
        # Quantiles
        features['q25'] = data.quantile(0.25, axis=1)
        features['q50'] = data.quantile(0.50, axis=1)
        features['q75'] = data.quantile(0.75, axis=1)
        features['iqr'] = features['q75'] - features['q25']
        
        # Moments
        features['skewness'] = data.apply(lambda x: stats.skew(x), axis=1)
        features['kurtosis'] = data.apply(lambda x: stats.kurtosis(x), axis=1)
        
        # Coefficient of variation
        features['cv'] = features['std'] / (features['mean'] + 1e-10)
        
        # Zero consumption
        features['zero_count'] = (data == 0).sum(axis=1)
        features['zero_pct'] = (features['zero_count'] / data.shape[1]) * 100
        
        # Variance
        features['variance'] = data.var(axis=1)
        
        logger.info(f"Extracted {len(features.columns)} statistical features")
        return features
    
    def extract_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from time series.
        
        Args:
            data: DataFrame with households as rows, date columns.
        
        Returns:
            DataFrame with temporal features.
        """
        features = pd.DataFrame(index=data.index)
        dates = pd.to_datetime(data.columns)
        
        # Day-of-week averages (Mon=0, Sun=6)
        for day_num in range(7):
            day_cols = [col for col, date in zip(data.columns, dates) if date.dayofweek == day_num]
            if day_cols:
                features[f'dow_{day_num}_mean'] = data[day_cols].mean(axis=1)
        
        # Weekday vs weekend
        weekday_cols = [col for col, date in zip(data.columns, dates) if date.dayofweek < 5]
        weekend_cols = [col for col, date in zip(data.columns, dates) if date.dayofweek >= 5]
        
        features['weekday_mean'] = data[weekday_cols].mean(axis=1) if weekday_cols else 0
        features['weekend_mean'] = data[weekend_cols].mean(axis=1) if weekend_cols else 0
        features['weekend_weekday_ratio'] = features['weekend_mean'] / (features['weekday_mean'] + 1e-10)
        
        # Monthly averages (Jan=1, Dec=12)
        for month_num in range(1, 13):
            month_cols = [col for col, date in zip(data.columns, dates) if date.month == month_num]
            if month_cols:
                features[f'month_{month_num}_mean'] = data[month_cols].mean(axis=1)
        
        # Quarterly averages
        for quarter in range(1, 5):
            quarter_months = [(quarter-1)*3 + i for i in range(1, 4)]
            quarter_cols = [col for col, date in zip(data.columns, dates) if date.month in quarter_months]
            if quarter_cols:
                features[f'quarter_{quarter}_mean'] = data[quarter_cols].mean(axis=1)
        
        # Trend coefficient (linear regression slope)
        features['trend_slope'] = data.apply(
            lambda row: stats.linregress(np.arange(len(row)), row.values)[0], 
            axis=1
        )
        
        logger.info(f"Extracted {len(features.columns)} temporal features")
        return features
    
    def extract_seasonality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract seasonality features from time series.
        
        Args:
            data: DataFrame with households as rows, date columns.
        
        Returns:
            DataFrame with seasonality features.
        """
        features = pd.DataFrame(index=data.index)
        dates = pd.to_datetime(data.columns)
        
        # Winter months (Dec, Jan, Feb)
        winter_months = [12, 1, 2]
        winter_cols = [col for col, date in zip(data.columns, dates) if date.month in winter_months]
        
        # Summer months (Jun, Jul, Aug)
        summer_months = [6, 7, 8]
        summer_cols = [col for col, date in zip(data.columns, dates) if date.month in summer_months]
        
        # Spring months (Mar, Apr, May)
        spring_months = [3, 4, 5]
        spring_cols = [col for col, date in zip(data.columns, dates) if date.month in spring_months]
        
        # Fall months (Sep, Oct, Nov)
        fall_months = [9, 10, 11]
        fall_cols = [col for col, date in zip(data.columns, dates) if date.month in fall_months]
        
        features['winter_mean'] = data[winter_cols].mean(axis=1) if winter_cols else 0
        features['summer_mean'] = data[summer_cols].mean(axis=1) if summer_cols else 0
        features['spring_mean'] = data[spring_cols].mean(axis=1) if spring_cols else 0
        features['fall_mean'] = data[fall_cols].mean(axis=1) if fall_cols else 0
        
        # Seasonal ratios
        features['winter_summer_ratio'] = features['winter_mean'] / (features['summer_mean'] + 1e-10)
        features['seasonal_amplitude'] = (
            features[['winter_mean', 'summer_mean', 'spring_mean', 'fall_mean']].max(axis=1) -
            features[['winter_mean', 'summer_mean', 'spring_mean', 'fall_mean']].min(axis=1)
        )
        
        # Seasonal decomposition features (sample-based due to computational cost)
        # We'll compute for a sample and use aggregate metrics
        try:
            sample_series = data.iloc[0].values
            if len(sample_series) >= 14:  # Need at least 2 periods for weekly seasonality
                decomposition = seasonal_decompose(sample_series, model='additive', period=7, extrapolate_trend='freq')
                # Use variance of seasonal component as a feature
                features['seasonal_strength'] = data.apply(
                    lambda row: np.var(seasonal_decompose(row.values, model='additive', period=7, extrapolate_trend='freq').seasonal) 
                    if len(row) >= 14 else 0, 
                    axis=1
                )
        except Exception as e:
            logger.warning(f"Seasonal decomposition failed: {e}")
            features['seasonal_strength'] = 0
        
        logger.info(f"Extracted {len(features.columns)} seasonality features")
        return features
    
    def extract_variability_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract variability features from time series.
        
        Args:
            data: DataFrame with households as rows, days as columns.
        
        Returns:
            DataFrame with variability features.
        """
        features = pd.DataFrame(index=data.index)
        
        # Rolling window standard deviation
        features['rolling_std_7'] = data.rolling(window=7, axis=1).std().mean(axis=1)
        features['rolling_std_30'] = data.rolling(window=30, axis=1).std().mean(axis=1)
        
        # Number of peaks and valleys
        features['num_peaks'] = data.apply(
            lambda row: len(signal.find_peaks(row.values)[0]), 
            axis=1
        )
        features['num_valleys'] = data.apply(
            lambda row: len(signal.find_peaks(-row.values)[0]), 
            axis=1
        )
        
        # Autocorrelation at different lags
        for lag in [1, 7, 30]:
            features[f'autocorr_lag_{lag}'] = data.apply(
                lambda row: pd.Series(row.values).autocorr(lag=lag) if len(row) > lag else 0,
                axis=1
            )
        
        # Difference statistics (day-to-day changes)
        diffs = data.diff(axis=1)
        features['diff_mean'] = diffs.mean(axis=1)
        features['diff_std'] = diffs.std(axis=1)
        features['diff_max'] = diffs.max(axis=1)
        features['diff_min'] = diffs.min(axis=1)
        
        logger.info(f"Extracted {len(features.columns)} variability features")
        return features
    
    def extract_shape_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract shape-based features from time series.
        
        Args:
            data: DataFrame with households as rows, days as columns.
        
        Returns:
            DataFrame with shape features.
        """
        features = pd.DataFrame(index=data.index)
        
        # Time series entropy (Shannon entropy)
        def calculate_entropy(series):
            """Calculate Shannon entropy of time series."""
            # Discretize into bins
            hist, _ = np.histogram(series, bins=20)
            hist = hist / hist.sum()  # Normalize
            hist = hist[hist > 0]  # Remove zeros
            return -np.sum(hist * np.log2(hist))
        
        features['entropy'] = data.apply(lambda row: calculate_entropy(row.values), axis=1)
        
        # Spectral features (FFT)
        def extract_spectral_features(series):
            """Extract spectral features using FFT."""
            fft_vals = np.abs(fft(series))
            fft_vals = fft_vals[:len(fft_vals)//2]  # Take positive frequencies
            
            # Dominant frequency magnitude
            dominant_freq_mag = np.max(fft_vals)
            
            # Spectral centroid
            freqs = np.fft.fftfreq(len(series))[:len(series)//2]
            spectral_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-10)
            
            # Spectral rolloff (95% of energy)
            cumsum = np.cumsum(fft_vals)
            rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            
            return dominant_freq_mag, spectral_centroid, spectral_rolloff
        
        spectral = data.apply(lambda row: extract_spectral_features(row.values), axis=1)
        features['spectral_dominant_freq'] = spectral.apply(lambda x: x[0])
        features['spectral_centroid'] = spectral.apply(lambda x: x[1])
        features['spectral_rolloff'] = spectral.apply(lambda x: x[2])
        
        # Hurst exponent (approximation using rescaled range)
        def hurst_exponent(series):
            """Calculate Hurst exponent (simplified)."""
            try:
                lags = range(2, min(20, len(series)//2))
                tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
                
                if len(tau) > 1 and all(t > 0 for t in tau):
                    poly = np.polyfit(np.log(lags), np.log(tau), 1)
                    return poly[0]
                else:
                    return 0.5  # Neutral value
            except:
                return 0.5
        
        features['hurst_exponent'] = data.apply(lambda row: hurst_exponent(row.values), axis=1)
        
        # Linearity (R² of linear fit)
        features['linearity'] = data.apply(
            lambda row: stats.linregress(np.arange(len(row)), row.values)[2]**2,
            axis=1
        )
        
        logger.info(f"Extracted {len(features.columns)} shape features")
        return features
    
    def extract_all_features(
        self, 
        data: pd.DataFrame,
        include_statistical: bool = True,
        include_temporal: bool = True,
        include_seasonality: bool = True,
        include_variability: bool = True,
        include_shape: bool = True
    ) -> pd.DataFrame:
        """Extract all features from time series.
        
        Args:
            data: DataFrame with households as rows, days as columns.
            include_statistical: Include statistical features.
            include_temporal: Include temporal features.
            include_seasonality: Include seasonality features.
            include_variability: Include variability features.
            include_shape: Include shape features.
        
        Returns:
            DataFrame with all extracted features.
        """
        all_features = pd.DataFrame(index=data.index)
        
        if include_statistical:
            stat_features = self.extract_statistical_features(data)
            all_features = pd.concat([all_features, stat_features], axis=1)
        
        if include_temporal:
            temp_features = self.extract_temporal_features(data)
            all_features = pd.concat([all_features, temp_features], axis=1)
        
        if include_seasonality:
            season_features = self.extract_seasonality_features(data)
            all_features = pd.concat([all_features, season_features], axis=1)
        
        if include_variability:
            var_features = self.extract_variability_features(data)
            all_features = pd.concat([all_features, var_features], axis=1)
        
        if include_shape:
            shape_features = self.extract_shape_features(data)
            all_features = pd.concat([all_features, shape_features], axis=1)
        
        self.feature_names = all_features.columns.tolist()
        
        logger.info(f"Extracted {len(all_features.columns)} total features for {len(all_features)} households")
        return all_features


if __name__ == "__main__":
    # Test the module
    from src.utils.data_loader import load_train_data
    
    print("Testing Feature Engineering Module...")
    data = load_train_data()
    
    # Test on a small sample for speed
    sample_data = data.head(100)
    
    engineer = FeatureEngineer(random_seed=42)
    
    print("\n1. Testing statistical features...")
    stat_feat = engineer.extract_statistical_features(sample_data)
    print(f"Shape: {stat_feat.shape}")
    print(f"Columns: {list(stat_feat.columns)}")
    
    print("\n2. Testing temporal features...")
    temp_feat = engineer.extract_temporal_features(sample_data)
    print(f"Shape: {temp_feat.shape}")
    print(f"Columns: {list(temp_feat.columns[:5])}... ({len(temp_feat.columns)} total)")
    
    print("\n3. Testing seasonality features...")
    season_feat = engineer.extract_seasonality_features(sample_data)
    print(f"Shape: {season_feat.shape}")
    print(f"Columns: {list(season_feat.columns)}")
    
    print("\n4. Testing variability features...")
    var_feat = engineer.extract_variability_features(sample_data)
    print(f"Shape: {var_feat.shape}")
    print(f"Columns: {list(var_feat.columns)}")
    
    print("\n5. Testing shape features...")
    shape_feat = engineer.extract_shape_features(sample_data)
    print(f"Shape: {shape_feat.shape}")
    print(f"Columns: {list(shape_feat.columns)}")
    
    print("\n6. Testing complete feature extraction...")
    all_feat = engineer.extract_all_features(sample_data)
    print(f"Total features: {all_feat.shape}")
    print(f"Feature names: {len(engineer.feature_names)}")
    print(f"Sample:\n{all_feat.head()}")
    
    print("\n✅ All tests passed!")
