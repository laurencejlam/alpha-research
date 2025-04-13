import pandas as pd
import numpy as np
from alpharesearch.factors.base_factor import BaseFactor


class VolumeSurprise(BaseFactor):
    """
    Volume Surprise Factor
    
    Computes a factor based on the relationship between volume changes and price movements.
    The idea is that unusually high volume accompanied by price movements can signal
    continuation of the trend.
    """
    
    def __init__(self, volume_lookback=20, name="VolumeSurprise"):
        """
        Initialize the Volume Surprise factor
        
        Parameters:
        -----------
        volume_lookback : int
            Number of trading days to use for volume baseline calculation
        name : str
            Name of the factor
        """
        super().__init__(name)
        self.volume_lookback = volume_lookback
    
    def compute(self, data):
        """
        Compute the volume surprise factor
        
        Parameters:
        -----------
        data : dict
            Dictionary with market data (prices, volumes, returns)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with volume surprise values for each asset
        """
        if 'volumes' not in data or 'returns' not in data:
            raise ValueError("Volume or returns data not found in input data")
            
        volumes = data['volumes']
        returns = data['returns']
        
        # Calculate average volume over the lookback period
        avg_volume = volumes.rolling(window=self.volume_lookback).mean()
        
        # Calculate volume surprise (ratio of current volume to average volume)
        volume_ratio = volumes / avg_volume
        
        # Sign the volume surprise with the sign of returns
        # High volume with positive returns is positive signal
        # High volume with negative returns is negative signal
        volume_surprise = volume_ratio * np.sign(returns)
        
        # Fill NaN values with 0
        volume_surprise = volume_surprise.fillna(0)
        
        self.factor_data = volume_surprise
        return volume_surprise
    
    def normalize(self, factor_data=None):
        """
        Normalize the factor values using percentile ranking within each time period
        
        Parameters:
        -----------
        factor_data : pd.DataFrame, optional
            Factor data to normalize. If None, use self.factor_data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with normalized factor values
        """
        if factor_data is None:
            factor_data = self.factor_data
            
        if factor_data is None:
            raise ValueError("Factor data is not computed yet. Call compute() first.")
            
        # Calculate percentile rank for each day (cross-sectional)
        normalized = factor_data.rank(axis=1, pct=True)
        
        # Scale to [-1, 1] range
        normalized = 2 * normalized - 1
        
        return normalized
    
    def get_decay_profile(self, max_days=10):
        """
        Get the signal decay profile
        
        Parameters:
        -----------
        max_days : int
            Maximum number of days to consider for decay
            
        Returns:
        --------
        list
            List of decay weights
        """
        # Volume effects typically decay exponentially
        # Half-life of about 2-3 days
        half_life = 2.5
        days = np.arange(1, max_days + 1)
        decay = np.exp(-np.log(2) * days / half_life)
        return decay / decay.sum()  # Normalize to sum to 1 