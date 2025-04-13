import pandas as pd
import numpy as np
from alpharesearch.factors.base_factor import BaseFactor


class MeanReversion(BaseFactor):
    """
    Mean Reversion (Short-term Reversal) Factor
    
    Computes the negative of recent returns, based on the theory that
    short-term price movements tend to revert back to the mean.
    """
    
    def __init__(self, lookback_period=5, name="MeanReversion"):
        """
        Initialize the Mean Reversion factor
        
        Parameters:
        -----------
        lookback_period : int
            Number of trading days to look back for recent returns
        name : str
            Name of the factor
        """
        super().__init__(name)
        self.lookback_period = lookback_period
    
    def compute(self, data):
        """
        Compute the mean reversion factor
        
        Parameters:
        -----------
        data : dict
            Dictionary with market data (prices, volumes, returns)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with mean reversion values for each asset
        """
        if 'prices' not in data:
            raise ValueError("Price data not found in input data")
            
        prices = data['prices']
        
        # Calculate recent returns
        past_prices = prices.shift(self.lookback_period)
        recent_returns = (prices - past_prices) / past_prices
        
        # Negative of recent returns is the mean reversion signal
        # Stocks that went down are expected to go up, and vice versa
        mean_reversion = -recent_returns
        
        self.factor_data = mean_reversion
        return mean_reversion
    
    def normalize(self, factor_data=None):
        """
        Normalize the factor values using z-score within each time period
        
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
            
        # Z-score normalization within each day
        normalized = factor_data.sub(factor_data.mean(axis=1), axis=0)
        normalized = normalized.div(factor_data.std(axis=1), axis=0)
        
        # Winsorize outliers
        normalized = normalized.clip(-3, 3)
        
        return normalized
    
    def get_forecast_horizon(self):
        """
        Get the natural forecast horizon for the factor
        
        Returns:
        --------
        int
            Suggested forecast horizon in trading days
        """
        # Mean reversion effects are usually short-lived
        # A good rule of thumb is 1-5 days for very short-term effects
        return min(5, self.lookback_period) 