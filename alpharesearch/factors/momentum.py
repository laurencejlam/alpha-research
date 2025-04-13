import pandas as pd
import numpy as np
from alpharesearch.factors.base_factor import BaseFactor


class Momentum(BaseFactor):
    """
    Price Momentum Factor
    
    Computes the price momentum based on past returns over a specified lookback period,
    with an optional skip period to avoid short-term reversal effects.
    """
    
    def __init__(self, lookback_period=252, skip_period=21, name="Momentum"):
        """
        Initialize the Momentum factor
        
        Parameters:
        -----------
        lookback_period : int
            Number of trading days to look back for momentum calculation
        skip_period : int
            Number of recent days to skip to avoid short-term reversal effects
        name : str
            Name of the factor
        """
        super().__init__(name)
        self.lookback_period = lookback_period
        self.skip_period = skip_period
    
    def compute(self, data):
        """
        Compute the momentum factor
        
        Parameters:
        -----------
        data : dict
            Dictionary with market data (prices, volumes, returns)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with momentum values for each asset
        """
        if 'prices' not in data:
            raise ValueError("Price data not found in input data")
            
        prices = data['prices']
        
        # Calculate returns over the lookback period, skipping the most recent days
        if self.skip_period > 0:
            # For each day, get price 'lookback_period + skip_period' days ago and 'skip_period' days ago
            past_prices = prices.shift(self.lookback_period + self.skip_period)
            recent_prices = prices.shift(self.skip_period)
            
            # Calculate returns between these two periods
            momentum = (recent_prices - past_prices) / past_prices
        else:
            # If no skip period, just calculate returns over the lookback period
            past_prices = prices.shift(self.lookback_period)
            momentum = (prices - past_prices) / past_prices
        
        self.factor_data = momentum
        return momentum
    
    def get_forecast_horizon(self):
        """
        Get the natural forecast horizon for the factor
        
        Returns:
        --------
        int
            Suggested forecast horizon in trading days
        """
        # A rule of thumb: momentum effects persist for about 30-90 days
        # for medium-term momentum (6-12 months lookback)
        if self.lookback_period < 126:  # Less than 6 months
            return 21  # 1 month forward
        elif self.lookback_period < 252:  # 6-12 months
            return 63  # 3 months forward
        else:  # 12+ months
            return 126  # 6 months forward 