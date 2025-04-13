import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseFactor(ABC):
    """Base class for all alpha factors"""
    
    def __init__(self, name):
        """
        Initialize the factor
        
        Parameters:
        -----------
        name : str
            Name of the factor
        """
        self.name = name
        self.factor_data = None
        self.factor_ranks = None
    
    @abstractmethod
    def compute(self, data):
        """
        Compute the factor values
        
        Parameters:
        -----------
        data : dict
            Dictionary with market data (prices, volumes, returns)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with factor values for each asset
        """
        pass
    
    def get_ranks(self, factor_data=None, ascending=False):
        """
        Rank securities based on factor values
        
        Parameters:
        -----------
        factor_data : pd.DataFrame, optional
            Factor data to rank. If None, use self.factor_data
        ascending : bool
            If True, rank in ascending order (smaller values get higher ranks)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with factor ranks for each asset
        """
        if factor_data is None:
            factor_data = self.factor_data
            
        if factor_data is None:
            raise ValueError("Factor data is not computed yet. Call compute() first.")
            
        # Rank cross-sectionally (by row)
        ranks = factor_data.rank(axis=1, ascending=ascending, method='first')
        self.factor_ranks = ranks
        return ranks
    
    def get_normalized_ranks(self, factor_ranks=None):
        """
        Normalize ranks to [-1, 1] range
        
        Parameters:
        -----------
        factor_ranks : pd.DataFrame, optional
            Factor ranks to normalize. If None, use self.factor_ranks
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with normalized factor ranks for each asset
        """
        if factor_ranks is None:
            factor_ranks = self.factor_ranks
            
        if factor_ranks is None:
            raise ValueError("Factor ranks are not computed yet. Call get_ranks() first.")
            
        # Normalize ranks to [-1, 1] range
        n = factor_ranks.max(axis=1)  # Number of assets per day
        normalized_ranks = 2 * (factor_ranks - 1) / (n - 1) - 1
        return normalized_ranks
    
    def get_quantiles(self, factor_data=None, num_quantiles=5):
        """
        Divide securities into quantiles based on factor values
        
        Parameters:
        -----------
        factor_data : pd.DataFrame, optional
            Factor data to use. If None, use self.factor_data
        num_quantiles : int
            Number of quantiles (buckets)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with quantile assignments for each asset
        """
        if factor_data is None:
            factor_data = self.factor_data
            
        if factor_data is None:
            raise ValueError("Factor data is not computed yet. Call compute() first.")
            
        # Calculate quantiles cross-sectionally
        quantiles = factor_data.apply(
            lambda x: pd.qcut(x, num_quantiles, labels=False, duplicates='drop'),
            axis=1
        )
        return quantiles
    
    def get_top_bottom_quantiles(self, quantiles, top_quantile=4, bottom_quantile=0):
        """
        Get binary indicators for top and bottom quantiles
        
        Parameters:
        -----------
        quantiles : pd.DataFrame
            Quantile assignments
        top_quantile : int
            Index of the top quantile
        bottom_quantile : int
            Index of the bottom quantile
            
        Returns:
        --------
        tuple of pd.DataFrame
            (top_quantile_indicators, bottom_quantile_indicators)
        """
        top_indicators = quantiles == top_quantile
        bottom_indicators = quantiles == bottom_quantile
        return top_indicators, bottom_indicators 