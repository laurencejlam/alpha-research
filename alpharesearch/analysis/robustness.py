import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class RobustnessAnalysis:
    """
    Class for performing robustness tests on alpha factors
    """
    
    def __init__(self):
        """Initialize the RobustnessAnalysis class"""
        pass
    
    @staticmethod
    def calculate_ic_decay(factor, returns, horizons=None):
        """
        Calculate Information Coefficient (IC) decay over different horizons
        
        Parameters:
        -----------
        factor : pd.DataFrame
            Factor values for each asset
        returns : pd.DataFrame
            Asset returns
        horizons : list
            List of horizons (in days) to test
            
        Returns:
        --------
        pd.Series
            Average IC for each horizon
        """
        if horizons is None:
            horizons = [1, 5, 10, 21, 63]
            
        ic_decay = pd.Series(index=horizons)
        
        for horizon in horizons:
            # Calculate ICs for each day
            ic_values = []
            
            for date in factor.index:
                # Get future returns at the specified horizon
                if date in returns.index and len(returns.index) > returns.index.get_loc(date) + horizon:
                    future_date = returns.index[returns.index.get_loc(date) + horizon]
                    
                    factor_values = factor.loc[date].dropna()
                    future_rets = returns.loc[future_date, factor_values.index].dropna()
                    
                    # Get common assets
                    common_assets = factor_values.index.intersection(future_rets.index)
                    
                    if len(common_assets) > 10:  # Require at least 10 assets
                        # Calculate rank correlation
                        ic = stats.spearmanr(factor_values[common_assets], future_rets[common_assets])[0]
                        ic_values.append(ic)
            
            # Calculate average IC for this horizon
            if ic_values:
                ic_decay[horizon] = np.mean(ic_values)
            else:
                ic_decay[horizon] = np.nan
                
        return ic_decay
    
    @staticmethod
    def calculate_turnover(factor, lookback=1):
        """
        Calculate factor turnover
        
        Parameters:
        -----------
        factor : pd.DataFrame
            Factor values for each asset
        lookback : int
            Lookback period for turnover calculation
            
        Returns:
        --------
        pd.Series
            Turnover for each time period
        """
        # Normalize factor to sum to 1 on each day
        normalized_factor = factor.divide(factor.abs().sum(axis=1), axis=0)
        
        # Calculate changes in factor values
        changes = normalized_factor.subtract(normalized_factor.shift(lookback))
        
        # Calculate turnover as the sum of absolute changes
        turnover = changes.abs().sum(axis=1)
        
        return turnover.dropna()
    
    @staticmethod
    def calculate_factor_correlation(factors):
        """
        Calculate correlation between multiple factors
        
        Parameters:
        -----------
        factors : dict
            Dictionary of {factor_name: factor_dataframe}
            
        Returns:
        --------
        pd.DataFrame
            Average correlation matrix between factors
        """
        if len(factors) < 2:
            raise ValueError("Need at least two factors to calculate correlations")
            
        # Get common dates and assets
        common_dates = set.intersection(*[set(factor.index) for factor in factors.values()])
        
        # Calculate correlations for each date
        correlations = []
        
        for date in common_dates:
            # Get factor values for this date
            factor_values = {}
            
            for name, factor in factors.items():
                factor_values[name] = factor.loc[date]
                
            # Create a DataFrame of factor values
            factor_df = pd.DataFrame(factor_values)
            
            # Calculate correlation matrix
            corr_matrix = factor_df.corr(method='spearman')
            correlations.append(corr_matrix)
            
        # Calculate average correlation matrix
        if correlations:
            avg_correlation = sum(correlations) / len(correlations)
            return avg_correlation
        else:
            return pd.DataFrame()
    
    @staticmethod
    def calculate_sector_exposure(factor, sector_mapping):
        """
        Calculate sector exposure of a factor
        
        Parameters:
        -----------
        factor : pd.DataFrame
            Factor values for each asset
        sector_mapping : pd.Series
            Mapping from asset to sector
            
        Returns:
        --------
        pd.DataFrame
            Sector exposures for each time period
        """
        # Normalize factor values
        normalized_factor = factor.subtract(factor.mean(axis=1), axis=0).divide(factor.std(axis=1), axis=0)
        
        # Create a DataFrame for sector exposures
        unique_sectors = sector_mapping.unique()
        sector_exposure = pd.DataFrame(index=factor.index, columns=unique_sectors)
        
        for date in factor.index:
            factor_values = normalized_factor.loc[date].dropna()
            
            # Get sector for each asset
            sectors = sector_mapping.reindex(factor_values.index)
            
            # Calculate average factor value by sector
            for sector in unique_sectors:
                sector_assets = sectors[sectors == sector].index
                if len(sector_assets) > 0:
                    sector_exposure.loc[date, sector] = factor_values[sector_assets].mean()
                else:
                    sector_exposure.loc[date, sector] = 0
                    
        return sector_exposure
    
    @staticmethod
    def plot_ic_decay(ic_decay, factor_name=None, figsize=(10, 6)):
        """
        Plot IC decay
        
        Parameters:
        -----------
        ic_decay : pd.Series
            IC values for different horizons
        factor_name : str
            Name of the factor
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Plot of IC decay
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot IC decay
        ic_decay.plot(ax=ax, marker='o', linestyle='-', linewidth=2)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('Horizon (days)')
        ax.set_ylabel('Information Coefficient (IC)')
        title = 'IC Decay'
        if factor_name:
            title += f' - {factor_name}'
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_turnover_distribution(turnover, factor_name=None, figsize=(10, 6)):
        """
        Plot turnover distribution
        
        Parameters:
        -----------
        turnover : pd.Series
            Turnover for each time period
        factor_name : str
            Name of the factor
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Histogram of turnover values
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        turnover.hist(ax=ax, bins=20, alpha=0.7)
        
        # Add vertical line at the mean
        mean_turnover = turnover.mean()
        ax.axvline(x=mean_turnover, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_turnover:.2f}')
        
        # Set labels and title
        ax.set_xlabel('Turnover')
        ax.set_ylabel('Frequency')
        title = 'Turnover Distribution'
        if factor_name:
            title += f' - {factor_name}'
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        return fig
    
    @staticmethod
    def plot_correlation_heatmap(correlation_matrix, figsize=(10, 8)):
        """
        Plot correlation heatmap
        
        Parameters:
        -----------
        correlation_matrix : pd.DataFrame
            Correlation matrix
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Heatmap of correlations
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Correlation')
        
        # Add ticks and labels
        factor_names = correlation_matrix.index
        ax.set_xticks(range(len(factor_names)))
        ax.set_yticks(range(len(factor_names)))
        ax.set_xticklabels(factor_names, rotation=45, ha='right')
        ax.set_yticklabels(factor_names)
        
        # Add title
        ax.set_title('Factor Correlation Matrix')
        
        # Add correlation values
        for i in range(len(factor_names)):
            for j in range(len(factor_names)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                              ha='center', va='center', color='black' if abs(correlation_matrix.iloc[i, j]) < 0.7 else 'white')
        
        fig.tight_layout()
        return fig 