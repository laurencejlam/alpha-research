import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


class FactorVisualization:
    """
    Class for visualizing alpha factor performance and characteristics
    """
    
    def __init__(self):
        """Initialize the FactorVisualization class"""
        # Set default style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        self.color_palette = sns.color_palette("viridis", 5)
    
    def plot_quantile_returns(self, quantile_returns, factor_name=None, figsize=(12, 6)):
        """
        Plot cumulative returns by quantile
        
        Parameters:
        -----------
        quantile_returns : dict
            Dictionary of {quantile: cumulative_returns_series}
        factor_name : str
            Name of the factor
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Plot of quantile returns
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot cumulative returns for each quantile
        for quantile, returns in quantile_returns.items():
            cumulative = (1 + returns).cumprod() - 1
            cumulative.plot(ax=ax, linewidth=2, 
                            label=f'Quantile {quantile}')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        title = 'Cumulative Returns by Quantile'
        if factor_name:
            title += f' - {factor_name}'
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_ic_time_series(self, ic_series, factor_name=None, figsize=(12, 6)):
        """
        Plot Information Coefficient (IC) time series
        
        Parameters:
        -----------
        ic_series : pd.Series
            Time series of IC values
        factor_name : str
            Name of the factor
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Plot of IC time series
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot IC time series
        ic_series.plot(ax=ax, color='blue', alpha=0.5, linewidth=1)
        
        # Add rolling average
        window = min(63, len(ic_series) // 5)  # 3-month rolling window or 1/5 of series length
        if window > 0:
            ic_rolling = ic_series.rolling(window=window).mean()
            ic_rolling.plot(ax=ax, color='red', linewidth=2, 
                            label=f'{window}-day Moving Average')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Information Coefficient (IC)')
        title = 'Information Coefficient (IC) Time Series'
        if factor_name:
            title += f' - {factor_name}'
        ax.set_title(title)
        
        # Add IC statistics
        mean_ic = ic_series.mean()
        ic_t_stat = (mean_ic / (ic_series.std() / np.sqrt(len(ic_series)))) if len(ic_series) > 0 else 0
        textstr = f'Mean IC: {mean_ic:.4f}\nIC t-stat: {ic_t_stat:.4f}\nIC > 0: {(ic_series > 0).mean():.1%}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.02, 0.05, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_ic_histogram(self, ic_series, factor_name=None, figsize=(10, 6)):
        """
        Plot histogram of Information Coefficient (IC) values
        
        Parameters:
        -----------
        ic_series : pd.Series
            Series of IC values
        factor_name : str
            Name of the factor
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Histogram of IC values
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        sns.histplot(ic_series, bins=30, kde=True, ax=ax)
        
        # Add vertical line at the mean
        mean_ic = ic_series.mean()
        ax.axvline(x=mean_ic, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_ic:.4f}')
        
        # Add vertical line at y=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Set labels and title
        ax.set_xlabel('Information Coefficient (IC)')
        ax.set_ylabel('Frequency')
        title = 'Information Coefficient (IC) Distribution'
        if factor_name:
            title += f' - {factor_name}'
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        return fig
    
    def plot_factor_returns(self, factor_returns, benchmark_returns=None, 
                           factor_name=None, figsize=(12, 6)):
        """
        Plot factor cumulative returns
        
        Parameters:
        -----------
        factor_returns : pd.Series
            Factor returns
        benchmark_returns : pd.Series, optional
            Benchmark returns for comparison
        factor_name : str
            Name of the factor
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Plot of cumulative returns
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate cumulative returns
        factor_cum_returns = (1 + factor_returns).cumprod() - 1
        
        # Plot factor cumulative returns
        name = factor_name if factor_name else 'Factor'
        factor_cum_returns.plot(ax=ax, color='blue', linewidth=2, label=name)
        
        # Plot benchmark if provided
        if benchmark_returns is not None:
            benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
            benchmark_cum_returns.plot(ax=ax, color='gray', linewidth=1, label='Benchmark')
            
            # Plot factor minus benchmark (excess returns)
            excess_returns = factor_returns - benchmark_returns
            excess_cum_returns = (1 + excess_returns).cumprod() - 1
            excess_cum_returns.plot(ax=ax, color='green', linewidth=1, 
                                    linestyle='--', label=f'{name} - Benchmark')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        title = 'Factor Cumulative Returns'
        if factor_name:
            title += f' - {factor_name}'
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_rolling_sharpe(self, returns, window=252, factor_name=None, figsize=(12, 6)):
        """
        Plot rolling Sharpe ratio
        
        Parameters:
        -----------
        returns : pd.Series
            Returns series
        window : int
            Rolling window size in days
        factor_name : str
            Name of the factor
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Plot of rolling Sharpe ratio
        """
        # Calculate rolling Sharpe ratio (annualized)
        rolling_mean = returns.rolling(window=window).mean() * 252
        rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot rolling Sharpe ratio
        rolling_sharpe.plot(ax=ax, color='purple', linewidth=2)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        
        # Add horizontal lines at y=1 and y=2
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        ax.axhline(y=2, color='green', linestyle='--', alpha=0.8, label='Sharpe = 2')
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel(f'{window}-day Rolling Sharpe Ratio')
        title = f'{window}-day Rolling Sharpe Ratio'
        if factor_name:
            title += f' - {factor_name}'
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_sector_exposure(self, sector_exposure, factor_name=None, figsize=(12, 8)):
        """
        Plot sector exposure heatmap
        
        Parameters:
        -----------
        sector_exposure : pd.DataFrame
            Sector exposures over time (rows are dates, columns are sectors)
        factor_name : str
            Name of the factor
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Heatmap of sector exposures
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create custom diverging colormap (blue-white-red)
        cmap = LinearSegmentedColormap.from_list('BuWtRd', ['blue', 'white', 'red'])
        
        # Create heatmap
        sns.heatmap(sector_exposure.T, cmap=cmap, center=0, 
                    robust=True, ax=ax, cbar_kws={'label': 'Exposure'})
        
        # Set labels and title
        ax.set_ylabel('Sector')
        ax.set_xlabel('Date')
        title = 'Sector Exposure Over Time'
        if factor_name:
            title += f' - {factor_name}'
        ax.set_title(title)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Adjust layout to make room for labels
        fig.tight_layout()
        
        return fig
    
    def plot_monthly_returns_heatmap(self, returns, factor_name=None, figsize=(12, 8)):
        """
        Plot monthly returns heatmap
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns series
        factor_name : str
            Name of the factor
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Heatmap of monthly returns
        """
        # Resample to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create DataFrame with returns by year and month
        returns_by_month = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        # Pivot to create year x month grid
        returns_pivot = returns_by_month.pivot(index='Year', columns='Month', values='Return')
        
        # Replace month numbers with names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        returns_pivot.columns = [month_names[i-1] for i in returns_pivot.columns]
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create custom diverging colormap (red-white-green)
        cmap = LinearSegmentedColormap.from_list('RdWtGn', ['red', 'white', 'green'])
        
        # Create heatmap
        sns.heatmap(returns_pivot, cmap=cmap, center=0, 
                    annot=True, fmt='.1%', ax=ax, linewidths=1, 
                    cbar_kws={'label': 'Monthly Return'})
        
        # Set labels and title
        ax.set_ylabel('Year')
        ax.set_xlabel('Month')
        title = 'Monthly Returns'
        if factor_name:
            title += f' - {factor_name}'
        ax.set_title(title)
        
        # Add yearly returns as an additional column
        yearly_returns = returns.resample('A').apply(lambda x: (1 + x).prod() - 1)
        yearly_returns = pd.Series(yearly_returns.values, index=yearly_returns.index.year)
        yearly_returns = yearly_returns.reindex(returns_pivot.index)
        
        # Add text for yearly returns
        for i, year in enumerate(returns_pivot.index):
            if year in yearly_returns.index:
                ax.text(returns_pivot.shape[1] + 0.5, i + 0.5, 
                       f'{yearly_returns[year]:.1%}', 
                       ha='center', va='center')
        
        # Add 'Year' label for the additional column
        ax.text(returns_pivot.shape[1] + 0.5, -0.5, 'Year', 
               ha='center', va='center', fontweight='bold')
        
        # Adjust layout to make room for labels
        fig.tight_layout()
        
        return fig 