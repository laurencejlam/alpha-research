import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class Portfolio:
    """
    Class for portfolio construction and analysis
    """
    
    def __init__(self, name="Portfolio"):
        """
        Initialize the portfolio
        
        Parameters:
        -----------
        name : str
            Name of the portfolio
        """
        self.name = name
        self.weights = None
        self.returns = None
        self.cumulative_returns = None
        self.metrics = {}
    
    def calculate_weights_from_signal(self, signal, quantiles=None, top_quantile=4, bottom_quantile=0,
                                      long_short=True, equal_weight=True):
        """
        Calculate portfolio weights based on factor signal
        
        Parameters:
        -----------
        signal : pd.DataFrame
            Factor signal for each asset
        quantiles : pd.DataFrame, optional
            Quantile assignments (if None, use signal directly)
        top_quantile : int
            Index of the top quantile (long)
        bottom_quantile : int
            Index of the bottom quantile (short)
        long_short : bool
            If True, construct a long-short portfolio; otherwise, long-only
        equal_weight : bool
            If True, equal weight within each quantile; otherwise, weight by signal strength
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with portfolio weights for each asset
        """
        if quantiles is not None:
            # Get indicators for top and bottom quantiles
            top_indicators = (quantiles == top_quantile).astype(float)
            bottom_indicators = (quantiles == bottom_quantile).astype(float)
            
            if long_short:
                # Long-short portfolio
                if equal_weight:
                    # Equal weight within each quantile
                    weights = top_indicators.div(top_indicators.sum(axis=1), axis=0)
                    weights = weights.subtract(bottom_indicators.div(bottom_indicators.sum(axis=1), axis=0), fill_value=0)
                else:
                    # Weight by signal strength
                    weights = top_indicators * signal
                    weights = weights.div(weights.sum(axis=1), axis=0)
                    bottom_weights = bottom_indicators * -signal
                    weights = weights.subtract(bottom_weights.div(bottom_weights.sum(axis=1), axis=0), fill_value=0)
            else:
                # Long-only portfolio
                if equal_weight:
                    # Equal weight
                    weights = top_indicators.div(top_indicators.sum(axis=1), axis=0)
                else:
                    # Weight by signal strength
                    weights = top_indicators * signal
                    weights = weights.div(weights.abs().sum(axis=1), axis=0)
        else:
            # Use signal directly for weights
            if long_short:
                # Normalize signal to sum to 0 (market neutral)
                weights = signal.subtract(signal.mean(axis=1), axis=0)
                # Scale so that absolute weights sum to 2 (1 for long, 1 for short)
                weights = weights.div(weights.abs().sum(axis=1) / 2, axis=0)
            else:
                # For long-only, keep only positive signals
                weights = signal.clip(lower=0)
                # Normalize to sum to 1
                weights = weights.div(weights.sum(axis=1), axis=0)
        
        # Replace NaN weights with 0
        weights = weights.fillna(0)
        
        self.weights = weights
        return weights
    
    def calculate_portfolio_returns(self, weights, returns, transaction_costs=None):
        """
        Calculate portfolio returns
        
        Parameters:
        -----------
        weights : pd.DataFrame
            Portfolio weights for each asset
        returns : pd.DataFrame
            Asset returns
        transaction_costs : float or pd.DataFrame, optional
            Transaction costs as a fraction of traded value
            
        Returns:
        --------
        pd.Series
            Portfolio returns
        """
        # Align weights and returns
        aligned_returns = returns.reindex(columns=weights.columns)
        
        # Calculate portfolio returns without transaction costs
        port_returns = (weights.shift(1) * aligned_returns).sum(axis=1)
        
        # Apply transaction costs if specified
        if transaction_costs is not None:
            # Calculate weight changes (turnover)
            weight_changes = weights.diff().abs()
            
            # Calculate transaction costs
            if isinstance(transaction_costs, (int, float)):
                costs = weight_changes.sum(axis=1) * transaction_costs
            else:
                # Asset-specific transaction costs
                costs = (weight_changes * transaction_costs).sum(axis=1)
                
            # Subtract costs from returns
            port_returns = port_returns - costs
            
        # Drop NaN values (first row)
        port_returns = port_returns.dropna()
        
        self.returns = port_returns
        return port_returns
    
    def calculate_cumulative_returns(self, returns=None):
        """
        Calculate cumulative returns
        
        Parameters:
        -----------
        returns : pd.Series, optional
            Portfolio returns. If None, use self.returns
            
        Returns:
        --------
        pd.Series
            Cumulative returns
        """
        if returns is None:
            returns = self.returns
            
        if returns is None:
            raise ValueError("Portfolio returns are not calculated yet")
            
        cumulative = (1 + returns).cumprod() - 1
        self.cumulative_returns = cumulative
        return cumulative
    
    def calculate_metrics(self, returns=None, risk_free_rate=0.0, annualization_factor=252):
        """
        Calculate portfolio performance metrics
        
        Parameters:
        -----------
        returns : pd.Series, optional
            Portfolio returns. If None, use self.returns
        risk_free_rate : float
            Daily risk-free rate
        annualization_factor : int
            Annualization factor (252 for daily returns)
            
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        if returns is None:
            returns = self.returns
            
        if returns is None:
            raise ValueError("Portfolio returns are not calculated yet")
            
        # Calculate basic metrics
        total_return = (1 + returns).prod() - 1
        excess_returns = returns - risk_free_rate
        
        # Annualized metrics
        ann_return = (1 + total_return) ** (annualization_factor / len(returns)) - 1
        ann_volatility = returns.std() * np.sqrt(annualization_factor)
        
        # Sharpe ratio
        sharpe_ratio = (ann_return - risk_free_rate * annualization_factor) / ann_volatility if ann_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1)
        max_drawdown = drawdown.min()
        
        # Calculate t-statistic
        t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
        
        # Calculate turnover if weights are available
        turnover = None
        if self.weights is not None:
            turnover = self.weights.diff().abs().sum(axis=1).mean()
        
        # Store metrics in a dictionary
        metrics = {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (returns > 0).mean(),
            'hit_ratio': (excess_returns > 0).mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'turnover': turnover
        }
        
        self.metrics = metrics
        return metrics
    
    def plot_cumulative_returns(self, benchmark_returns=None, figsize=(12, 6)):
        """
        Plot cumulative returns
        
        Parameters:
        -----------
        benchmark_returns : pd.Series, optional
            Benchmark returns for comparison
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Plot of cumulative returns
        """
        if self.cumulative_returns is None:
            self.calculate_cumulative_returns()
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot portfolio cumulative returns
        self.cumulative_returns.plot(ax=ax, color='blue', linewidth=2, label=self.name)
        
        # Plot benchmark if provided
        if benchmark_returns is not None:
            benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
            benchmark_cum_returns.plot(ax=ax, color='gray', linewidth=1, label='Benchmark')
            
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_drawdown(self, figsize=(12, 6)):
        """
        Plot drawdowns
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Plot of drawdowns
        """
        if self.returns is None:
            raise ValueError("Portfolio returns are not calculated yet")
            
        # Calculate drawdowns
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1) * 100  # Convert to percentage
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot drawdowns
        drawdown.plot(ax=ax, color='red', linewidth=1)
        
        ax.set_title('Portfolio Drawdowns')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
        
        return fig
    
    def calculate_ic(self, signal, forward_returns, horizon=1):
        """
        Calculate Information Coefficient (IC)
        
        Parameters:
        -----------
        signal : pd.DataFrame
            Factor signal
        forward_returns : pd.DataFrame
            Forward returns
        horizon : int
            Forward return horizon in days
            
        Returns:
        --------
        pd.Series
            Time series of IC values
        """
        # Shift returns back by horizon to align with signals
        aligned_returns = forward_returns.shift(-horizon)
        
        # Calculate rank IC (Spearman correlation) for each day
        ic_series = pd.Series(index=signal.index)
        
        for date in signal.index:
            if date in aligned_returns.index:
                signals = signal.loc[date].dropna()
                future_rets = aligned_returns.loc[date, signals.index].dropna()
                
                # Get common assets
                common_assets = signals.index.intersection(future_rets.index)
                
                if len(common_assets) > 10:  # Require at least 10 assets for meaningful correlation
                    ic = stats.spearmanr(signals[common_assets], future_rets[common_assets])[0]
                    ic_series[date] = ic
        
        return ic_series.dropna() 