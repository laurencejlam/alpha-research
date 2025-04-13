import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

from alpharesearch.data import DataLoader
from alpharesearch.factors import Momentum, MeanReversion, VolumeSurprise
from alpharesearch.analysis import Portfolio, RobustnessAnalysis
from alpharesearch.visualization import FactorVisualization
from alpharesearch.utils import ensure_dir, save_data, load_data, get_timestamps


def run_factor_analysis(
    start_date=None,
    end_date=None,
    momentum_lookback=252,
    momentum_skip=21,
    reversal_lookback=5,
    volume_lookback=20,
    num_stocks=100,
    transaction_costs=0.001,
    output_dir='results'
):
    """
    Run the complete factor analysis pipeline
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    momentum_lookback : int
        Lookback period for momentum factor in days
    momentum_skip : int
        Skip period for momentum factor in days
    reversal_lookback : int
        Lookback period for reversal factor in days
    volume_lookback : int
        Lookback period for volume surprise factor in days
    num_stocks : int
        Number of stocks to analyze
    transaction_costs : float
        Transaction costs as fraction of traded value
    output_dir : str
        Directory to save results
    """
    # Create output directory
    ensure_dir(output_dir)
    timestamp = get_timestamps()
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    ensure_dir(run_dir)
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    print(f"Running factor analysis from {start_date} to {end_date}")
    print(f"Results will be saved to {run_dir}")
    
    # Load data
    print("\n1. Loading market data...")
    data_loader = DataLoader(start_date=start_date, end_date=end_date)
    
    # Get S&P 500 tickers
    tickers = data_loader.get_sp500_tickers()
    
    # If num_stocks is less than number of tickers, select a random subset
    if num_stocks < len(tickers):
        np.random.seed(42)  # For reproducibility
        tickers = np.random.choice(tickers, size=num_stocks, replace=False).tolist()
        data_loader.tickers = tickers
    
    # Fetch data
    market_data = data_loader.fetch_data()
    
    # Get market returns (S&P 500)
    sp500_data = data_loader.get_market_data(market_ticker='^GSPC')
    sp500_returns = sp500_data['Adj Close'].pct_change().dropna()
    
    # Save market data info
    with open(os.path.join(run_dir, 'data_info.txt'), 'w') as f:
        f.write(f"Start date: {start_date}\n")
        f.write(f"End date: {end_date}\n")
        f.write(f"Number of stocks: {len(data_loader.tickers)}\n")
        f.write(f"Ticker list: {', '.join(data_loader.tickers)}\n\n")
        
        f.write("Data statistics:\n")
        f.write(f"Number of trading days: {len(market_data['returns'])}\n")
        f.write(f"Missing values in returns: {market_data['returns'].isna().sum().sum()}\n")
        f.write(f"Missing values in prices: {market_data['prices'].isna().sum().sum()}\n")
        f.write(f"Missing values in volumes: {market_data['volumes'].isna().sum().sum()}\n")
    
    # Create factors
    print("\n2. Computing alpha factors...")
    # 2.1 Momentum factor
    momentum = Momentum(lookback_period=momentum_lookback, skip_period=momentum_skip)
    momentum_signal = momentum.compute(market_data)
    momentum_quantiles = momentum.get_quantiles(num_quantiles=5)
    
    # 2.2 Mean Reversion factor
    mean_reversion = MeanReversion(lookback_period=reversal_lookback)
    reversal_signal = mean_reversion.compute(market_data)
    reversal_quantiles = mean_reversion.get_quantiles(num_quantiles=5)
    
    # 2.3 Volume Surprise factor
    volume_surprise = VolumeSurprise(volume_lookback=volume_lookback)
    volume_signal = volume_surprise.compute(market_data)
    volume_quantiles = volume_surprise.get_quantiles(num_quantiles=5)
    
    # 3. Run portfolio analysis
    print("\n3. Running portfolio analysis...")
    # Initialize visualization
    viz = FactorVisualization()
    
    # Dictionary to store results
    results = {
        'factors': {
            'momentum': momentum_signal,
            'reversal': reversal_signal,
            'volume_surprise': volume_signal
        },
        'quantiles': {
            'momentum': momentum_quantiles,
            'reversal': reversal_quantiles,
            'volume_surprise': volume_quantiles
        },
        'metrics': {},
        'portfolio_returns': {}
    }
    
    # Loop through factors and create portfolios
    for factor_name, factor_signal, factor_quantiles in [
        ('Momentum', momentum_signal, momentum_quantiles),
        ('Reversal', reversal_signal, reversal_quantiles),
        ('VolumeSurprise', volume_signal, volume_quantiles)
    ]:
        print(f"\nAnalyzing {factor_name} factor...")
        
        # Create portfolio
        portfolio = Portfolio(name=factor_name)
        
        # Calculate weights
        weights = portfolio.calculate_weights_from_signal(
            factor_signal, 
            quantiles=factor_quantiles,
            top_quantile=4,  # Long the highest quantile
            bottom_quantile=0,  # Short the lowest quantile
            long_short=True,  # Long-short portfolio
            equal_weight=True  # Equal weight within each quantile
        )
        
        # Calculate portfolio returns
        portfolio_returns = portfolio.calculate_portfolio_returns(
            weights, 
            market_data['returns'],
            transaction_costs=transaction_costs
        )
        
        # Calculate cumulative returns
        portfolio.calculate_cumulative_returns()
        
        # Calculate metrics
        metrics = portfolio.calculate_metrics(risk_free_rate=0.0)
        results['metrics'][factor_name] = metrics
        results['portfolio_returns'][factor_name] = portfolio_returns
        
        # Save plots
        factor_dir = os.path.join(run_dir, factor_name)
        ensure_dir(factor_dir)
        
        # Cumulative returns plot
        fig = portfolio.plot_cumulative_returns(benchmark_returns=sp500_returns)
        fig.savefig(os.path.join(factor_dir, 'cumulative_returns.png'))
        plt.close(fig)
        
        # Drawdown plot
        fig = portfolio.plot_drawdown()
        fig.savefig(os.path.join(factor_dir, 'drawdowns.png'))
        plt.close(fig)
        
        # Calculate IC
        ic_series = portfolio.calculate_ic(factor_signal, market_data['returns'], horizon=1)
        
        # IC time series plot
        fig = viz.plot_ic_time_series(ic_series, factor_name=factor_name)
        fig.savefig(os.path.join(factor_dir, 'ic_time_series.png'))
        plt.close(fig)
        
        # IC histogram plot
        fig = viz.plot_ic_histogram(ic_series, factor_name=factor_name)
        fig.savefig(os.path.join(factor_dir, 'ic_histogram.png'))
        plt.close(fig)
        
        # Rolling Sharpe ratio plot
        fig = viz.plot_rolling_sharpe(portfolio_returns, window=63, factor_name=factor_name)
        fig.savefig(os.path.join(factor_dir, 'rolling_sharpe.png'))
        plt.close(fig)
        
        # Monthly returns heatmap
        fig = viz.plot_monthly_returns_heatmap(portfolio_returns, factor_name=factor_name)
        fig.savefig(os.path.join(factor_dir, 'monthly_returns.png'))
        plt.close(fig)
        
        # Save metrics to text file
        with open(os.path.join(factor_dir, 'metrics.txt'), 'w') as f:
            f.write(f"{factor_name} Factor Performance Metrics\n")
            f.write("=" * 40 + "\n\n")
            
            # Format metrics for readability
            f.write(f"Total Return: {metrics['total_return']:.2%}\n")
            f.write(f"Annualized Return: {metrics['annualized_return']:.2%}\n")
            f.write(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"Max Drawdown: {metrics['max_drawdown']:.2%}\n")
            f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
            f.write(f"Average Daily Return: {portfolio_returns.mean():.4%}\n")
            f.write(f"t-statistic: {metrics['t_statistic']:.2f}\n")
            f.write(f"p-value: {metrics['p_value']:.4f}\n")
            
            if metrics['turnover'] is not None:
                f.write(f"Average Turnover: {metrics['turnover']:.2%}\n")
                
            f.write("\nIC Statistics:\n")
            f.write(f"Mean IC: {ic_series.mean():.4f}\n")
            f.write(f"IC t-statistic: {(ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series)))):.2f}\n")
            f.write(f"IC > 0: {(ic_series > 0).mean():.2%}\n")
    
    # 4. Robustness analysis
    print("\n4. Running robustness analysis...")
    robustness = RobustnessAnalysis()
    
    # Factor correlation analysis
    factor_correlation = robustness.calculate_factor_correlation(results['factors'])
    
    # Create correlation heatmap
    fig = robustness.plot_correlation_heatmap(factor_correlation)
    fig.savefig(os.path.join(run_dir, 'factor_correlation.png'))
    plt.close(fig)
    
    # Calculate IC decay for each factor
    ic_decay_results = {}
    for factor_name, factor_signal in results['factors'].items():
        ic_decay = robustness.calculate_ic_decay(factor_signal, market_data['returns'])
        ic_decay_results[factor_name] = ic_decay
        
        # Plot IC decay
        fig = robustness.plot_ic_decay(ic_decay, factor_name=factor_name)
        fig.savefig(os.path.join(run_dir, f'{factor_name}_ic_decay.png'))
        plt.close(fig)
    
    # Calculate turnover for each factor
    turnover_results = {}
    for factor_name, factor_signal in results['factors'].items():
        turnover = robustness.calculate_turnover(factor_signal)
        turnover_results[factor_name] = turnover
        
        # Plot turnover distribution
        fig = robustness.plot_turnover_distribution(turnover, factor_name=factor_name)
        fig.savefig(os.path.join(run_dir, f'{factor_name}_turnover.png'))
        plt.close(fig)
    
    # 5. Save combined results
    print("\n5. Generating combined results...")
    
    # Create a combined cumulative returns plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each factor's cumulative returns
    for factor_name, returns in results['portfolio_returns'].items():
        cum_returns = (1 + returns).cumprod() - 1
        cum_returns.plot(ax=ax, linewidth=2, label=factor_name)
    
    # Plot S&P 500 for reference
    sp500_cum_returns = (1 + sp500_returns).cumprod() - 1
    sp500_cum_returns.plot(ax=ax, color='black', linestyle='--', linewidth=1, label='S&P 500')
    
    ax.set_title('Cumulative Returns Comparison')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.savefig(os.path.join(run_dir, 'combined_returns.png'))
    plt.close(fig)
    
    # Create a summary table of metrics
    metrics_df = pd.DataFrame()
    
    for factor_name, metrics in results['metrics'].items():
        metrics_series = pd.Series(metrics, name=factor_name)
        metrics_df = pd.concat([metrics_df, metrics_series], axis=1)
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(run_dir, 'metrics_summary.csv'))
    
    # Create a text summary file
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write("Alpha Factor Research Summary\n")
        f.write("==========================\n\n")
        
        f.write(f"Analysis period: {start_date} to {end_date}\n")
        f.write(f"Number of stocks: {len(data_loader.tickers)}\n")
        f.write(f"Transaction costs: {transaction_costs:.2%}\n\n")
        
        f.write("Factor Parameters:\n")
        f.write(f"- Momentum: {momentum_lookback} day lookback, {momentum_skip} day skip\n")
        f.write(f"- Reversal: {reversal_lookback} day lookback\n")
        f.write(f"- Volume Surprise: {volume_lookback} day volume lookback\n\n")
        
        f.write("Performance Summary:\n")
        for factor_name, metrics in results['metrics'].items():
            f.write(f"- {factor_name}:\n")
            f.write(f"  Annualized Return: {metrics['annualized_return']:.2%}\n")
            f.write(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"  Max Drawdown: {metrics['max_drawdown']:.2%}\n")
            f.write(f"  t-statistic: {metrics['t_statistic']:.2f}\n\n")
        
        f.write("Factor Correlations:\n")
        for i in range(len(factor_correlation)):
            for j in range(i+1, len(factor_correlation)):
                f.write(f"- {factor_correlation.index[i]} vs {factor_correlation.index[j]}: {factor_correlation.iloc[i, j]:.2f}\n")
    
    print(f"\nAnalysis complete. Results saved to {run_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run alpha factor research pipeline')
    parser.add_argument('--start-date', help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', help='End date in YYYY-MM-DD format')
    parser.add_argument('--momentum-lookback', type=int, default=252, help='Lookback period for momentum factor')
    parser.add_argument('--momentum-skip', type=int, default=21, help='Skip period for momentum factor')
    parser.add_argument('--reversal-lookback', type=int, default=5, help='Lookback period for reversal factor')
    parser.add_argument('--volume-lookback', type=int, default=20, help='Lookback period for volume surprise factor')
    parser.add_argument('--num-stocks', type=int, default=100, help='Number of stocks to analyze')
    parser.add_argument('--transaction-costs', type=float, default=0.001, help='Transaction costs')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    
    args = parser.parse_args()
    
    run_factor_analysis(
        start_date=args.start_date,
        end_date=args.end_date,
        momentum_lookback=args.momentum_lookback,
        momentum_skip=args.momentum_skip,
        reversal_lookback=args.reversal_lookback,
        volume_lookback=args.volume_lookback,
        num_stocks=args.num_stocks,
        transaction_costs=args.transaction_costs,
        output_dir=args.output_dir
    ) 