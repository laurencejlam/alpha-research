#!/usr/bin/env python
"""
Simple example script demonstrating the alpha factor research pipeline
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from alpharesearch.data import DataLoader
from alpharesearch.factors import Momentum, MeanReversion, VolumeSurprise
from alpharesearch.analysis import Portfolio


def run_example():
    """Run a simple example of the alpha factor research pipeline"""
    
    print("Alpha Factor Research Example")
    print("=============================\n")
    
    # Set date range (last 2 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    
    # Define a small set of stocks to analyze
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
    
    print(f"Analyzing {len(tickers)} stocks from {start_date} to {end_date}")
    
    # Load data
    data_loader = DataLoader(tickers=tickers, start_date=start_date, end_date=end_date)
    market_data = data_loader.fetch_data()
    
    # Compute factors
    print("\nComputing alpha factors...")
    
    # Momentum factor (6-month lookback, 1-month skip)
    momentum = Momentum(lookback_period=126, skip_period=21)
    momentum_signal = momentum.compute(market_data)
    
    # Mean Reversion factor (1-week lookback)
    mean_reversion = MeanReversion(lookback_period=5)
    reversal_signal = mean_reversion.compute(market_data)
    
    # Volume Surprise factor
    volume_surprise = VolumeSurprise(volume_lookback=20)
    volume_signal = volume_surprise.compute(market_data)
    
    # Create portfolios
    print("\nCreating portfolios...")
    
    # Create a momentum portfolio
    momentum_portfolio = Portfolio(name="Momentum")
    momentum_weights = momentum_portfolio.calculate_weights_from_signal(
        momentum_signal, 
        long_short=True
    )
    momentum_returns = momentum_portfolio.calculate_portfolio_returns(
        momentum_weights, 
        market_data['returns'],
        transaction_costs=0.001
    )
    momentum_cumulative = momentum_portfolio.calculate_cumulative_returns()
    momentum_metrics = momentum_portfolio.calculate_metrics()
    
    # Print momentum metrics
    print("\nMomentum Portfolio Performance:")
    print(f"Total Return: {momentum_metrics['total_return']:.2%}")
    print(f"Annualized Return: {momentum_metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {momentum_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {momentum_metrics['max_drawdown']:.2%}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    run_example() 