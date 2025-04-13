#!/usr/bin/env python
"""
Minimal example script for alpha factor research
"""

from datetime import datetime, timedelta
from alpharesearch.data import DataLoader
from alpharesearch.factors import Momentum
from alpharesearch.analysis import Portfolio

# Set date range (last 2 years)
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')

# Define a small set of stocks to analyze
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']

print(f"Analyzing {len(tickers)} stocks from {start_date} to {end_date}")

# Load data
data_loader = DataLoader(tickers=tickers, start_date=start_date, end_date=end_date)
market_data = data_loader.fetch_data()

# Compute momentum factor
momentum = Momentum(lookback_period=126, skip_period=21)
momentum_signal = momentum.compute(market_data)

# Create a momentum portfolio
portfolio = Portfolio(name="Momentum")
weights = portfolio.calculate_weights_from_signal(momentum_signal, long_short=True)
returns = portfolio.calculate_portfolio_returns(weights, market_data['returns'])
portfolio.calculate_cumulative_returns()
metrics = portfolio.calculate_metrics()

# Print performance metrics
print("\nMomentum Portfolio Performance:")
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Annualized Return: {metrics['annualized_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}") 