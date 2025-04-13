# Quantitative Alpha Factor Research

This project focuses on researching, implementing, and evaluating alpha factors for equity trading.

## Alpha Factors Implemented

1. Momentum (Price Momentum)
2. Mean Reversion (Short-term Reversal)
3. Volume Surprise (Volume-Price Relationship)

## Project Structure

- `data/`: Data fetching and preprocessing
- `factors/`: Implementation of alpha factor calculations
- `analysis/`: Performance evaluation and statistical testing
- `visualization/`: Charts and visualization tools
- `utils/`: Helper functions and utilities

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the main pipeline:
```
python main.py
```

## Features

- Factor calculation across stocks
- Securities ranking based on signal strength
- Long-short portfolio simulation
- Performance metrics: Sharpe ratio, t-statistics, IC, drawdown
- Robustness tests including signal decay, turnover analysis
- Transaction cost modeling 