# Quantitative Alpha Factor Research Project

A comprehensive research framework for discovering, evaluating, and analyzing alpha factors in equity markets.

## Overview

This project provides tools to:

1. Research and implement alpha factors (momentum, mean reversion, volume surprise)
2. Calculate these factors across stocks
3. Rank securities based on signal strength
4. Run long-short or market-neutral portfolios
5. Measure performance with key metrics (Sharpe, t-stat, IC, drawdown)
6. Perform robustness tests (decay, turnover, correlation with other signals)
7. Model transaction costs

## Project Structure

```
alpharesearch/
├── data/               # Data acquisition and preprocessing
├── factors/            # Alpha factor implementations
├── analysis/           # Portfolio construction and metrics
├── visualization/      # Plotting and visualization tools
├── utils/              # Helper functions and utilities
└── main.py             # Main execution script
```

## Alpha Factors Implemented

1. **Momentum (Price Momentum)**: Captures medium to long-term price trends
2. **Mean Reversion (Short-term Reversal)**: Captures short-term price reversals
3. **Volume Surprise**: Captures volume-price relationship patterns

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/alpharesearch.git
cd alpharesearch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the analysis with default parameters:

```bash
python -m alpharesearch.main
```

### Custom Analysis

Customize parameters for a more specific analysis:

```bash
python -m alpharesearch.main --start-date 2018-01-01 --end-date 2023-01-01 --num-stocks 50 --momentum-lookback 126 --transaction-costs 0.002
```

### Parameters

- `--start-date`: Analysis start date (YYYY-MM-DD)
- `--end-date`: Analysis end date (YYYY-MM-DD)
- `--momentum-lookback`: Lookback period for momentum factor (days)
- `--momentum-skip`: Skip period for momentum factor (days)
- `--reversal-lookback`: Lookback period for reversal factor (days)
- `--volume-lookback`: Lookback period for volume surprise factor (days)
- `--num-stocks`: Number of stocks to analyze
- `--transaction-costs`: Transaction costs as fraction of traded value
- `--output-dir`: Directory to save results

## Example Output

After running the analysis, you'll get:

1. **Portfolio Performance**: Cumulative returns, drawdowns, rolling Sharpe ratios
2. **Factor Analysis**: IC time series, IC histograms, monthly returns heatmaps
3. **Robustness Tests**: IC decay, turnover distribution, factor correlations
4. **Summary Statistics**: Performance metrics, t-statistics, p-values

Results are saved to the `results/` directory (or your specified output directory).

## Extending the Project

To add a new alpha factor:

1. Create a new class inheriting from `BaseFactor` in the `factors` directory
2. Implement the `compute()` method
3. Add the factor to the main execution pipeline

## License

This project is released under the MIT License.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice and should not be used for trading without extensive testing. 