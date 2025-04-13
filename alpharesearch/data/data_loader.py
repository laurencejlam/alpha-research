import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm

class DataLoader:
    def __init__(self, tickers=None, start_date=None, end_date=None, data_dir='data/market_data'):
        """
        Initialize the DataLoader with tickers and date range
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        data_dir : str
            Directory to save/load data
        """
        self.tickers = tickers if tickers else []
        self.start_date = start_date if start_date else (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
    def add_tickers(self, tickers):
        """Add tickers to the current ticker list"""
        if isinstance(tickers, str):
            tickers = [tickers]
        self.tickers.extend([ticker for ticker in tickers if ticker not in self.tickers])
        
    def fetch_data(self, force_download=False):
        """
        Fetch data for all tickers from Yahoo Finance
        
        Parameters:
        -----------
        force_download : bool
            If True, download new data even if cached data exists
            
        Returns:
        --------
        dict
            Dictionary with keys 'prices', 'volumes', 'returns'
        """
        cache_file = os.path.join(self.data_dir, f"market_data_{self.start_date}_{self.end_date}.pkl")
        
        # Check if cached data exists
        if os.path.exists(cache_file) and not force_download:
            print(f"Loading cached data from {cache_file}")
            return pd.read_pickle(cache_file)
        
        print(f"Fetching data for {len(self.tickers)} tickers...")
        
        # Fetch data for each ticker
        all_data = {}
        for ticker in tqdm(self.tickers):
            try:
                data = yf.download(
                    ticker, 
                    start=self.start_date, 
                    end=self.end_date, 
                    progress=False
                )
                all_data[ticker] = data
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        # Organize data into price, volume, and returns dataframes
        prices = pd.DataFrame({ticker: data['Adj Close'] for ticker, data in all_data.items()})
        volumes = pd.DataFrame({ticker: data['Volume'] for ticker, data in all_data.items()})
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        result = {
            'prices': prices,
            'volumes': volumes,
            'returns': returns
        }
        
        # Cache the data
        pd.to_pickle(result, cache_file)
        print(f"Data saved to {cache_file}")
        
        return result
    
    def get_sp500_tickers(self):
        """
        Get tickers for S&P 500 companies
        
        Returns:
        --------
        list
            List of S&P 500 tickers
        """
        try:
            # Fetch S&P 500 components from Wikipedia
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            sp500_df = table[0]
            tickers = sp500_df['Symbol'].tolist()
            
            # Clean tickers (remove special characters)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            self.tickers = tickers
            return tickers
        except Exception as e:
            print(f"Error fetching S&P 500 tickers: {e}")
            return []
    
    def get_market_data(self, market_ticker='^GSPC'):
        """
        Get market (benchmark) data
        
        Parameters:
        -----------
        market_ticker : str
            Ticker symbol for the market index
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with market data
        """
        try:
            market_data = yf.download(
                market_ticker, 
                start=self.start_date, 
                end=self.end_date, 
                progress=False
            )
            return market_data
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return pd.DataFrame() 