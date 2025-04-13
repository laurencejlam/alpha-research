import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime


def ensure_dir(directory):
    """
    Create directory if it doesn't exist
    
    Parameters:
    -----------
    directory : str
        Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_data(data, filename, directory='data/processed'):
    """
    Save data to pickle file
    
    Parameters:
    -----------
    data : object
        Data to save
    filename : str
        Filename
    directory : str
        Directory to save to
    """
    ensure_dir(directory)
    with open(os.path.join(directory, filename), 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {os.path.join(directory, filename)}")


def load_data(filename, directory='data/processed'):
    """
    Load data from pickle file
    
    Parameters:
    -----------
    filename : str
        Filename
    directory : str
        Directory to load from
        
    Returns:
    --------
    object
        Loaded data
    """
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
        
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_timestamps():
    """
    Get current timestamp string
    
    Returns:
    --------
    str
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def winsorize(data, limits=(0.01, 0.01)):
    """
    Winsorize data to remove outliers
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Data to winsorize
    limits : tuple
        (lower percentile, upper percentile) to clip
        
    Returns:
    --------
    pd.DataFrame or pd.Series
        Winsorized data
    """
    if isinstance(data, pd.DataFrame):
        return data.apply(lambda x: winsorize(x, limits), axis=0)
    elif isinstance(data, pd.Series):
        lower_percentile = data.quantile(limits[0])
        upper_percentile = data.quantile(1 - limits[1])
        return data.clip(lower=lower_percentile, upper=upper_percentile)
    else:
        raise TypeError("Data must be a pandas DataFrame or Series")


def neutralize(data, factor):
    """
    Neutralize data with respect to a factor
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to neutralize
    factor : pd.DataFrame
        Factor to neutralize against
        
    Returns:
    --------
    pd.DataFrame
        Neutralized data
    """
    neutralized = pd.DataFrame(index=data.index, columns=data.columns)
    
    for date in data.index:
        if date in factor.index:
            y = data.loc[date].dropna()
            x = factor.loc[date, y.index].dropna()
            
            # Get common assets
            common_assets = y.index.intersection(x.index)
            
            if len(common_assets) > 5:  # Require at least 5 assets
                y_common = y[common_assets]
                x_common = x[common_assets]
                
                # Simple linear regression
                x_with_const = pd.DataFrame({'const': np.ones(len(x_common)), 'factor': x_common})
                beta = np.linalg.pinv(x_with_const.T @ x_with_const) @ (x_with_const.T @ y_common)
                
                # Residuals
                y_pred = beta[0] + beta[1] * x_common
                residuals = y_common - y_pred
                
                neutralized.loc[date, common_assets] = residuals
                
    return neutralized


def get_quantile_returns(returns, quantiles, num_quantiles=5):
    """
    Calculate returns by quantile
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    quantiles : pd.DataFrame
        Quantile assignments for each asset
    num_quantiles : int
        Number of quantiles
        
    Returns:
    --------
    dict
        Dictionary of {quantile: returns_series}
    """
    quantile_returns = {}
    
    for q in range(num_quantiles):
        # Create portfolio of assets in this quantile
        is_in_quantile = quantiles == q
        
        # Calculate equal-weighted returns
        q_returns = []
        
        for date in returns.index:
            if date in is_in_quantile.index:
                # Get assets in this quantile on this date
                assets_in_q = is_in_quantile.loc[date]
                assets_in_q = assets_in_q[assets_in_q].index
                
                if len(assets_in_q) > 0:
                    # Calculate equal-weighted return
                    q_return = returns.loc[date, assets_in_q].mean()
                    q_returns.append((date, q_return))
        
        if q_returns:
            quantile_returns[q] = pd.Series([r for _, r in q_returns], 
                                           index=[d for d, _ in q_returns])
    
    return quantile_returns


def calculate_drawdowns(returns):
    """
    Calculate drawdowns from returns
    
    Parameters:
    -----------
    returns : pd.Series
        Returns series
        
    Returns:
    --------
    pd.Series
        Drawdowns series
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative / running_max - 1)
    return drawdowns


def calculate_cagr(returns):
    """
    Calculate Compound Annual Growth Rate
    
    Parameters:
    -----------
    returns : pd.Series
        Returns series
        
    Returns:
    --------
    float
        CAGR
    """
    total_return = (1 + returns).prod() - 1
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1
    return cagr 