import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check pandas version
required_pandas_version = "1.0.0"
if pd.__version__ < required_pandas_version:
    logger.error(f"pandas version {pd.__version__} is outdated. Please upgrade to {required_pandas_version} or later.")
    raise ImportError(f"pandas version {pd.__version__} is outdated. Please upgrade to {required_pandas_version} or later.")

def calculate_rsi(data, periods=14, column='Close'):
    """
    Calculate Relative Strength Index (RSI) for a given price column.
    
    Args:
        data (pd.Series): Price data (e.g., Close prices).
        periods (int): Number of periods for RSI calculation.
        column (str): Column name to use for RSI.
    
    Returns:
        pd.Series: RSI values.
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def engineer_features(df, is_intraday=False):
    """
    Engineer features for daily or intraday data.
    
    Args:
        df (pd.DataFrame): Input DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
        is_intraday (bool): True for intraday data, False for daily data.
    
    Returns:
        pd.DataFrame: DataFrame with additional feature columns.
    """
    try:
        logger.info(f"Engineering features for {'intraday' if is_intraday else 'daily'} data")
        df = df.copy()  # Avoid modifying the input DataFrame
        
        # Ensure required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing columns for feature engineering: {missing_cols}")
            raise ValueError(f"Missing columns for feature engineering: {missing_cols}")
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change() * 100  # Percentage returns
        
        # Calculate price range
        df['Price_Range'] = df['High'] - df['Low']
        
        # Calculate moving averages and volatility
        if is_intraday:
            # Intraday: Use 10-minute and 60-minute windows
            df['MA_10min'] = df['Close'].rolling(window=10).mean()
            df['MA_60min'] = df['Close'].rolling(window=60).mean()
            df['Volatility_60min'] = df['Returns'].rolling(window=60).std()
        else:
            # Daily: Use 5-day and 20-day windows
            df['MA_5d'] = df['Close'].rolling(window=5).mean()
            df['MA_20d'] = df['Close'].rolling(window=20).mean()
            df['Volatility_20d'] = df['Returns'].rolling(window=20).std()
        
        # Calculate RSI
        df['RSI'] = calculate_rsi(df['Close'], periods=14)
        
        # Drop rows with NaN values from feature calculations
        df.dropna(inplace=True)
        
        logger.info(f"Feature engineering completed. New columns: {list(df.columns)}")
        return df
    
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

def download_daily_data(ticker='^GSPC', start_date='2020-07-20', end_date='2025-07-20'):
    """
    Download daily S&P 500 data using yfinance, clean it, and engineer features.
    
    Args:
        ticker (str): Ticker symbol for S&P 500 (^GSPC).
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
    
    Returns:
        pd.DataFrame: Cleaned and feature-engineered daily data.
    """
    try:
        logger.info(f"Downloading daily data for {ticker} from {start_date} to {end_date}")
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=False)
        
        if df.empty:
            logger.error("No data downloaded. Check ticker or date range.")
            raise ValueError("No data downloaded.")
        
        logger.info(f"Downloaded DataFrame columns: {list(df.columns)}")
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Multi-level columns detected. Flattening to single level.")
            df.columns = [col[0] for col in df.columns]  # Use first level (e.g., 'Close' from ('Close', '^GSPC'))
        
        # Verify required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing columns in downloaded data: {missing_cols}")
            raise ValueError(f"Missing columns in downloaded data: {missing_cols}")
        
        # Clean data
        df = df[required_columns].copy()
        df.dropna(inplace=True)
        
        # Ensure Close is numeric
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if df['Close'].isna().any():
            logger.warning("Non-numeric values found in Close column after conversion. Dropping invalid rows.")
            df.dropna(subset=['Close'], inplace=True)
        
        if not df.empty:
            # Remove outliers
            close_mean = df['Close'].mean()
            close_std = df['Close'].std()
            df = df[df['Close'].between(close_mean - 3 * close_std, close_mean + 3 * close_std)]
        else:
            logger.warning("DataFrame is empty after cleaning. No data to process.")
        
        # Engineer features
        df = engineer_features(df, is_intraday=False)
        
        # Save to CSV
        os.makedirs('data/raw', exist_ok=True)
        output_path = 'data/raw/sp500_daily_features.csv'
        df.to_csv(output_path)
        logger.info(f"Daily data with features saved to {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading daily data: {str(e)}")
        raise

def download_intraday_data(ticker='^GSPC'):
    """
    Download intra-day S&P 500 data using yfinance, clean it, and engineer features.
    
    Args:
        ticker (str): Ticker symbol for S&P 500 (^GSPC).
    
    Returns:
        pd.DataFrame: Intra-day data with features (1-minute intervals).
    """
    try:
        logger.info(f"Downloading intra-day data for {ticker}")
        df = yf.download(ticker, interval='1m', period='7d', auto_adjust=False)
        
        if df.empty:
            logger.error("No intra-day data retrieved. Check ticker or yfinance limits.")
            raise ValueError("No intra-day data retrieved.")
        
        logger.info(f"Downloaded intra-day DataFrame columns: {list(df.columns)}")
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Multi-level columns detected in intra-day data. Flattening to single level.")
            df.columns = [col[0] for col in df.columns]
        
        # Verify required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing columns in intra-day data: {missing_cols}")
            raise ValueError(f"Missing columns in intra-day data: {missing_cols}")
        
        # Ensure UTC timezone
        df.index = df.index.tz_convert('UTC')
        
        # Clean data
        df = df[required_columns].copy()
        df.dropna(inplace=True)
        
        # Ensure Close is numeric
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if df['Close'].isna().any():
            logger.warning("Non-numeric values found in Close column after conversion. Dropping invalid rows.")
            df.dropna(subset=['Close'], inplace=True)
        
        if not df.empty:
            # Remove outliers
            close_mean = df['Close'].mean()
            close_std = df['Close'].std()
            df = df[df['Close'].between(close_mean - 3 * close_std, close_mean + 3 * close_std)]
        
        # Engineer features
        df = engineer_features(df, is_intraday=True)
        
        # Save updated cache
        os.makedirs('data/intraday', exist_ok=True)
        output_path = f'data/intraday/{ticker}_intraday_features.csv'
        df.to_csv(output_path)
        logger.info(f"Intra-day data with features saved to {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading intra-day data: {str(e)}")
        raise

def main():
    """Main function to download, clean, and engineer features for both daily and intra-day data."""
    try:
        # Download and process daily data
        daily_df = download_daily_data()
        logger.info(f"Daily data shape: {daily_df.shape}")
        
        # Download and process intra-day data
        intraday_df = download_intraday_data()
        logger.info(f"Intra-day data shape: {intraday_df.shape}")
        
        return daily_df, intraday_df
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    daily_data, intraday_data = main()
    print("Daily data head:\n", daily_data.head())
    print("Intra-day data head:\n", intraday_data.head())
