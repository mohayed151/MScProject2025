import pandas as pd
import yfinance as yf
import intraday
from datetime import datetime, timedelta
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_daily_data(ticker='^GSPC', start_date='2020-07-20', end_date='2025-07-20'):
    """
    Download daily S&P 500 data using yfinance and clean it.
    
    Args:
        ticker (str): Ticker symbol for S&P 500 (^GSPC).
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
    
    Returns:
        pd.DataFrame: Cleaned daily data.
    """
    try:
        logger.info(f"Downloading daily data for {ticker} from {start_date} to {end_date}")
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            logger.error("No data downloaded. Check ticker or date range.")
            raise ValueError("No data downloaded.")
        
        # Clean data
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.dropna(inplace=True)  # Remove rows with missing values
        
        # Remove outliers (e.g., price anomalies > 3 standard deviations)
        df = df[df['Close'].between(df['Close'].mean() - 3 * df['Close'].std(),
                                    df['Close'].mean() + 3 * df['Close'].std())]
        
        # Save to CSV
        os.makedirs('data/raw', exist_ok=True)
        output_path = 'data/raw/sp500_daily.csv'
        df.to_csv(output_path)
        logger.info(f"Daily data saved to {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading daily data: {str(e)}")
        raise

def download_intraday_data(ticker='^GSPC'):
    """
    Download and cache intra-day S&P 500 data using intraday module.
    
    Args:
        ticker (str): Ticker symbol for S&P 500 (^GSPC).
    
    Returns:
        pd.DataFrame: Cached intra-day data (1-minute intervals).
    """
    try:
        logger.info(f"Updating intra-day data for {ticker}")
        df = intraday.update_ticker(ticker)
        
        if df is None or df.empty:
            logger.error("No intra-day data retrieved. Check intraday module or cache.")
            raise ValueError("No intra-day data retrieved.")
        
        # Ensure UTC timezone for consistency
        df.index = df.index.tz_convert('UTC')
        
        # Clean data
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.dropna(inplace=True)
        
        # Remove outliers
        df = df[df['Close'].between(df['Close'].mean() - 3 * df['Close'].std(),
                                    df['Close'].mean() + 3 * df['Close'].std())]
        
        # Save updated cache
        os.makedirs('data/intraday', exist_ok=True)
        output_path = f'data/intraday/{ticker}_intraday.csv'
        df.to_csv(output_path)
        logger.info(f"Intra-day data saved to {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading intra-day data: {str(e)}")
        raise

def main():
    """Main function to download and clean both daily and intra-day data."""
    try:
        # Download daily data
        daily_df = download_daily_data()
        logger.info(f"Daily data shape: {daily_df.shape}")
        
        # Download intra-day data
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
