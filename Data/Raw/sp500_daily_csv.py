import pandas as pd
import yfinance as yf
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check pandas version
required_pandas_version = "1.0.0"
if pd.__version__ < required_pandas_version:
    logger.error(f"pandas version {pd.__version__} is outdated. Please upgrade to {required_pandas_version} or later.")
    raise ImportError(f"pandas version {pd.__version__} is outdated. Please upgrade to {required_pandas_version} or later.")

def download_sp500_data(ticker='^GSPC', start_date='2020-07-20', end_date='2025-07-20', output_path='data/raw/sp500_raw.csv'):
    """
    Download daily S&P 500 data using yfinance and save to CSV.
    
    Args:
        ticker (str): Ticker symbol for S&P 500 (^GSPC).
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        output_path (str): Path to save the CSV file.
    
    Returns:
        pd.DataFrame: Downloaded data.
    """
    try:
        logger.info(f"Downloading daily data for {ticker} from {start_date} to {end_date}")
        # Download data with yfinance
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=False)
        
        if df.empty:
            logger.error("No data downloaded. Check ticker or date range.")
            raise ValueError("No data downloaded.")
        
        # Log raw columns
        logger.info(f"Downloaded DataFrame columns: {list(df.columns)}")
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Multi-level columns detected. Flattening to single level.")
            df.columns = [col[0] for col in df.columns]  # Use first level (e.g., 'Close' from ('Close', '^GSPC'))
        
        # Verify required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
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
        
        if df.empty:
            logger.error("DataFrame is empty after cleaning.")
            raise ValueError("DataFrame is empty after cleaning.")
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path)
        logger.info(f"Data saved to {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading or saving data: {str(e)}")
        raise

def main():
    """Main function to download and save S&P 500 data."""
    try:
        df = download_sp500_data()
        print("Data head:\n", df.head())
        return df
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    df = main()
