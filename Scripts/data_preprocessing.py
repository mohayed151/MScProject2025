import pandas as pd
import yfinance as yf
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_5min_intraday_60days(ticker='^GSPC', period='60d', interval='5m'):
    """
    Download 5-minute intraday S&P 500 data for up to 60 days.
    
    Args:
        ticker (str): Ticker symbol (e.g., '^GSPC').
        period (str): Period for data retrieval (e.g., '60d').
        interval (str): Data interval (e.g., '5m' for 5-minute).
    
    Returns:
        pd.DataFrame: 5-minute intraday data.
    """
    try:
        logger.info(f"Downloading 5-minute intraday data for {ticker} (period={period}, interval={interval})")
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
        
        if df.empty:
            logger.error("No data downloaded. Check ticker, period, or interval.")
            raise ValueError("No data downloaded.")
        
        # Log DataFrame structure
        logger.info(f"DataFrame columns: {list(df.columns)}")
        logger.info(f"DataFrame shape: {df.shape}")
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Multi-level columns detected. Flattening to single level.")
            df.columns = [col[0] for col in df.columns]
        
        # Verify required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing columns in downloaded data: {missing_cols}")
        
        # Clean data
        df = df[required_columns].copy()
        df.dropna(inplace=True)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if df['Close'].isna().any():
            logger.warning("Non-numeric values found in Close column. Dropping invalid rows.")
            df.dropna(subset=['Close'], inplace=True)
        df.index = df.index.tz_convert('UTC')
        
        # Save to CSV
        os.makedirs('data/intraday', exist_ok=True)
        output_path = f'data/intraday/{ticker}_intraday_5min_60days.csv'
        df.to_csv(output_path)
        logger.info(f"Data saved to {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading 5-minute intraday data: {str(e)}")
        raise

if __name__ == "__main__":
    intraday_data = download_5min_intraday_60days(ticker='^GSPC', period='60d', interval='5m')
    print("5-minute data head:\n", intraday_data.head())
    print("Columns:", list(intraday_data.columns))
