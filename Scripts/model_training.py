import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

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
        df = df.copy()
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing columns for feature engineering: {missing_cols}")
            raise ValueError(f"Missing columns for feature engineering: {missing_cols}")
        
        df['Returns'] = df['Close'].pct_change() * 100
        df['Price_Range'] = df['High'] - df['Low']
        
        if is_intraday:
            df['MA_10min'] = df['Close'].rolling(window=10).mean()
            df['MA_60min'] = df['Close'].rolling(window=60).mean()
            df['Volatility_60min'] = df['Returns'].rolling(window=60).std()
        else:
            df['MA_5d'] = df['Close'].rolling(window=5).mean()
            df['MA_20d'] = df['Close'].rolling(window=20).mean()
            df['Volatility_20d'] = df['Returns'].rolling(window=20).std()
        
        df['RSI'] = calculate_rsi(df['Close'], periods=14)
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
        
        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Multi-level columns detected. Flattening to single level.")
            df.columns = [col[0] for col in df.columns]
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing columns in downloaded data: {missing_cols}")
            raise ValueError(f"Missing columns in downloaded data: {missing_cols}")
        
        df = df[required_columns].copy()
        df.dropna(inplace=True)
        
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if df['Close'].isna().any():
            logger.warning("Non-numeric values found in Close column after conversion. Dropping invalid rows.")
            df.dropna(subset=['Close'], inplace=True)
        
        if not df.empty:
            close_mean = df['Close'].mean()
            close_std = df['Close'].std()
            df = df[df['Close'].between(close_mean - 3 * close_std, close_mean + 3 * close_std)]
        
        df = engineer_features(df, is_intraday=False)
        
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
        
        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Multi-level columns detected in intra-day data. Flattening to single level.")
            df.columns = [col[0] for col in df.columns]
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing columns in intra-day data: {missing_cols}")
            raise ValueError(f"Missing columns in intra-day data: {missing_cols}")
        
        df.index = df.index.tz_convert('UTC')
        df = df[required_columns].copy()
        df.dropna(inplace=True)
        
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if df['Close'].isna().any():
            logger.warning("Non-numeric values found in Close column after conversion. Dropping invalid rows.")
            df.dropna(subset=['Close'], inplace=True)
        
        if not df.empty:
            close_mean = df['Close'].mean()
            close_std = df['Close'].std()
            df = df[df['Close'].between(close_mean - 3 * close_std, close_mean + 3 * close_std)]
        
        df = engineer_features(df, is_intraday=True)
        
        os.makedirs('data/intraday', exist_ok=True)
        output_path = f'data/intraday/{ticker}_intraday_features.csv'
        df.to_csv(output_path)
        logger.info(f"Intra-day data with features saved to {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading intra-day data: {str(e)}")
        raise

def prepare_data_for_training(df):
    """
    Prepare data for model training by creating features and target.
    
    Args:
        df (pd.DataFrame): DataFrame with engineered features.
    
    Returns:
        pd.DataFrame: DataFrame with features and target.
    """
    try:
        logger.info("Preparing data for training")
        df = df.copy()
        
        # Create target: 1 if next day's Close > current Close, else 0
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)  # Drop rows with NaN target
        
        # Features to use
        feature_columns = ['Returns', 'Price_Range', 'MA_5d', 'MA_20d', 'Volatility_20d', 'RSI']
        if not all(col in df.columns for col in feature_columns):
            missing_cols = [col for col in feature_columns if col not in df.columns]
            logger.error(f"Missing feature columns for training: {missing_cols}")
            raise ValueError(f"Missing feature columns for training: {missing_cols}")
        
        logger.info(f"Prepared data shape: {df.shape}")
        return df, feature_columns  # Return DataFrame and feature columns
    
    except Exception as e:
        logger.error(f"Error preparing data for training: {str(e)}")
        raise

def train_model(df, feature_columns):
    """
    Train a Random Forest Classifier on the prepared data.
    
    Args:
        df (pd.DataFrame): DataFrame with features and target.
        feature_columns (list): List of feature column names.
    
    Returns:
        tuple: (trained model, test accuracy, classification report)
    """
    try:
        logger.info("Training Random Forest Classifier")
        
        # Split features and target
        X = df[feature_columns]
        y = df['Target']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Down/No Change', 'Up'])
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{report}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/rf_model.joblib'
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model, accuracy, report
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def main():
    """Main function to download, process, and train model."""
    try:
        # Download and process daily data
        daily_df = download_daily_data()
        logger.info(f"Daily data shape: {daily_df.shape}")
        
        # Download intraday data (for reference)
        intraday_df = download_intraday_data()
        logger.info(f"Intra-day data shape: {intraday_df.shape}")
        
        # Prepare data for training
        train_data, feature_columns = prepare_data_for_training(daily_df)
        
        # Save prepared data
        os.makedirs('data/processed', exist_ok=True)
        train_data_path = 'data/processed/sp500_train_data.csv'
        train_data.to_csv(train_data_path)
        logger.info(f"Training data saved to {train_data_path}")
        
        # Train model
        model, accuracy, report = train_model(train_data, feature_columns)
        
        # Save feature columns for backtesting
        feature_columns_path = 'models/feature_columns.txt'
        with open(feature_columns_path, 'w') as f:
            f.write(','.join(feature_columns))
        logger.info(f"Feature columns saved to {feature_columns_path}")
        
        return daily_df, intraday_df, model, feature_columns
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    daily_data, intraday_data, model, feature_columns = main()
    print("Daily data head:\n", daily_data.head())
    print("Intra-day data head:\n", intraday_data.head())
