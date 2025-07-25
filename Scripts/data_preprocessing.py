import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_rsi(data, periods=14):
    """
    Calculate Relative Strength Index (RSI) for a given series.
    
    Args:
        data (pd.Series): Price series (e.g., Close prices).
        periods (int): Lookback period for RSI calculation.
    
    Returns:
        pd.Series: RSI values.
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def preprocess_data(input_path='data/intraday/^GSPC_intraday_5min_60days.csv', sequence_length=60, random_state=42):
    """
    Preprocess 5-minute S&P 500 data for LSTM and XGBoost models.
    
    Args:
        input_path (str): Path to input CSV file.
        sequence_length (int): Number of timesteps for sequences.
        random_state (int): Seed for reproducibility.
    
    Returns:
        tuple: (X_lstm, X_xgboost, y, close_scaler, feature_columns)
            - X_lstm: 3D array for LSTM [samples, sequence_length, features]
            - X_xgboost: 2D array for XGBoost [samples, sequence_length * features]
            - y: Target array (next minute's scaled Close price)
            - close_scaler: MinMaxScaler for Close column
            - feature_columns: List of feature columns
    """
    try:
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        
        # Verify required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Clean data
        df = df[required_columns].copy()
        df.dropna(inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        if df.isna().any().any():
            logger.warning("Non-numeric values found. Dropping invalid rows.")
            df.dropna(inplace=True)
        logger.info(f"Cleaned data shape: {df.shape}")
        
        # Feature engineering
        df['RSI'] = calculate_rsi(df['Close'], periods=14)
        df['MA10'] = df['Close'].rolling(window=2).mean()  # 10-minute MA (2 * 5min)
        df['MA60'] = df['Close'].rolling(window=12).mean()  # 60-minute MA (12 * 5min)
        df.dropna(inplace=True)  # Drop rows with NaN from rolling calculations
        logger.info(f"Data shape after feature engineering: {df.shape}")
        
        # Define feature columns
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'RSI', 'MA10', 'MA60']
        df = df[feature_columns].copy()
        
        # Scale features
        np.random.seed(random_state)  # Ensure reproducibility
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_columns])
        
        # Separate scaler for Close to inverse-transform predictions
        close_scaler = MinMaxScaler()
        close_scaler.fit(df[['Close']])
        
        # Create sequences
        X_lstm, X_xgboost, y = [], [], []
        for i in range(sequence_length, len(scaled_data)):
            X_lstm.append(scaled_data[i - sequence_length:i])  # 3D for LSTM
            X_xgboost.append(scaled_data[i - sequence_length:i].flatten())  # 2D for XGBoost
            y.append(scaled_data[i, feature_columns.index('Close')])  # Next Close price
        
        X_lstm = np.array(X_lstm)
        X_xgboost = np.array(X_xgboost)
        y = np.array(y)
        
        logger.info(f"Created {len(X_lstm)} sequences")
        logger.info(f"X_lstm shape: {X_lstm.shape}")
        logger.info(f"X_xgboost shape: {X_xgboost.shape}")
        logger.info(f"y shape: {y.shape}")
        
        # Save preprocessed data
        os.makedirs('data/preprocessed', exist_ok=True)
        np.save('data/preprocessed/X_lstm.npy', X_lstm)
        np.save('data/preprocessed/X_xgboost.npy', X_xgboost)
        np.save('data/preprocessed/y.npy', y)
        import joblib
        joblib.dump(close_scaler, 'data/preprocessed/close_scaler.joblib')
        joblib.dump(feature_columns, 'data/preprocessed/feature_columns.joblib')
        logger.info("Preprocessed data saved to data/preprocessed/")
        
        return X_lstm, X_xgboost, y, close_scaler, feature_columns
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

if __name__ == "__main__":
    # Run preprocessing
    X_lstm, X_xgboost, y, close_scaler, feature_columns = preprocess_data()
    print("X_lstm shape:", X_lstm.shape)
    print("X_xgboost shape:", X_xgboost.shape)
    print("y shape:", y.shape)
    print("Feature columns:", feature_columns)
