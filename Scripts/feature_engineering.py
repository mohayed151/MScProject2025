import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
import logging
import os
import joblib

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

def feature_engineering(input_path='data/intraday/^GSPC_intraday_5min_60days.csv', random_state=42):
    """
    Perform feature engineering on 5-minute S&P 500 data, including RSI and moving averages,
    and compute permutation importance to justify feature selection.
    
    Args:
        input_path (str): Path to input CSV file.
        random_state (int): Seed for reproducibility.
    
    Returns:
        tuple: (df_enhanced, scaler, close_scaler, feature_columns, perm_importance)
            - df_enhanced: DataFrame with original and new features.
            - scaler: MinMaxScaler for all features.
            - close_scaler: MinMaxScaler for Close column.
            - feature_columns: List of feature columns.
            - perm_importance: Dictionary of permutation importance scores.
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
        logger.info("Computing technical indicators")
        df['RSI'] = calculate_rsi(df['Close'], periods=14)  # 14-period RSI
        df['MA10'] = df['Close'].rolling(window=2).mean()  # 10-minute MA (2 * 5min)
        df['MA60'] = df['Close'].rolling(window=12).mean()  # 60-minute MA (12 * 5min)
        df.dropna(inplace=True)  # Drop rows with NaN from rolling calculations
        logger.info(f"Data shape after feature engineering: {df.shape}")
        
        # Define feature columns
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'RSI', 'MA10', 'MA60']
        df_enhanced = df[feature_columns].copy()
        
        # Scale features (fit scaler for later use, but store unscaled data for now)
        np.random.seed(random_state)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_enhanced[feature_columns])
        
        # Separate scaler for Close to inverse-transform predictions
        close_scaler = MinMaxScaler()
        close_scaler.fit(df_enhanced[['Close']])
        
        # Permutation importance analysis to justify feature selection
        logger.info("Computing permutation importance")
        X = scaled_data
        y = df_enhanced['Close'].shift(-1).dropna().values  # Next Close price as target
        X = X[:-1]  # Align X with y
        model = LinearRegression()
        model.fit(X, y)
        perm_result = permutation_importance(model, X, y, n_repeats=10, random_state=random_state)
        perm_importance = {feature_columns[i]: perm_result.importances_mean[i] for i in range(len(feature_columns))}
        logger.info(f"Permutation importance: {perm_importance}")
        
        # Save enhanced data and metadata
        os.makedirs('data/enhanced', exist_ok=True)
        output_path = 'data/enhanced/^GSPC_enhanced_5min_60days.csv'
        df_enhanced.to_csv(output_path)
        joblib.dump(scaler, 'data/enhanced/scaler.joblib')
        joblib.dump(close_scaler, 'data/enhanced/close_scaler.joblib')
        joblib.dump(feature_columns, 'data/enhanced/feature_columns.joblib')
        joblib.dump(perm_importance, 'data/enhanced/perm_importance.joblib')
        logger.info(f"Enhanced data saved to {output_path}")
        
        return df_enhanced, scaler, close_scaler, feature_columns, perm_importance
    
    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        raise

if __name__ == "__main__":
    # Run feature engineering
    df_enhanced, scaler, close_scaler, feature_columns, perm_importance = feature_engineering()
    print("Enhanced DataFrame head:\n", df_enhanced.head())
    print("Feature columns:", feature_columns)
    print("Permutation importance:", perm_importance)
