import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import keras_tuner as kt
import logging
import os
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data(df, target='Close', lookback=60):
    """
    Prepare data for training by creating sequences for LSTM and scaling features.
    
    Args:
        df (pd.DataFrame): Data with features from feature_engineering.py.
        target (str): Target column for prediction.
        lookback (int): Number of time steps for LSTM sequences.
    
    Returns:
        tuple: Scaled X, y, and scaler objects.
    """
    try:
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 
                    'MACD_Signal', 'BB_High', 'BB_Low', 'VWAP', 'Sentiment']
        X = df[features].values
        y = df[target].shift(-1).dropna().values  # Predict next minute's Close
        X = X[:-1]  # Align with shifted y
        
        # Scale features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # Create sequences for LSTM
        X_seq = []
        y_seq = []
        for i in range(lookback, len(X_scaled)):
            X_seq.append(X_scaled[i-lookback:i])
            y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        return X_scaled, y, X_seq, y_seq, scaler_X
    
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def train_xgboost(X, y):
    """
    Train XGBoost model with hyperparameter tuning using GridSearchCV.
    
    Args:
        X (np.array): Scaled features.
        y (np.array): Target values.
    
    Returns:
        XGBRegressor: Trained model.
    """
    try:
        logger.info("Training XGBoost model")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        }
        model = XGBRegressor(objective='reg:squarederror')
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)
        
        logger.info(f"Best XGBoost params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    except Exception as e:
        logger.error(f"Error training XGBoost: {str(e)}")
        raise

def build_lstm_model(hp, input_shape):
    """
    Build LSTM model for Keras Tuner.
    
    Args:
        hp (kt.HyperParameters): Hyperparameters.
        input_shape (tuple): Input shape for LSTM.
    
    Returns:
        tf.keras.Model: LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32),
                   input_shape=input_shape, return_sequences=True))
    model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32)))
    model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(X_seq, y_seq):
    """
    Train LSTM model with hyperparameter tuning using Keras Tuner.
    
    Args:
        X_seq (np.array): Sequential features for LSTM.
        y_seq (np.array): Target values.
    
    Returns:
        tf.keras.Model: Trained LSTM model.
    """
    try:
        logger.info("Training LSTM model")
        tuner = kt.Hyperband(
            lambda hp: build_lstm_model(hp, (X_seq.shape[1], X_seq.shape[2])),
            objective='val_loss',
            max_epochs=10,
            directory='kt_dir',
            project_name='lstm_tuning'
        )
        
        X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
        tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
        
        best_model = tuner.get_best_models(num_models=1)[0]
        logger.info("Best LSTM model retrieved")
        return best_model
    
    except Exception as e:
        logger.error(f"Error training LSTM: {str(e)}")
        raise

def main():
    """
    Main function to load features, train models, and save them.
    """
    try:
        # Load feature data
        input_path = 'data/features/sp500_intraday_features.csv'
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} not found.")
        
        df = pd.read_csv(input_path, index_col='Timestamp', parse_dates=True)
        logger.info(f"Loaded features with shape: {df.shape}")
        
        # Prepare data
        X, y, X_seq, y_seq, scaler_X = prepare_data(df)
        
        # Train models
        xgb_model = train_xgboost(X, y)
        lstm_model = train_lstm(X_seq, y_seq)
        
        # Save models
        os.makedirs('models', exist_ok=True)
        xgb_model.save_model('models/xgboost_model.json')
        lstm_model.save('models/lstm_model.h5')
        logger.info("Models saved to models/")
        
        return xgb_model, lstm_model
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    xgb_model, lstm_model = main()
    print("XGBoost and LSTM models trained successfully.")
