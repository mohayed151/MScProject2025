"""
Trading strategy implementation - modernized version of the original getMyPosition function
"""
import numpy as np
import pandas as pd
import math
import logging
from pathlib import Path
import joblib # For loading feature lists

# Import necessary classes and configurations
# Corrected to use relative imports for model_training and feature_engineering
try:
    from .model_training import ModelTrainer # Corrected to relative import
    from .feature_engineering import FeatureEngineer # Corrected to relative import
    from config import MODELS_DIR, TICKER_SYMBOLS, SMA_PERIOD, LOOKBACK_PERIOD, \
                       COMM_RATE, PREDICTION_THRESHOLD, MIN_RETURN_THRESHOLD, \
                       MAX_POSITION, STOP_LOSS_PCT
except ImportError as e:
    logging.error(f"Import error: {e}. Ensure config.py, model_training.py, and feature_engineering.py are correctly set up.")
    exit()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingStrategy:
    def __init__(self):
        """
        Initialize trading strategy.
        Loads models and pre-calculated feature data.
        """
        self.model_trainer = ModelTrainer()
        self.feature_engineer = FeatureEngineer()
        
        # Load models and features during initialization
        self.load_models_and_features()

        # Load ML feature columns (common across all tickers)
        try:
            self.ml_feature_columns = joblib.load(MODELS_DIR / 'ml_feature_columns.pkl')
            logger.info("Loaded ML feature columns for strategy.")
        except FileNotFoundError as e:
            logger.error(f"ml_feature_columns.pkl not found: {e}. Please run feature_engineering.py first.")
            raise

    def load_models_and_features(self):
        """
        Load trained models and ensure feature data is accessible.
        """
        logger.info("Loading models and ensuring features are accessible...")
        
        # Load trained models via ModelTrainer
        try:
            self.model_trainer.load_models()
            logger.info(f"Loaded {len(self.model_trainer.models)} models and {len(self.model_trainer.scalers)} scalers.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
            
        # Ensure feature data is loaded into FeatureEngineer's cache
        # This will either load from file or create if not present
        try:
            self.feature_engineer.create_all_features(save_features=False) 
            logger.info(f"Features data loaded/ensured for {len(self.feature_engineer.features_data)} tickers.")
        except Exception as e:
            logger.error(f"Error loading/creating features: {e}")
            raise
            
    def get_my_position(self, current_data_row: pd.Series, prev_data_row: pd.Series, ticker: str):
        """
        Calculates the desired position for a single ticker at the current time step.
        This function consumes pre-calculated features.

        Args:
            current_data_row (pd.Series): A single row of feature-engineered data for the current timestamp.
            prev_data_row (pd.Series): A single row of feature-engineered data for the previous timestamp.
            ticker (str): The ticker symbol for which to calculate the position.

        Returns:
            float: The calculated desired position (units to hold) for the instrument.
        """
        # Extract necessary data from the current and previous rows
        current_close = current_data_row['Close'] # 'Close' is the actual price
        prev_close = prev_data_row['Close']

        current_log_close = current_data_row['Log_Close']
        prev_log_close = prev_data_row['Log_Close']
        current_log_close_sma = current_data_row[f'Log_Close_SMA{SMA_PERIOD}']
        prev_log_close_sma = prev_data_row[f'Log_Close_SMA{SMA_PERIOD}']

        # High_Period and Low_Period are now available from feature_engineering.py
        current_high_period = current_data_row[f'High_{LOOKBACK_PERIOD}_Period']
        current_low_period = current_data_row[f'Low_{LOOKBACK_PERIOD}_Period']

        # Get the latest ML features for prediction
        # Ensure latest_ml_features is a DataFrame with correct columns for scaler.transform
        latest_ml_features = current_data_row[self.ml_feature_columns].to_frame().T 
        
        # Check for NaNs in ML features before predicting
        if latest_ml_features.isnull().any().any():
            logger.warning(f"ML features for {ticker} at {current_data_row.name} contain NaNs. Skipping prediction.")
            return 0.0

        # Make prediction using trained models (Ridge and Random Forest)
        predicted_return_ridge = self.model_trainer.predict_returns(ticker, model_type='ridge', latest_data=latest_ml_features)
        predicted_return_rf = self.model_trainer.predict_returns(ticker, model_type='rf', latest_data=latest_ml_features)

        if predicted_return_ridge is None or predicted_return_rf is None:
            logger.warning(f"Could not get prediction for {ticker}. Returning 0 position.")
            return 0.0

        # Combined prediction (simple average)
        combined_prediction = (predicted_return_ridge + predicted_return_rf) / 2.0

        # --- Strategy Logic ---
        desired_units = 0.0 # Default to no change in position

        # Logarithmic SMA Crossover conditions
        long_crossover = (prev_log_close < prev_log_close_sma) and \
                         (current_log_close > current_log_close_sma)
        
        short_crossover = (prev_log_close > prev_log_close_sma) and \
                          (current_log_close < current_log_close_sma)

        # Apply the same logic as original getMyPosition for position sizing
        if abs(combined_prediction) < MIN_RETURN_THRESHOLD:
            desired_units = 0
        else:
            target_return_scaling = 1000 # This 'targetReturn' is a scaling factor from your original code
            net_return_after_comm = combined_prediction - (COMM_RATE * 2.0) # Account for round-trip commission
            
            most_recent_price = current_close
            max_units_allowed = math.floor(MAX_POSITION / most_recent_price) # Max units based on MAX_POSITION value

            # Calculate raw units
            calculated_raw_units = max_units_allowed * net_return_after_comm * target_return_scaling
            
            # Decide direction and cap units
            if combined_prediction > 0 and long_crossover: # Only go long if prediction is positive AND crossover
                desired_units = min(calculated_raw_units, max_units_allowed) # Cap at max_units
            elif combined_prediction < 0 and short_crossover: # Only go short if prediction is negative AND crossover
                desired_units = max(calculated_raw_units, -max_units_allowed) # Cap at -max_units_allowed
            else: # If prediction doesn't align with crossover, or too small, stay flat
                desired_units = 0

        return desired_units
