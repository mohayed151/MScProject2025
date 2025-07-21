import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backtest_strategy(df, model, model_type='xgboost', lookback=60, threshold=0.001):
    """
    Backtest a trading strategy using ML predictions.
    
    Args:
        df (pd.DataFrame): Data with features and actual Close prices.
        model: Trained XGBoost or LSTM model.
        model_type (str): 'xgboost' or 'lstm'.
        lookback (int): Lookback period for LSTM sequences.
        threshold (float): Minimum predicted price change to trigger a trade.
    
    Returns:
        pd.DataFrame: Backtest results with metrics.
    """
    try:
        logger.info(f"Backtesting {model_type} model")
        df = df.copy()
        df['Predicted_Close'] = np.nan
        
        # Generate predictions
        if model_type == 'xgboost':
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 
                        'MACD_Signal', 'BB_High', 'BB_Low', 'VWAP', 'Sentiment']
            X = df[features].values
            df['Predicted_Close'] = model.predict(X)
        
        elif model_type == 'lstm':
            X_seq = []
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 
                        'MACD_Signal', 'BB_High', 'BB_Low', 'VWAP', 'Sentiment']
            X = df[features].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            for i in range(lookback, len(X_scaled)):
                X_seq.append(X_scaled[i-lookback:i])
            X_seq = np.array(X_seq)
            df.iloc[lookback:, df.columns.get_loc('Predicted_Close')] = model.predict(X_seq).flatten()
        
        else:
            raise ValueError("Invalid model_type. Use 'xgboost' or 'lstm'.")
        
        # Trading strategy: Buy if predicted increase > threshold, sell if decrease > threshold
        df['Signal'] = 0
        df['Price_Change'] = df['Predicted_Close'].pct_change()
        df.loc[df['Price_Change'] > threshold, 'Signal'] = 1  # Buy
        df.loc[df['Price_Change'] < -threshold, 'Signal'] = -1  # Sell
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift(1)
        df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod() - 1
        
        # Calculate metrics
        sharpe_ratio = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252 * 390)  # Annualized (390 min/day)
        max_drawdown = (df['Cumulative_Returns'].cummax() - df['Cumulative_Returns']).max()
        total_return = df['Cumulative_Returns'].iloc[-1]
        
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}, Total Return: {total_return:.2%}, Max Drawdown: {max_drawdown:.2%}")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        output_path = f'results/{model_type}_backtest.csv'
        df.to_csv(output_path)
        logger.info(f"Backtest results saved to {output_path}")
        
        return df, {'Sharpe_Ratio': sharpe_ratio, 'Total_Return': total_return, 'Max_Drawdown': max_drawdown}
    
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        raise

def main():
    """
    Main function to load data, load models, and backtest strategies.
    """
    try:
        # Load feature data
        input_path = 'data/features/sp500_intraday_features.csv'
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} not found.")
        
        df = pd.read_csv(input_path, index_col='Timestamp', parse_dates=True)
        logger.info(f"Loaded features with shape: {df.shape}")
        
        # Load models
        xgb_model = XGBRegressor()
        xgb_model.load_model('models/xgboost_model.json')
        lstm_model = load_model('models/lstm_model.h5')
        
        # Backtest both models
        xgb_results, xgb_metrics = backtest_strategy(df, xgb_model, model_type='xgboost')
        lstm_results, lstm_metrics = backtest_strategy(df, lstm_model, model_type='lstm')
        
        logger.info(f"XGBoost Metrics: {xgb_metrics}")
        logger.info(f"LSTM Metrics: {lstm_metrics}")
        
        return xgb_results, lstm_results, xgb_metrics, lstm_metrics
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    xgb_results, lstm_results, xgb_metrics, lstm_metrics = main()
    print("Backtesting completed. Metrics:", xgb_metrics, lstm_metrics)
