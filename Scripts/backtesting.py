import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.metrics import accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backtest_model(data_path='data/processed/sp500_train_data.csv', model_path='models/rf_model.joblib', 
                  feature_columns_path='models/feature_columns.txt'):
    """
    Perform walk-forward backtesting on the model using saved data and model.
    
    Args:
        data_path (str): Path to the processed data CSV.
        model_path (str): Path to the trained model.
        feature_columns_path (str): Path to the feature columns text file.
    
    Returns:
        pd.DataFrame: Backtest results with predictions, target, and returns.
    """
    try:
        logger.info("Starting backtesting")
        
        # Load data
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(f"Loaded data columns: {list(df.columns)}")
        
        # Load feature columns
        with open(feature_columns_path, 'r') as f:
            feature_columns = f.read().split(',')
        logger.info(f"Loaded feature columns: {feature_columns}")
        
        # Load model
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Ensure required columns
        required_columns = feature_columns + ['Close', 'Returns', 'Target']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing columns for backtesting: {missing_cols}")
            raise ValueError(f"Missing columns for backtesting: {missing_cols}")
        
        # Initialize backtest columns
        df['Prediction'] = 0
        df['Strategy_Return'] = 0.0
        
        # Walk-forward backtesting
        train_size = int(len(df) * 0.6)  # Initial training size
        step_size = 20  # Number of days to step forward
        for i in range(train_size, len(df) - 1, step_size):
            train_df = df.iloc[:i]
            test_df = df.iloc[i:i + step_size]
            
            if len(test_df) == 0:
                continue
            
            # Prepare training data
            X_train = train_df[feature_columns]
            y_train = train_df['Target']
            
            # Retrain model
            model.fit(X_train, y_train)
            
            # Prepare test data
            X_test = test_df[feature_columns]
            predictions = model.predict(X_test)
            
            # Store predictions
            df.loc[test_df.index, 'Prediction'] = predictions
            
            # Calculate strategy returns (buy if predict up, hold if down)
            df.loc[test_df.index, 'Strategy_Return'] = np.where(
                predictions == 1, test_df['Returns'], 0.0
            )
        
        # Calculate cumulative returns
        df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return'] / 100).cumprod() - 1
        df['Cumulative_Market_Return'] = (1 + df['Returns'] / 100).cumprod() - 1
        
        # Calculate performance metrics
        total_trades = (df['Prediction'] == 1).sum()
        accuracy = accuracy_score(df['Target'].iloc[train_size:], df['Prediction'].iloc[train_size:])
        annualized_return = df['Strategy_Return'].mean() * 252  # Assuming 252 trading days
        sharpe_ratio = (df['Strategy_Return'].mean() * 252) / (df['Strategy_Return'].std() * np.sqrt(252)) if df['Strategy_Return'].std() != 0 else 0
        
        logger.info(f"Backtest results: Total trades = {total_trades}, Accuracy = {accuracy:.4f}")
        logger.info(f"Annualized return: {annualized_return:.4f}%")
        logger.info(f"Sharpe ratio: {sharpe_ratio:.4f}")
        
        # Save backtest results
        os.makedirs('data/backtest', exist_ok=True)
        output_path = 'data/backtest/sp500_backtest_results.csv'
        df.to_csv(output_path)
        logger.info(f"Backtest results saved to {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        raise

def main():
    """Main function to run backtesting."""
    try:
        backtest_results = backtest_model()
        print("Backtest results head:\n", backtest_results[['Close', 'Target', 'Prediction', 'Strategy_Return', 'Cumulative_Strategy_Return']].head())
        return backtest_results
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    backtest_results = main()
