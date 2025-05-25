"""
Configuration and parameters for the intraday trading strategy
"""
import os
from pathlib import Path
from datetime import datetime, date

# --- Directory Setup ---
# Get the base directory of the project (one level up from config.py)
BASE_DIR = Path(__file__).resolve().parent

# Define paths for data, models, and results
RAW_DATA_DIR = BASE_DIR / 'Data' / 'raw'
FEATURES_DATA_DIR = BASE_DIR / 'Data' / 'features'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'data').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'plots').mkdir(parents=True, exist_ok=True)

# --- Data Source Configuration ---
DATA_SOURCE = "mt5"  # Changed to MT5 as requested

# --- Trading Parameters ---
NUM_INSTRUMENTS = 1  # Currently set for single instrument, can be expanded
COMM_RATE = 0.0005  # Commission rate per trade (e.g., 0.05% of trade value)
MAX_POSITION = 10000  # Maximum absolute position value in USD
INITIAL_CAPITAL = 100000.0  # Starting capital for backtesting
SLIPPAGE = 0.0001  # Slippage as a percentage of trade value (e.g., 0.01%)

# Strategy thresholds
MIN_RETURN_THRESHOLD = 0.0001  # Minimum predicted return to consider a trade
PREDICTION_THRESHOLD = 0.0005  # Threshold for prediction to open a position (e.g., 0.05% expected return)
STOP_LOSS_PCT = 0.015  # Fixed percentage for Stop Loss (e.g., 1.5%) - used as fallback or initial setting

# --- Data Parameters ---
# Ticker symbols
TICKER_SYMBOLS = ["^GSPC"]  # S&P 500 index for yfinance

# Date range configuration (5 years ending July 1st, 2025)
END_DATE = date(2025, 7, 1)  # July 1st, 2025
START_DATE = date(2020, 7, 1)  # July 1st, 2020 (5 years before)

# For backward compatibility with existing code
PERIOD = "5y"  # This will be overridden by START_DATE and END_DATE
INTERVAL = "5m"  # Data interval

# --- MT5 Configuration (SECURE VERSION) ---
# Load MT5 credentials from environment variables for security
# Set these in your environment:
# export MT5_LOGIN=30439343
# export MT5_PASSWORD=your_actual_password
# export MT5_SERVER=Deriv-Demo

MT5_LOGIN = os.getenv('MT5_LOGIN')
MT5_PASSWORD = os.getenv('MT5_PASSWORD') 
MT5_SERVER = os.getenv('MT5_SERVER', 'Deriv-Demo')  # Default server
MT5_PATH = os.getenv('MT5_PATH')  # Optional path to MT5 terminal

# Validate MT5 credentials are provided
if DATA_SOURCE == "mt5":
    if not MT5_LOGIN or not MT5_PASSWORD:
        print("⚠️  WARNING: MT5 credentials not found in environment variables!")
        print("Please set the following environment variables:")
        print("  export MT5_LOGIN=your_account_number")
        print("  export MT5_PASSWORD=your_password")
        print("  export MT5_SERVER=your_server (optional, defaults to Deriv-Demo)")
        print("\nOn Windows, use 'set' instead of 'export'")

# MT5 Symbol mapping for common tickers
MT5_SYMBOL_MAP = {
    "^GSPC": "SPX500",      # S&P 500
    "^DJI": "US30",         # Dow Jones
    "^IXIC": "NAS100",      # NASDAQ
    "EURUSD": "EURUSD",     # EUR/USD
    "GBPUSD": "GBPUSD",     # GBP/USD
    "USDJPY": "USDJPY",     # USD/JPY
    "XAUUSD": "XAUUSD",     # Gold
    "XBRUSD": "XBRUSD",     # Oil (Brent)
    "XTIUSD": "XTIUSD",     # Oil (WTI)
    # Add more mappings as needed
}

# --- Feature Engineering Parameters ---
ROLLING_WINDOWS = [3, 5, 10, 15, 20, 50]  # Windows for rolling statistics
PRICE_CHANGE_LAGS = [1, 2, 3]  # Lags for percentage change features
SMA_PERIOD = 14  # Period for Simple Moving Average (used in Log_Close_SMA)
LOOKBACK_PERIOD = 10  # Period for High/Low lookback (for dynamic SL/TP)
MIN_DATA_DAYS = 50  # Minimum number of data points required after feature engineering

# --- Model Training Parameters ---
RANDOM_STATE = 42  # For reproducibility
RIDGE_CV_FOLDS = 5  # Number of folds for RidgeCV cross-validation
RF_ESTIMATORS = 100  # Number of trees in Random Forest

# --- Output Parameters ---
SAVE_PLOTS = True  # Whether to save generated plots to the results/plots directory

# --- Validation ---
def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Validate date range
    if START_DATE >= END_DATE:
        errors.append("START_DATE must be before END_DATE")
    
    # Validate data source
    if DATA_SOURCE not in ["yfinance", "mt5"]:
        errors.append("DATA_SOURCE must be 'yfinance' or 'mt5'")
    
    # Validate MT5 credentials if using MT5
    if DATA_SOURCE == "mt5":
        if not MT5_LOGIN:
            errors.append("MT5_LOGIN environment variable must be set when using MT5")
        if not MT5_PASSWORD:
            errors.append("MT5_PASSWORD environment variable must be set when using MT5")
    
    # Validate trading parameters
    if INITIAL_CAPITAL <= 0:
        errors.append("INITIAL_CAPITAL must be positive")
    
    if COMM_RATE < 0 or COMM_RATE > 1:
        errors.append("COMM_RATE must be between 0 and 1")
    
    # Validate symbols
    if not TICKER_SYMBOLS:
        errors.append("TICKER_SYMBOLS cannot be empty")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    return True

# Run validation when config is imported
try:
    validate_config()
except ValueError as e:
    print(f"⚠️  Configuration Error: {e}")
    if DATA_SOURCE == "mt5" and (not MT5_LOGIN or not MT5_PASSWORD):
        print("\n💡 To fix MT5 credentials, run these commands:")
        print("   export MT5_LOGIN=30439343")
        print("   export MT5_PASSWORD=your_new_password")
        print("   export MT5_SERVER=Deriv-Demo")

# Print configuration summary
if __name__ == "__main__":
    print("=== Trading Strategy Configuration ===")
    print(f"Data Source: {DATA_SOURCE}")
    print(f"Symbols: {TICKER_SYMBOLS}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Interval: {INTERVAL}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Commission Rate: {COMM_RATE:.4f}")
    print(f"Data Directory: {RAW_DATA_DIR}")
    
    if DATA_SOURCE == "mt5":
        print(f"MT5 Symbols: {[MT5_SYMBOL_MAP.get(s, s) for s in TICKER_SYMBOLS]}")
        print(f"MT5 Login: {'***' + str(MT5_LOGIN)[-4:] if MT5_LOGIN else 'NOT SET'}")
        print(f"MT5 Server: {MT5_SERVER}")
        
        if not MT5_LOGIN or not MT5_PASSWORD:
            print("\n⚠️  MT5 credentials not properly configured!")