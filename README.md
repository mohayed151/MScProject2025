# MSc Project 2025 — Intraday Algorithmic Trading with Machine Learning

This project explores intraday trading strategies on the SPY ETF using machine learning models (LSTM, XGBoost) and statistical evaluation techniques.  
It includes:
- Data acquisition (Polygon.io)
- Feature engineering with technical indicators
- Baseline and ML model training
- Walk-forward backtesting with costs
- Statistical significance testing (Diebold–Mariano, Sharpe CIs)

## Structure


🏗️ Project Structure
MSCProject2025/
├── config.py                  # Configuration and parameters
├── requirements.txt           # Python dependencies
├── main.py                    # Main pipeline orchestrator
├── src/                   # Directory for modular Python scripts
│   ├── data_preprocessing.py      # Data download and management
│   ├── feature_engineering.py # Feature creation and processing
│   ├── model_training.py      # ML model training and evaluation
│   ├── trading_strategy.py    # Trading strategy implementation
│   └── backtesting.py         # Backtesting and performance analysis
├── README.md                  # This file
├── Data/                      # Data directory
│   ├── raw/                   # Raw OHLCV price data
│   └── features/              # Feature-engineered data
├── models/                    # Trained models and scalers
├── results/                   # Backtest results, logs, and visualizations
│   ├── data/                  # CSVs of equity curves, trade logs
│   └── plots/                 # PNG plots of performance
└── .gitignore                 # Git ignore file

🚀 Features
Data Management
Automatic stock price data download using Yahoo Finance (OHLCV data)

Data validation and preprocessing

Support for data updates and incremental downloads

Configurable ticker symbols and time periods

Feature Engineering
Technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands)

Price momentum features

Volatility indicators

Lagged return features

Rolling statistical measures

High/Low period features for dynamic Stop Loss/Take Profit

Machine Learning Models
Ridge Regression with cross-validation

Random Forest Regression

Time series cross-validation

Feature scaling and preprocessing

Model performance evaluation

Trading Strategy
Position sizing based on predicted returns

Transaction cost consideration

Risk management with position limits

Dynamic Stop Loss and Take Profit levels based on historical price action or fixed percentage

Compatible with the original getMyPosition interface logic

Backtesting
Comprehensive performance metrics (Total Return, Annualized Return, Sharpe, Max Drawdown, etc.)

Transaction cost simulation

Slippage modeling

Drawdown analysis

Visualization of equity curve, daily returns, and costs

📋 Requirements
Python 3.8+

See requirements.txt for complete list of dependencies

🔧 Installation
Clone the repository:

git clone <your-repo-url>
cd trading_strategy

Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

🎯 Usage
The main.py script orchestrates the entire pipeline.

Quick Start (Full Pipeline)
Run the complete pipeline with default settings. It will download data, engineer features, train models, and run the backtest. Subsequent runs will use cached data/models unless forced.

python main.py --step full

Command Line Options
You can control the pipeline steps and behavior using command-line arguments with main.py:

Run specific pipeline components:

# Download data only
python main.py --step download

# Create features only
python main.py --step features

# Train models only
python main.py --step training

# Run backtesting only
python main.py --step backtest

Force re-execution of a step (e.g., force re-download):

python main.py --step download --force-download
python main.py --step features --force-features
python main.py --step training --force-training
# For full pipeline with forces:
python main.py --step full --force-download --force-features --force-training

Skip backtesting (when running full pipeline):

python main.py --step full --no-backtest

Check pipeline status (shows existing files and their sizes):

python main.py --status

Enable verbose logging (shows DEBUG level messages):

python main.py --verbose

⚙️ Configuration
Modify config.py to customize various parameters:

Trading Parameters: NUM_INSTRUMENTS, COMM_RATE, MAX_POSITION, INITIAL_CAPITAL, SLIPPAGE, MIN_RETURN_THRESHOLD, STOP_LOSS_PCT, PREDICTION_THRESHOLD.

Data Parameters: TICKER_SYMBOLS, PERIOD, INTERVAL.

Feature Parameters: ROLLING_WINDOWS, PRICE_CHANGE_LAGS, SMA_PERIOD, LOOKBACK_PERIOD, MIN_DATA_DAYS.

Model Parameters: RANDOM_STATE, RIDGE_CV_FOLDS, RF_ESTIMATORS.

Output Parameters: SAVE_PLOTS.

Example config.py variables:

# Trading
NUM_INSTRUMENTS = 1 # Currently set for single instrument, can be expanded
COMM_RATE = 0.0005
MAX_POSITION = 10000
INITIAL_CAPITAL = 100000.0
SLIPPAGE = 0.0001
STOP_LOSS_PCT = 0.015 # Used for fixed percentage SL/TP fallback

# Data
TICKER_SYMBOLS = ["^GSPC"] # Example: S&P 500 index
PERIOD = "60d"
INTERVAL = "5m"

# Features
ROLLING_WINDOWS = [3, 5, 10, 15, 20, 50]
PRICE_CHANGE_LAGS = [1, 2, 3]
SMA_PERIOD = 14
LOOKBACK_PERIOD = 10 # For High/Low period features

# Model Training
RF_ESTIMATORS = 100

# Results
SAVE_PLOTS = True

📊 Output and Results
The pipeline generates several outputs in the Data, models, and results directories:

Data Files
Data/raw/combined_ohlcv_prices_{INTERVAL}.csv: Raw OHLCV price data for all tickers.

Data/features/combined_features_{INTERVAL}_ml_strategy.csv: Combined feature-engineered data for all tickers.

models/ml_feature_columns.pkl: List of ML feature column names.

models/strategy_base_columns.pkl: List of all columns used in the strategy.

Models
models/{TICKER}_{MODEL_TYPE}_model.joblib: Trained machine learning models (e.g., GSPC_ridge_model.joblib).

models/{TICKER}_{MODEL_TYPE}_scaler.joblib: Scalers used for feature preprocessing.

models/performance_metrics.joblib: Dictionary of model performance metrics.

Results
results/trading_strategy.log: Detailed log of pipeline execution.

results/data/equity_curve_overall_{INTERVAL}.csv: CSV of the overall portfolio equity curve.

results/data/trade_log_overall_{INTERVAL}.csv: CSV detailing all executed trades.

results/model_performance_summary.csv: Summary of trained model performance.

results/plots/ml_ensemble_strategy_equity_curve_overall_{INTERVAL}.png: Plot of the overall equity curve.

results/plots/ml_ensemble_strategy_drawdown_overall_{INTERVAL}.png: Plot of the overall drawdown.

results/plots/backtest_results_{INTERVAL}.png: Comprehensive backtest results plots (portfolio value, returns, costs).