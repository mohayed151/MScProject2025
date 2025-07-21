# MScProject2025

# Intra-day Algorithmic Trading Strategies Using Machine Learning

## Overview
MSc Data Science project to develop ML-based intra-day trading strategies for S&P 500 stocks using 1-minute interval data.

## Setup
1. Clone the repo: `git clone [repo-link]`
2. Clone intraday module: `git clone https://github.com/marcusschiesser/intraday.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Run scripts in order: `data_preprocessing.py`, `feature_engineering.py`, `model_training.py`, `backtesting.py`

## Data
- Daily data: S&P 500 (^GSPC) from 2020-07-20 to 2025-07-20, saved in `data/raw/sp500_daily.csv`.
- Intra-day data: 1-minute intervals for last 30 days, saved in `data/intraday/^GSPC_intraday.csv`.
- Features: Technical indicators and sentiment scores in `data/features/sp500_intraday_features.csv`.

## Models
- XGBoost and LSTM models saved in `models/`.

## Results
- Backtest results in `results/` with metrics (Sharpe ratio, returns, drawdown).

## Progress
- June 2025: Data collection and preprocessing.
- July 2025: Feature engineering, model training, and backtesting.
