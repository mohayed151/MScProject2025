import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/SPY_processed.csv"
TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/val.csv"
TEST_FILE = "data/test.csv"

def create_target(df):
    """
    Binary classification target: 1 if next close > current close, else 0.
    """
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df

def create_lagged_features(df, lags=5):
    """
    Create lagged features for model input.
    """
    df = df.copy()
    for col in ["close", "rsi", "macd", "macd_signal", "macd_diff",
                "ema_10", "ema_20", "bb_bbm", "bb_bbh", "bb_bbl",
                "bb_width", "returns", "volatility"]:
        for lag in range(1, lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df.dropna(inplace=True)
    return df

def chronological_split(df, train_ratio=0.7, val_ratio=0.15):
    """
    Split time series chronologically into train, validation, and test sets.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run data_preprocessing.py first.")

    print("[INFO] Loading processed dataset...")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

    print("[INFO] Creating target variable...")
    df = create_target(df)

    print("[INFO] Creating lagged features...")
    df = create_lagged_features(df, lags=5)

    print("[INFO] Splitting into train, validation, and test sets...")
    train_df, val_df, test_df = chronological_split(df)

    # Save datasets
    train_df.to_csv(TRAIN_FILE)
    val_df.to_csv(VAL_FILE)
    test_df.to_csv(TEST_FILE)

    print(f"[DONE] Feature engineering complete.")
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows")
