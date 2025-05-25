import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
TEST_FILE = "data/test.csv"
LSTM_PRED_FILE = "data/lstm_predictions.csv"
XGB_PRED_FILE = "data/xgboost_predictions.csv"

def equity_curve(returns):
    return (1 + returns).cumprod()

if __name__ == "__main__":
    # Load test data
    test_df = pd.read_csv(TEST_FILE, index_col=0, parse_dates=True)

    # Load predictions
    lstm_preds = pd.read_csv(LSTM_PRED_FILE)
    xgb_preds = pd.read_csv(XGB_PRED_FILE)

    # Sequence length used in LSTM
    timesteps = 10

    # Align LSTM predictions with shifted test data
    test_df_lstm = test_df.iloc[timesteps-1:]  # drop first timesteps-1 rows
    lstm_returns = np.where(lstm_preds["pred"] == 1, test_df_lstm["returns"], -test_df_lstm["returns"])

    # XGBoost predictions align with full test_df
    xgb_returns = np.where(xgb_preds["pred"] == 1, test_df["returns"], -test_df["returns"])

    # Buy & Hold returns
    bh_returns = test_df["returns"]

    # Compute equity curves
    eq_lstm = equity_curve(lstm_returns)
    eq_xgb = equity_curve(xgb_returns)
    eq_bh = equity_curve(bh_returns)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(test_df_lstm.index, eq_lstm, label="LSTM Strategy", color="blue")
    plt.plot(test_df.index, eq_xgb, label="XGBoost Strategy", color="green")
    plt.plot(test_df.index, eq_bh, label="Buy & Hold", color="orange", linestyle="--")
    plt.title("Equity Curve Comparison - LSTM vs XGBoost vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/equity_curve_all.png")
    print("[DONE] Combined equity curve saved to data/equity_curve_all.png")
