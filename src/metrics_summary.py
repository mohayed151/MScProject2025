import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from scipy.stats import ttest_rel

TEST_FILE = "data/test.csv"
LSTM_PRED_FILE = "data/lstm_predictions.csv"
XGB_PRED_FILE = "data/xgboost_predictions.csv"

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    return np.sqrt(252*78) * (np.mean(excess_returns) / np.std(excess_returns))  # 5-min bars

if __name__ == "__main__":
    # Load data
    test_df = pd.read_csv(TEST_FILE, index_col=0, parse_dates=True)
    lstm_preds = pd.read_csv(LSTM_PRED_FILE)
    xgb_preds = pd.read_csv(XGB_PRED_FILE)

    # Compute returns
    lstm_returns = np.where(lstm_preds["pred"] == 1, test_df["returns"], -test_df["returns"])
    xgb_returns = np.where(xgb_preds["pred"] == 1, test_df["returns"], -test_df["returns"])
    bh_returns = test_df["returns"]

    # Metrics storage
    results = []

    # LSTM
    acc_lstm = accuracy_score(test_df["target"], lstm_preds["pred"])
    prec_lstm = precision_score(test_df["target"], lstm_preds["pred"])
    sharpe_lstm = calculate_sharpe_ratio(lstm_returns)
    t_stat, p_val = ttest_rel(lstm_returns, bh_returns)
    results.append(["LSTM", acc_lstm, prec_lstm, sharpe_lstm, p_val])

    # XGBoost
    acc_xgb = accuracy_score(test_df["target"], xgb_preds["pred"])
    prec_xgb = precision_score(test_df["target"], xgb_preds["pred"])
    sharpe_xgb = calculate_sharpe_ratio(xgb_returns)
    t_stat, p_val = ttest_rel(xgb_returns, bh_returns)
    results.append(["XGBoost", acc_xgb, prec_xgb, sharpe_xgb, p_val])

    # Buy & Hold
    sharpe_bh = calculate_sharpe_ratio(bh_returns)
    results.append(["Buy & Hold", None, None, sharpe_bh, None])

    # Create DataFrame
    metrics_df = pd.DataFrame(results, columns=["Strategy", "Accuracy", "Precision", "Sharpe", "p-value vs B&H"])

    # Save results
    metrics_df.to_csv("data/metrics_summary.csv", index=False)
    metrics_df.to_markdown("data/metrics_summary.md", index=False)

    print("[DONE] Metrics summary saved to:")
    print(" - data/metrics_summary.csv")
    print(" - data/metrics_summary.md")
    print(metrics_df)
