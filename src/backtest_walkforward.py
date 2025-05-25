import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
import torch
import torch.nn as nn
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# Config
TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/val.csv"
TEST_FILE = "data/test.csv"
timesteps = 10
cost_per_trade = 0.0001
slippage = 0.00005

# --- LSTM model definition (must match train_model.py) ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- Risk metrics ---
def calculate_sharpe(returns):
    if np.std(returns) == 0:
        return 0.0
    return np.sqrt(252*78) * (np.mean(returns) / np.std(returns))

def calculate_sortino(returns):
    downside = returns[returns < 0]
    if np.std(downside) == 0:
        return 0.0
    return np.sqrt(252*78) * (np.mean(returns) / np.std(downside))

def max_drawdown(equity):
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return dd.min()

def calmar_ratio(returns):
    equity = (1 + returns).cumprod()
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    annual_return = (equity.iloc[-1] ** (252/len(equity))) - 1
    return annual_return / mdd

def win_rate(returns):
    return (returns > 0).sum() / len(returns)

def profit_factor(returns):
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    return gross_profit / gross_loss if gross_loss != 0 else np.inf

def turnover(preds):
    return np.sum(preds[1:] != preds[:-1]) / len(preds)

def exposure(preds):
    return np.mean(preds)

def equity_curve(returns):
    return (1 + returns).cumprod()

# --- Sequence creation for LSTM ---
def create_sequences(X, timesteps):
    X_seq = []
    for i in range(len(X) - timesteps + 1):
        X_seq.append(X[i:i+timesteps])
    return np.array(X_seq)

# --- Walk-forward backtest ---
def walk_forward_backtest(df, model_func=None, is_sequence=False, model_name=""):
    step_size = int(len(df) * 0.15)
    start = 0
    metrics = []
    all_returns = []
    all_preds = []

    while start + step_size < len(df):
        train_df = df.iloc[:start+step_size]
        test_df = df.iloc[start+step_size:start+2*step_size]

        if len(test_df) == 0:
            break

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df.drop(columns=["target", "returns"]))
        y_train = train_df["target"].values
        X_test = scaler.transform(test_df.drop(columns=["target", "returns"]))
        y_test = test_df["target"].values

        # Predictions
        if model_func is None:
            preds = np.where(test_df["returns"].shift(1) > 0, 1, 0)
        else:
            preds = model_func(X_train, y_train, X_test, is_sequence=is_sequence)

        # Sequence alignment for LSTM
        if is_sequence:
            test_df_aligned = test_df.iloc[timesteps-1:]
            preds = preds[:len(test_df_aligned)]
        else:
            test_df_aligned = test_df

        # Store predictions for DM test
        all_preds.extend(preds)

        # Strategy returns after costs
        strat_rets = np.where(preds == 1, test_df_aligned["returns"], -test_df_aligned["returns"])
        trade_changes = np.sum(preds[1:] != preds[:-1])
        strat_rets -= (trade_changes * (cost_per_trade + slippage)) / len(strat_rets)

        all_returns.extend(strat_rets)
        # Save dates for equity plotting
        if not hasattr(walk_forward_backtest, "dates_storage"):
            walk_forward_backtest.dates_storage = []
        aligned_dates = test_df_aligned.index[:len(strat_rets)]
        walk_forward_backtest.dates_storage.extend(aligned_dates)

        # Save returns for Sharpe ratio comparison
        if model_name:
            if not hasattr(walk_forward_backtest, "returns_storage"):
                walk_forward_backtest.returns_storage = {}
            if model_name not in walk_forward_backtest.returns_storage:
                walk_forward_backtest.returns_storage[model_name] = []
            walk_forward_backtest.returns_storage[model_name].extend(strat_rets)


        # Risk metrics
        eq = equity_curve(pd.Series(strat_rets))
        metrics.append([
            calculate_sharpe(strat_rets),
            calculate_sortino(strat_rets),
            calmar_ratio(pd.Series(strat_rets)),
            max_drawdown(eq),
            win_rate(strat_rets),
            profit_factor(strat_rets),
            turnover(preds),
            exposure(preds)
        ])

        start += step_size

    metrics_df = pd.DataFrame(metrics, columns=[
        "Sharpe", "Sortino", "Calmar", "MaxDD",
        "HitRate", "ProfitFactor", "Turnover", "Exposure"
    ])

    # Save predictions to CSV
    if model_name:
        pred_df = pd.DataFrame({"pred": all_preds})
        pred_df.to_csv(f"data/walkforward_preds_{model_name}.csv", index=False)
        print(f"[DONE] Saved walk-forward predictions for {model_name} to data/walkforward_preds_{model_name}.csv")

    return np.array(all_returns), metrics_df.mean()

# --- Real model loaders ---
def lstm_model(X_train, y_train, X_test, is_sequence=True):
    input_size = X_train.shape[1]
    model = LSTMClassifier(input_size, hidden_size=64, num_layers=1, dropout=0.0)
    model.load_state_dict(torch.load("data/best_lstm_model.pth"))
    model.eval()

    # Create sequences
    X_train_seq = torch.tensor(create_sequences(X_train, timesteps), dtype=torch.float32)
    X_test_seq = torch.tensor(create_sequences(X_test, timesteps), dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_test_seq)
        probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int).flatten()
    return preds

def xgb_model(X_train, y_train, X_test, is_sequence=False):
    model = XGBClassifier()
    model.load_model("data/xgb_model.json")
    preds = model.predict(X_test)
    return preds

# --- Main ---
if __name__ == "__main__":
    # Load full dataset with features
    train_df = pd.read_csv(TRAIN_FILE, index_col=0, parse_dates=True)
    val_df = pd.read_csv(VAL_FILE, index_col=0, parse_dates=True)
    test_df = pd.read_csv(TEST_FILE, index_col=0, parse_dates=True)
    df = pd.concat([train_df, val_df, test_df]).sort_index()

    print("[INFO] Running LSTM walk-forward...")
    lstm_rets, lstm_metrics = walk_forward_backtest(df, lstm_model, is_sequence=True, model_name="lstm")

    print("[INFO] Running XGBoost walk-forward...")
    xgb_rets, xgb_metrics = walk_forward_backtest(df, xgb_model, model_name="xgb")

    print("[INFO] Running Last Price baseline...")
    lp_rets, lp_metrics = walk_forward_backtest(df, model_func=None, model_name="lastprice")

    print("[INFO] Running Buy & Hold...")
    bh_rets = df["returns"].values
    bh_metrics = [
        calculate_sharpe(bh_rets),
        calculate_sortino(bh_rets),
        calmar_ratio(pd.Series(bh_rets)),
        max_drawdown(equity_curve(pd.Series(bh_rets))),
        win_rate(bh_rets),
        profit_factor(bh_rets),
        np.nan,
        np.nan
    ]

    summary = pd.DataFrame([
        ["LSTM"] + list(lstm_metrics),
        ["XGBoost"] + list(xgb_metrics),
        ["LastPrice"] + list(lp_metrics),
        ["BuyHold"] + bh_metrics
    ], columns=["Strategy", "Sharpe", "Sortino", "Calmar", "MaxDD", "HitRate", "ProfitFactor", "Turnover", "Exposure"])

    summary.to_csv("data/walkforward_metrics.csv", index=False)
    print(summary)

    dates = pd.to_datetime(walk_forward_backtest.dates_storage)
    min_len = min(len(dates), len(lstm_rets), len(xgb_rets), len(lp_rets), len(bh_rets))
    dates = dates[:min_len]
    plt.figure(figsize=(12,6))
    plt.plot(dates, equity_curve(pd.Series(lstm_rets[:min_len], index=dates)), label="LSTM")
    plt.plot(dates, equity_curve(pd.Series(xgb_rets[:min_len], index=dates)), label="XGBoost")
    plt.plot(dates, equity_curve(pd.Series(lp_rets[:min_len], index=dates)), label="Last Price")
    plt.plot(dates, equity_curve(pd.Series(bh_rets[:min_len], index=dates)), label="Buy & Hold", linestyle="--")
    plt.title("Walk-Forward Equity Curves (After Costs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/walkforward_equity.png")
    print("[DONE] Walk-forward equity plot saved to data/walkforward_equity.png")

    # Save returns CSVs for Sharpe tests
    if hasattr(walk_forward_backtest, "returns_storage"):
        for model_name, rets in walk_forward_backtest.returns_storage.items():
            pd.DataFrame({"returns": rets}).to_csv(f"data/walkforward_returns_{model_name}.csv", index=False)
            print(f"[DONE] Saved walk-forward returns for {model_name} to data/walkforward_returns_{model_name}.csv")

