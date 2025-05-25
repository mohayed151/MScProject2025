import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

# ---------------------------
# Model Definition
# ---------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out  # logits

# ---------------------------
# Utility Functions
# ---------------------------
def load_datasets():
    train_df = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv("data/val.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("data/test.csv", index_col=0, parse_dates=True)
    return train_df, val_df, test_df

def create_sequences(X, y, timesteps):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps + 1):
        X_seq.append(X[i:i+timesteps])
        y_seq.append(y[i+timesteps-1])
    return np.array(X_seq), np.array(y_seq)

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    return np.sqrt(252*78) * (np.mean(excess_returns) / np.std(excess_returns))

def plot_equity_curves(strategy_returns, bh_returns, index):
    equity_strategy = (1 + strategy_returns).cumprod()
    equity_bh = (1 + bh_returns).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(index, equity_strategy, label="LSTM Strategy", color="blue")
    plt.plot(index, equity_bh, label="Buy & Hold", color="orange", linestyle="--")
    plt.title("Equity Curve Comparison - LSTM vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/lstm_equity_curve.png")
    print("[DONE] Equity curve comparison saved to data/lstm_equity_curve.png")

def bootstrap_sharpe_diff(strat_returns, bh_returns, n_bootstrap=1000):
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(strat_returns), len(strat_returns), replace=True)
        s_sharpe = calculate_sharpe_ratio(strat_returns[idx])
        bh_sharpe = calculate_sharpe_ratio(bh_returns[idx])
        diffs.append(s_sharpe - bh_sharpe)
    return np.percentile(diffs, [2.5, 97.5])

def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(filename)
    print(f"[DONE] Confusion matrix saved to {filename}")

# ---------------------------
# Main Script
# ---------------------------
if __name__ == "__main__":
    train_df, val_df, test_df = load_datasets()

    # Features / Targets
    X_train_raw = train_df.drop(columns=["target", "returns"]).values
    y_train_raw = train_df["target"].values
    X_val_raw = val_df.drop(columns=["target", "returns"]).values
    X_test_raw = test_df.drop(columns=["target", "returns"]).values
    y_val_raw = val_df["target"].values
    y_test_raw = test_df["target"].values

    # Scale features
    scaler = StandardScaler()
    X_train_raw = scaler.fit_transform(X_train_raw)
    X_val_raw = scaler.transform(X_val_raw)
    X_test_raw = scaler.transform(X_test_raw)

    # Sequence parameters
    timesteps = 10
    X_train_seq, y_train_seq = create_sequences(X_train_raw, y_train_raw, timesteps)
    X_val_seq, y_val_seq = create_sequences(X_val_raw, y_val_raw, timesteps)
    X_test_seq, y_test_seq = create_sequences(X_test_raw, y_test_raw, timesteps)

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_seq, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_seq, dtype=torch.float32).view(-1, 1)

    # Class weights
    pos_weight = (len(y_train_seq) - np.sum(y_train_seq)) / np.sum(y_train_seq)
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)

    # Model parameters
    model = LSTMClassifier(input_size=X_train_t.shape[2], hidden_size=64, num_layers=1, dropout=0.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    best_val_loss = np.inf
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        print(f"Epoch [{epoch+1}/50], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), "data/best_lstm_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("[INFO] Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(torch.load("data/best_lstm_model.pth"))

    # Predictions
    model.eval()
    with torch.no_grad():
        y_pred_prob = torch.sigmoid(model(X_test_t)).numpy()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Confusion matrix
    plot_confusion(y_test_seq, y_pred, "LSTM Confusion Matrix", "data/lstm_confusion_matrix.png")

    # Classification metrics
    acc = accuracy_score(y_test_seq, y_pred)
    prec = precision_score(y_test_seq, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print("\nClassification Report:\n", classification_report(y_test_seq, y_pred))

    # Strategy returns
    strategy_returns = np.where(y_pred.flatten() == 1,
                                test_df["returns"].iloc[timesteps-1:],
                                -test_df["returns"].iloc[timesteps-1:])
    sharpe_strategy = calculate_sharpe_ratio(strategy_returns)
    bh_returns = test_df["returns"].iloc[timesteps-1:]
    sharpe_bh = calculate_sharpe_ratio(bh_returns)

    print(f"Strategy Sharpe Ratio: {sharpe_strategy:.4f}")
    print(f"Buy & Hold Sharpe Ratio: {sharpe_bh:.4f}")

    # Paired t-test
    t_stat, p_value = ttest_rel(strategy_returns, bh_returns)
    print(f"T-test statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("✅ Statistically significant difference at 5% level.")
    else:
        print("❌ No statistically significant difference at 5% level.")

    # Bootstrapped Sharpe CI
    ci_low, ci_high = bootstrap_sharpe_diff(np.array(strategy_returns), np.array(bh_returns))
    print(f"Bootstrapped Sharpe Difference 95% CI: ({ci_low:.4f}, {ci_high:.4f})")

    # Plots
    plot_equity_curves(pd.Series(strategy_returns, index=test_df.index[timesteps-1:]),
                       pd.Series(bh_returns, index=test_df.index[timesteps-1:]),
                       index=test_df.index[timesteps-1:])

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig("data/lstm_loss_curve.png")
    print("[DONE] Loss curve saved to data/lstm_loss_curve.png")

pd.DataFrame({"pred": y_pred.flatten()}).to_csv("data/lstm_predictions.csv", index=False)
torch.save(model.state_dict(), "data/best_lstm_model.pth")

