import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

# ---------------------------
# Utility Functions
# ---------------------------
def load_datasets():
    train_df = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv("data/val.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("data/test.csv", index_col=0, parse_dates=True)
    return train_df, val_df, test_df

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    return np.sqrt(252*78) * (np.mean(excess_returns) / np.std(excess_returns))

def plot_equity_curves(strategy_returns, bh_returns, index):
    equity_strategy = (1 + strategy_returns).cumprod()
    equity_bh = (1 + bh_returns).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(index, equity_strategy, label="XGBoost Strategy", color="blue")
    plt.plot(index, equity_bh, label="Buy & Hold", color="orange", linestyle="--")
    plt.title("Equity Curve Comparison - XGBoost vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/xgboost_equity_curve.png")
    print("[DONE] Equity curve saved to data/xgboost_equity_curve.png")

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

    # Features / Target
    X_train = train_df.drop(columns=["target", "returns"]).values
    X_val   = val_df.drop(columns=["target", "returns"]).values
    X_test  = test_df.drop(columns=["target", "returns"]).values
    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_test = test_df["target"].values

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Combine train & val for final training
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    # Train XGBoost model
    print("[INFO] Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        eval_metric="logloss"
    )
    model.fit(X_train_full, y_train_full)

    # Predictions
    y_pred = model.predict(X_test)

    # Confusion Matrix
    plot_confusion(y_test, y_pred, "XGBoost Confusion Matrix", "data/xgboost_confusion_matrix.png")

    # Feature Importance
    plt.figure(figsize=(10, 6))
    plot_importance(model, max_num_features=15, importance_type="gain", title="XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig("data/xgboost_feature_importance.png")
    print("[DONE] Feature importance plot saved to data/xgboost_feature_importance.png")

    # Classification metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Strategy & Benchmark returns
    strategy_returns = np.where(y_pred.flatten() == 1,
                                test_df["returns"],
                                -test_df["returns"])
    sharpe_strategy = calculate_sharpe_ratio(strategy_returns)
    bh_returns = test_df["returns"]
    sharpe_bh = calculate_sharpe_ratio(bh_returns)

    print(f"XGBoost Strategy Sharpe Ratio: {sharpe_strategy:.4f}")
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

    # Equity Curves
    plot_equity_curves(pd.Series(strategy_returns, index=test_df.index),
                       pd.Series(bh_returns, index=test_df.index),
                       index=test_df.index)
pd.DataFrame({"pred": y_pred.flatten()}).to_csv("data/xgboost_predictions.csv", index=False)
model.save_model("data/xgb_model.json")
print("[DONE] XGBoost model saved to data/xgb_model.json")

