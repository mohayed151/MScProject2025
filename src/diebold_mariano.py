import pandas as pd
import numpy as np
from scipy.stats import t

# --------------------------
# DM Test
# --------------------------
def diebold_mariano_test(e1, e2, crit="MSE"):
    if crit == "MSE":
        d = e1**2 - e2**2
    elif crit == "MAD":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("crit must be 'MSE' or 'MAD'")
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=len(d) - 1))
    return dm_stat, p_value

# --------------------------
# Sharpe Ratio Functions
# --------------------------
def sharpe_ratio(returns):
    if np.std(returns) == 0:
        return 0.0
    return np.sqrt(252 * 78) * (np.mean(returns) / np.std(returns))

def bootstrap_sharpe_diff(r1, r2, n_bootstrap=1000):
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(r1), len(r1), replace=True)
        diffs.append(sharpe_ratio(r1[idx]) - sharpe_ratio(r2[idx]))
    return np.percentile(diffs, [2.5, 97.5])

def significance_marker(p):
    if p < 0.01:
        return "\\textsuperscript{***}"
    elif p < 0.05:
        return "\\textsuperscript{**}"
    elif p < 0.1:
        return "\\textsuperscript{*}"
    return ""

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # Load ground truth
    df = pd.concat([
        pd.read_csv("data/train.csv", index_col=0),
        pd.read_csv("data/val.csv", index_col=0),
        pd.read_csv("data/test.csv", index_col=0)
    ])
    y_true = df["target"].values

    # Load predictions
    preds_lstm = pd.read_csv("data/walkforward_preds_lstm.csv")["pred"].values
    preds_xgb = pd.read_csv("data/walkforward_preds_xgb.csv")["pred"].values
    preds_lp = pd.read_csv("data/walkforward_preds_lastprice.csv")["pred"].values

    # Align lengths
    min_len = min(len(y_true), len(preds_lstm), len(preds_xgb), len(preds_lp))
    y_true = y_true[-min_len:]
    preds_lstm = preds_lstm[-min_len:]
    preds_xgb = preds_xgb[-min_len:]
    preds_lp = preds_lp[-min_len:]

    # Errors
    e_lstm = (preds_lstm != y_true).astype(int)
    e_xgb = (preds_xgb != y_true).astype(int)
    e_lp = (preds_lp != y_true).astype(int)

    # Load returns
    returns_lstm = pd.read_csv("data/walkforward_returns_lstm.csv")["returns"].values
    returns_xgb = pd.read_csv("data/walkforward_returns_xgb.csv")["returns"].values
    returns_lp = pd.read_csv("data/walkforward_returns_lastprice.csv")["returns"].values

    # Store results
    dm_results = []
    sharpe_results = []

    comparisons = [
        ("LSTM", e_lstm, returns_lstm, "XGBoost", e_xgb, returns_xgb),
        ("LSTM", e_lstm, returns_lstm, "LastPrice", e_lp, returns_lp),
        ("XGBoost", e_xgb, returns_xgb, "LastPrice", e_lp, returns_lp),
    ]

    print("\n📊 Diebold-Mariano Forecast Accuracy Tests:\n")
    for name_a, e_a, r_a, name_b, e_b, r_b in comparisons:
        dm_stat, p_val = diebold_mariano_test(e_a, e_b, crit="MSE")
        dm_results.append([name_a, name_b, dm_stat, p_val])
        print(f"{name_a} vs {name_b}: DM stat = {dm_stat:.4f}, p-value = {p_val:.4f}")

    print("\n📊 Sharpe Ratio Difference Tests:\n")
    for name_a, e_a, r_a, name_b, e_b, r_b in comparisons:
        sr_a = sharpe_ratio(r_a)
        sr_b = sharpe_ratio(r_b)
        ci_low, ci_high = bootstrap_sharpe_diff(r_a, r_b)
        sharpe_results.append([sr_a, sr_b, ci_low, ci_high])
        print(f"{name_a} vs {name_b}: Sharpe {sr_a:.4f} vs {sr_b:.4f}, CI diff = ({ci_low:.4f}, {ci_high:.4f})")

    # --------------------------
    # Generate LaTeX table
    # --------------------------
    latex_output = "\\begin{table}[H]\n\\centering\n\\caption{Diebold--Mariano and Sharpe Ratio Difference Tests}\n"
    latex_output += "\\begin{tabular}{l l r r r r r r}\n"
    latex_output += "\\hline\n"
    latex_output += "Model A & Model B & DM Stat & p-value & Sharpe A & Sharpe B & CI Low & CI High \\\\\n"
    latex_output += "\\hline\n"

    for i in range(len(dm_results)):
        name_a, name_b, dm_stat, p_val = dm_results[i]
        sr_a, sr_b, ci_low, ci_high = sharpe_results[i]
        p_with_star = f"{p_val:.3f}{significance_marker(p_val)}"
        latex_output += f"{name_a} & {name_b} & {dm_stat:.4f} & {p_with_star} & {sr_a:.4f} & {sr_b:.4f} & {ci_low:.4f} & {ci_high:.4f} \\\\\n"

    latex_output += "\\hline\n\\end{tabular}\n\\label{tab:dm_sharpe_results}\n\\end{table}\n"

    with open("data/dm_sharpe_table.tex", "w") as f:
        f.write(latex_output)

    print("\n[DONE] LaTeX table saved to data/dm_sharpe_table.tex")
