"""
Fama-French Three-Factor Model
================================
Runs FF3 regressions on simulated equity portfolios,
decomposes returns into market/size/value factors, and reports alpha.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def load_ff3_data():
    """Monthly FF3 factors (%) — hardcoded 2020-2024 snapshot from Ken French library."""
    monthly = {
        "2020-01":(-6.01,-1.29,-5.21,0.13),"2020-02":(-8.23,-0.69,-3.66,0.12),
        "2020-03":(-13.46,2.56,-1.45,0.13),"2020-04":(13.39,-2.32,-3.41,0.01),
        "2020-05":(5.40,2.74,-4.00,0.01),  "2020-06":(2.55,3.83,-2.99,0.01),
        "2020-07":(5.76,-4.45,-3.21,0.01), "2020-08":(7.64,-0.24,-3.03,0.01),
        "2020-09":(-3.63,0.08,-1.24,0.01), "2020-10":(-2.10,4.14,3.98,0.01),
        "2020-11":(12.48,5.60,2.35,0.01),  "2020-12":(4.65,4.87,3.33,0.01),
        "2021-01":(-0.04,3.23,3.67,0.00),  "2021-02":(2.90,2.73,7.09,0.00),
        "2021-03":(3.07,0.43,7.25,0.00),   "2021-04":(5.10,-2.89,1.52,0.00),
        "2021-05":(0.27,-0.43,4.83,0.00),  "2021-06":(2.74,0.41,-4.65,0.00),
        "2021-07":(1.45,-3.29,-2.02,0.00), "2021-08":(2.91,-0.23,-0.47,0.00),
        "2021-09":(-4.45,-0.05,2.85,0.00), "2021-10":(6.94,-3.18,0.52,0.00),
        "2021-11":(-1.82,-2.38,2.35,0.00), "2021-12":(3.50,0.44,4.02,0.01),
        "2022-01":(-5.86,-0.71,8.33,0.01), "2022-02":(-2.54,-1.30,2.98,0.01),
        "2022-03":(3.44,-0.55,2.33,0.02),  "2022-04":(-9.04,-0.94,0.57,0.02),
        "2022-05":(-0.12,0.27,3.55,0.03),  "2022-06":(-8.77,-2.05,0.42,0.06),
        "2022-07":(9.57,-1.84,-5.70,0.17), "2022-08":(-3.94,-1.71,1.31,0.19),
        "2022-09":(-9.37,-2.14,1.33,0.24), "2022-10":(8.02,4.37,7.47,0.25),
        "2022-11":(5.10,-0.73,9.39,0.31),  "2022-12":(-6.20,-1.13,2.42,0.35),
        "2023-01":(6.97,4.51,-4.41,0.38),  "2023-02":(-2.51,0.64,1.03,0.38),
        "2023-03":(3.50,-3.89,-6.76,0.40), "2023-04":(0.49,-0.19,-0.53,0.41),
        "2023-05":(0.35,-2.10,-2.70,0.44), "2023-06":(6.59,3.06,-2.57,0.44),
        "2023-07":(3.25,0.50,-0.40,0.45),  "2023-08":(-1.62,-2.65,-1.38,0.45),
        "2023-09":(-4.60,-1.10,-1.30,0.45),"2023-10":(-2.21,-3.53,-1.72,0.46),
        "2023-11":(9.05,3.40,-2.11,0.46),  "2023-12":(4.84,3.05,-0.78,0.44),
        "2024-01":(0.54,-3.97,-0.28,0.44), "2024-02":(5.51,1.71,2.30,0.44),
        "2024-03":(3.09,2.15,5.14,0.44),   "2024-04":(-4.14,-2.44,1.96,0.44),
        "2024-05":(4.83,-0.69,-1.87,0.44), "2024-06":(3.29,-2.87,-0.67,0.44),
    }
    rows = [{"date": pd.to_datetime(d), "Mkt-RF": v[0], "SMB": v[1],
             "HML": v[2], "RF": v[3]} for d, v in monthly.items()]
    return pd.DataFrame(rows).set_index("date") / 100

def simulate_portfolios(ff, seed=42):
    np.random.seed(seed)
    n = len(ff)
    eps = np.random.normal(0, 0.015, (n, 3))
    # [alpha/month, beta_mkt, beta_smb, beta_hml]
    specs = {
        "Growth_Tech": (0.003,  1.20, -0.30, -0.50),
        "Value_Small":  (0.001,  0.85,  0.60,  0.70),
        "Balanced":    (0.002,  1.00,  0.10,  0.10),
    }
    out = {}
    for i, (name, (a, b1, b2, b3)) in enumerate(specs.items()):
        out[name] = a + b1*ff["Mkt-RF"] + b2*ff["SMB"] + b3*ff["HML"] + eps[:, i]
    return pd.DataFrame(out, index=ff.index)

def run_regression(port_ret, ff):
    excess = port_ret - ff["RF"]
    X = np.column_stack([np.ones(len(ff)), ff["Mkt-RF"], ff["SMB"], ff["HML"]])
    y = excess.values
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha, b_mkt, b_smb, b_hml = coef
    yhat = X @ coef
    ss_res = np.sum((y - yhat)**2); ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    n, k = len(y), 4
    se = np.sqrt(np.diag(ss_res/(n-k) * np.linalg.inv(X.T @ X)))
    t_a = alpha / se[0]
    p_a = 2*(1 - stats.t.cdf(abs(t_a), df=n-k))
    return {"alpha_m": alpha, "alpha_ann": alpha*12,
            "beta_mkt": b_mkt, "beta_smb": b_smb, "beta_hml": b_hml,
            "r2": r2, "t_alpha": t_a, "p_alpha": p_a,
            "fitted": pd.Series(yhat, index=excess.index),
            "residuals": pd.Series(y - yhat, index=excess.index)}

def plot_all(portfolios, ff, results):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("Fama-French Three-Factor Model\nReturn Decomposition & Alpha Analysis",
                 fontsize=13, fontweight="bold")
    colors = {"Growth_Tech": "#1A4F8A", "Value_Small": "#D62728", "Balanced": "#2CA02C"}
    names = list(results.keys())

    ax = axes[0, 0]
    for name, col in colors.items():
        cum = (1 + portfolios[name]).cumprod() - 1
        ax.plot(portfolios.index, cum*100, color=col, lw=1.8, label=name)
    ax.plot(ff.index, ((1+ff["Mkt-RF"]).cumprod()-1)*100, "k--", lw=1, label="Mkt-RF")
    ax.set_title("Cumulative Excess Returns"); ax.set_ylabel("Return (%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[0, 1]
    x = np.arange(3); w = 0.25
    for i, (key, label, col) in enumerate([("beta_mkt","Mkt-RF","#1A4F8A"),
                                             ("beta_smb","SMB","#FF7F0E"),
                                             ("beta_hml","HML","#2CA02C")]):
        ax.bar(x + i*w, [results[n][key] for n in names], w, label=label, color=col, alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Factor Loadings (β)"); ax.set_ylabel("Loading")
    ax.set_xticks(x + w); ax.set_xticklabels(names, fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    alphas = [results[n]["alpha_ann"]*100 for n in names]
    pvals  = [results[n]["p_alpha"] for n in names]
    bars = ax.bar(names, alphas, color=["#2CA02C" if a>0 else "#D62728" for a in alphas], alpha=0.8)
    for bar, p in zip(bars, pvals):
        sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, sig, ha="center", fontsize=11)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Annualised Alpha (% p.a.)  * p<0.1  ** p<0.05")
    ax.set_ylabel("Alpha (%)"); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    r2s = [results[n]["r2"]*100 for n in names]
    ax.bar(names, r2s, color=["#1A4F8A","#D62728","#2CA02C"], alpha=0.8)
    ax.set_title("R² — Variance Explained by FF3 Factors")
    ax.set_ylabel("R² (%)"); ax.set_ylim(0, 105); ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(r2s): ax.text(i, v+1, f"{v:.1f}%", ha="center", fontsize=9)

    ax = axes[2, 0]
    name = "Balanced"
    exc = portfolios[name] - ff["RF"]
    ax.scatter(results[name]["fitted"], exc, alpha=0.5, s=20, color="#1A4F8A")
    mn, mx = results[name]["fitted"].min(), results[name]["fitted"].max()
    ax.plot([mn, mx], [mn, mx], "r--", lw=1, label="Perfect fit")
    ax.set_title(f"Actual vs Fitted ({name})")
    ax.set_xlabel("FF3 Fitted"); ax.set_ylabel("Actual Excess Return")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    for name, col in colors.items():
        ax.plot(results[name]["residuals"].index, results[name]["residuals"]*100,
                color=col, lw=0.8, alpha=0.7, label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Residuals (Unexplained Return)"); ax.set_ylabel("Residual (%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig("fama_french_analysis.png", dpi=150, bbox_inches="tight")
    print("Saved: fama_french_analysis.png")
    plt.close()

def main():
    ff = load_ff3_data()
    portfolios = simulate_portfolios(ff)
    print("── FF3 Regression Results ──────────────────────────────")
    results = {}
    for name in portfolios.columns:
        res = run_regression(portfolios[name], ff)
        results[name] = res
        sig = "***" if res["p_alpha"]<0.01 else "**" if res["p_alpha"]<0.05 else "*" if res["p_alpha"]<0.10 else ""
        print(f"\n  {name}:")
        print(f"    Alpha (ann.): {res['alpha_ann']*100:.2f}%  t={res['t_alpha']:.2f}  p={res['p_alpha']:.3f} {sig}")
        print(f"    β_Mkt={res['beta_mkt']:.3f}  β_SMB={res['beta_smb']:.3f}  β_HML={res['beta_hml']:.3f}  R²={res['r2']*100:.1f}%")
    plot_all(portfolios, ff, results)

if __name__ == "__main__":
    main()
