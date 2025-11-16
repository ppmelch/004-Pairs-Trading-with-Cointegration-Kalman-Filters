from libraries import *
from kalman import KalmanFilter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
from kalman import KalmanFilter

# =============================
# COLORS (UNIFIED BLUE THEME)
# =============================
BLUE        = "#1D4782"
BLUE_LIGHT  = "#5A78A1"
BLUE_SOFT   = "#A9BBD6"
RED_SOFT    = "#EA6767"

sns.set_style("whitegrid")

# ===============================================================
# 1) NORMALIZED PRICE SERIES (BLUE THEME)
# ===============================================================
def plot_normalized_data(data: pd.DataFrame):
    norm = (data - data.mean()) / data.std()

    plt.figure(figsize=(10, 5))
    plt.plot(norm.iloc[:, 0], label=data.columns[0], color=BLUE)
    plt.plot(norm.iloc[:, 1], label=data.columns[1], color=RED_SOFT)

    plt.title("Normalized Price Series")
    plt.xlabel("Date")
    plt.ylabel("Z-Score Normalized")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===============================================================
# 2) SPREAD WITH ±1σ ±2σ BANDS (BLUE THEME)
# ===============================================================
def plot_spread(data: pd.DataFrame, beta_series=None):
    y = data.iloc[:, 0]
    x = data.iloc[:, 1]

    if beta_series is None:
        model = sm.OLS(y, sm.add_constant(x)).fit()
        spread = model.resid
    else:
        beta = beta_series.reindex(data.index).fillna(method="ffill")
        spread = y - beta * x

    μ = spread.mean()
    σ = spread.std()

    plt.figure(figsize=(10, 5))
    plt.plot(spread, color=BLUE, lw=1.3, label="Spread")

    plt.axhline(μ, ls="--", color="red", label="Mean")
    plt.fill_between(spread.index, μ - σ, μ + σ, alpha=0.15, color="grey", label="±1σ")
    plt.fill_between(spread.index, μ - 2*σ, μ + 2*σ, alpha=0.10, color=RED_SOFT, label="±2σ")

    plt.title(f"Spread — {data.columns[0]} vs {data.columns[1]}")
    plt.xlabel("Date")
    plt.ylabel("Spread (y − βx)")
    plt.grid(True, alpha=0.25)
    plt.legend( loc="upper left")
    plt.tight_layout()
    plt.show()


# ===============================================================
# 3) KALMAN HEDGE RATIO (BLUE THEME)
# ===============================================================
def plot_dynamic_hedge_ratio(data: pd.DataFrame):
    kf = KalmanFilter(n=2)
    betas, alphas = [], []

    for _, row in data.iterrows():
        y = row.iloc[0]
        x = row.iloc[1]

        w_pred, P_pred = kf.predict()
        w_upd, _ = kf.update(np.array([1, x]), y, w_pred, P_pred)

        alphas.append(w_upd[0])
        betas.append(w_upd[1])

    betas = pd.Series(betas, index=data.index)
    alphas = pd.Series(alphas, index=data.index)

    plt.figure(figsize=(10, 5))
    plt.plot(betas, lw=1.6, color=BLUE, label="β_t (Hedge Ratio)")
    plt.plot(alphas, lw=1.6, color=RED_SOFT, label="α_t (Intercept)")

    plt.title("Dynamic Hedge Ratio (Kalman Filter)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return alphas, betas


# ===============================================================
# 4) EIGENVECTORS (BLUE THEME)
# ===============================================================
def plot_kalman_eigenvectors(data: pd.DataFrame):
    kf = KalmanFilter(n=2)
    v1_list, v2_list = [], []

    for _, row in data.iterrows():
        p1 = float(row.iloc[0])
        p2 = float(row.iloc[1])

        # Model: v1*p1 + v2*p2 = 0
        y_t = 0.0
        w_pred, P_pred = kf.predict()
        w_upd, _ = kf.update(np.array([p1, p2]), y_t, w_pred, P_pred)

        v1_list.append(w_upd[0])
        v2_list.append(w_upd[1])

    v1 = pd.Series(v1_list, index=data.index)
    v2 = pd.Series(v2_list, index=data.index)

    plt.figure(figsize=(10, 5))
    plt.plot(v1, label="V1t", color=BLUE, lw=1.6)
    plt.plot(v2, label="V2t", color=BLUE_LIGHT, lw=1.6)

    plt.title("Dynamic Estimated Eigenvectors (Kalman 2)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return v1, v2


# ===============================================================
# 5) SPREAD VS VECM (BLUE THEME)
# ===============================================================
def plot_spread_vs_vecm(spread_series: pd.Series, vecm_series: pd.Series):

    common = spread_series.index.intersection(vecm_series.index)
    s = spread_series.loc[common]
    v = vecm_series.loc[common]

    plt.figure(figsize=(10, 5))
    plt.plot(s, label="Spread", color=BLUE)
    plt.plot(v, label="VECM Prediction", color=BLUE_LIGHT)

    plt.title("Comparison: Spread vs Normalized VECM")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===============================================================
# 6) TRADING SIGNALS (BLUE THEME)
# ===============================================================
def plot_trading_signals(z_series: pd.Series, long_idx, short_idx, exit_idx):

    plt.figure(figsize=(10, 5))

    plt.plot(z_series.index, z_series.values, label="Z-Score", color=BLUE)

    plt.scatter(long_idx, z_series.loc[long_idx], marker="^",
                color=BLUE, s=60, label="Long Entry")

    plt.scatter(short_idx, z_series.loc[short_idx], marker="v",
                color=BLUE_LIGHT, s=60, label="Short Entry")

    plt.scatter(exit_idx, z_series.loc[exit_idx], marker="x",
                color=RED_SOFT, s=80, label="Exit")

    plt.axhline(1, ls="--", color=RED_SOFT)
    plt.axhline(-1, ls="--", color=RED_SOFT)

    plt.title("Trading Signals — Normalized VECM (Kalman 2)")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===============================================================
# 7) TRADE RETURNS HISTOGRAM (BLUE THEME)
# ===============================================================
def plot_trade_returns(pnl_list: list):

    pnl = pd.Series(pnl_list)
    μ = pnl.mean()
    med = pnl.median()

    plt.figure(figsize=(10, 5))
    sns.histplot(pnl, kde=True, color=RED_SOFT, bins=20)

    plt.axvline(μ, ls="--", color=BLUE, label=f"Mean = {μ:,.2f}")
    plt.axvline(med, ls="--", color=BLUE_LIGHT, label=f"Median = {med:,.2f}")

    plt.title("Distribution of Trade Returns")
    plt.xlabel("PnL per Trade ($)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===============================================================
# 8) SINGLE SPLIT PORTFOLIO CURVE (BLUE THEME)
# ===============================================================
def plot_single_split(port_series: pd.Series, title="Portfolio Value"):
    plt.figure(figsize=(10, 5))
    plt.plot(port_series.index, port_series.values, lw=1.8, color=BLUE)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


# ===============================================================
# 9) TEST VS VALIDATION (BLUE THEME)
# ===============================================================
def plot_test_validation(p_test: pd.Series, p_val: pd.Series):
    plt.figure(figsize=(10, 5))
    plt.plot(p_test.index, p_test.values, lw=1.6, color=BLUE, label="TEST")
    plt.plot(p_val.index, p_val.values, lw=1.6, color=RED_SOFT, label="VALIDATION")

    plt.title("Test vs Validation Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===============================================================
# 10) PORTFOLIO SPLITS (TRAIN/TEST/VALIDATION) BLUE THEME
# ===============================================================
def plot_portfolio_splits(port_train: pd.Series,
                          port_test: pd.Series,
                          port_val: pd.Series) -> None:

    full_port = pd.concat([port_train, port_test, port_val])

    t_start     = full_port.index[0]
    t_train_end = port_train.index[-1]
    t_test_end  = port_test.index[-1]
    t_val_end   = port_val.index[-1]

    plt.figure(figsize=(10, 5))

    plt.axvspan(t_start, t_train_end, color=RED_SOFT,  alpha=0.12, label="Train (60%)")
    plt.axvspan(t_train_end, t_test_end, color=BLUE_LIGHT, alpha=0.12, label="Test (20%)")
    plt.axvspan(t_test_end, t_val_end, color=BLUE,       alpha=0.12, label="Validation (20%)")

    plt.plot(full_port.index, full_port.values, lw=1.4, color=BLUE, label="Portfolio Value")

    plt.title("Portfolio Value Evolution", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
