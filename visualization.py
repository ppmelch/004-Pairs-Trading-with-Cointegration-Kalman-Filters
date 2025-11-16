from libraries import *
from kalman import KalmanFilter


BLUE = "#1D4782"
BLUE_LIGHT = "#5A78A1"
BLUE_SOFT = "#A9BBD6"
RED_SOFT = "#EA6767"

sns.set_style("whitegrid")


def plot_normalized_data(data: pd.DataFrame):
    """Plot normalized (z-scored) price series."""
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


def plot_spread(data: pd.DataFrame, beta_series=None):
    """Plot spread y − βx with ±1σ and ±2σ bands."""
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

    plt.figure(figsize=(12, 5))
    plt.plot(spread, color=BLUE, lw=1.8, label="Spread")
    plt.axhline(μ, ls="--", color=RED_SOFT, label="Mean")

    plt.fill_between(spread.index, μ - σ, μ + σ, alpha=0.15, color=BLUE_SOFT, label="±1σ")
    plt.fill_between(spread.index, μ - 2 * σ, μ + 2 * σ, alpha=0.10, color=RED_SOFT, label="±2σ")

    plt.title(f"Spread — {data.columns[0]} vs {data.columns[1]}")
    plt.xlabel("Date")
    plt.ylabel("Spread (y − βx)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_dynamic_hedge_ratio(data: pd.DataFrame):
    """Estimate and plot dynamic hedge ratio β(t) from KalmanFilter n=2."""
    kf = KalmanFilter(n=2)
    betas = []

    for _, row in data.iterrows():
        y = float(row.iloc[0])
        x = float(row.iloc[1])
        w_pred, P_pred = kf.predict()
        w_upd, _ = kf.update(np.array([1, x]), y, w_pred, P_pred)
        betas.append(w_upd[1])

    betas = pd.Series(betas, index=data.index)
    mean_beta = betas.mean()

    plt.figure(figsize=(12, 5))
    plt.plot(betas, color=BLUE, lw=1.8, label="β(t)")
    plt.axhline(mean_beta, ls="--", lw=1.3, color=RED_SOFT, label=f"Mean β = {mean_beta:.4f}")
    plt.title("Hedge Ratio Evolution (Kalman 1)")
    plt.xlabel("Date")
    plt.ylabel("β(t)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return betas


def plot_kalman_eigenvectors(data: pd.DataFrame, eigenvector=None):
    """Plot the static first cointegration eigenvector from Johansen."""
    if eigenvector is None:
        raise ValueError("Provide eigenvector=[v1,v2]")

    v1 = np.full(len(data), eigenvector[0])
    v2 = np.full(len(data), eigenvector[1])

    plt.figure(figsize=(12, 5))
    plt.plot(data.index, v1, color=BLUE, lw=1.8, label="v1")
    plt.plot(data.index, v2, color=BLUE_LIGHT, lw=1.8, label="v2")
    plt.title("First Eigenvector from Johansen Cointegration Test")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_kalman_vecm(spread: pd.Series) -> pd.Series:
    """
    Smooth the raw spread with a 1D Kalman filter.
    This acts like a VECM-like filtered signal.
    The Kalman gain ~0.05 produces a nice smooth curve.
    """

    # Inicialización
    x = spread.iloc[0]
    P = 1.0
    Q = 0.01   # process noise (suavidad)
    R = 1.0    # measurement noise

    smoothed = []

    for z in spread:
        # Predicción
        x_pred = x
        P_pred = P + Q

        # Ganancia de Kalman
        K = P_pred / (P_pred + R)

        # Actualización
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred

        smoothed.append(x)

    return pd.Series(smoothed, index=spread.index)

def plot_spread_vs_vecm(spread, vecm_signal):
    """Compare raw spread vs. Kalman-smoothed VECM-like signal."""
    c = spread.index.intersection(vecm_signal.index)
    s = spread.loc[c]
    v = vecm_signal.loc[c]

    plt.figure(figsize=(12, 5))
    plt.plot(s, color=BLUE, lw=1.8, label="Spread")
    plt.plot(v, color=BLUE_LIGHT, lw=1.8, label="VECM Signal")
    plt.title("Spread vs. Normalized VECM Prediction")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_single_split(port_series: pd.Series, title="Portfolio Value"):
    """Plot a single portfolio time series."""
    plt.figure(figsize=(10, 5))
    plt.plot(port_series.index, port_series.values, lw=1.8, color=BLUE)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_test_validation(p_test: pd.Series, p_val: pd.Series):
    """Compare test vs validation portfolios."""
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


def plot_trade_returns(pnl_list: list):
    """Distribution of trade-level profit/loss."""
    pnl = pd.Series(pnl_list)
    μ = pnl.mean()
    med = pnl.median()

    plt.figure(figsize=(10, 5))
    sns.histplot(pnl, kde=True, color=BLUE, bins=20)
    plt.axvline(μ, ls="--", color=RED_SOFT, label=f"Mean = {μ:,.2f}")
    plt.axvline(med, ls="--", color="red", label=f"Median = {med:,.2f}")
    plt.title("Distribution of Trade Returns")
    plt.xlabel("Profit per Trade ($)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_portfolio_splits(port_train: pd.Series, port_test: pd.Series, port_val: pd.Series):
    """Plot Train/Test/Validation shaded portfolio evolution."""
    full_port = pd.concat([port_train, port_test, port_val])

    t0 = full_port.index[0]
    t1 = port_train.index[-1]
    t2 = port_test.index[-1]
    t3 = port_val.index[-1]

    plt.figure(figsize=(10, 5))
    plt.axvspan(t0, t1, color=RED_SOFT, alpha=0.12, label="Train")
    plt.axvspan(t1, t2, color=BLUE_LIGHT, alpha=0.12, label="Test")
    plt.axvspan(t2, t3, color=BLUE, alpha=0.12, label="Validation")

    plt.plot(full_port.index, full_port.values, lw=1.4, color=BLUE)
    plt.title("Portfolio Value Evolution")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
