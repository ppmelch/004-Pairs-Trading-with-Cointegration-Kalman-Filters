from libraries import *
from classes import config, Position
from kalman import KalmanFilter


def get_portfolio_value(cash, longs, shorts, y, x):
    """
    Compute the current portfolio value by marking long and short positions to market.

    Parameters
    ----------
    cash : float
        Current cash balance.
    longs : list
        Active long positions.
    shorts : list
        Active short positions.
    y : float
        Latest price of asset Y.
    x : float
        Latest price of asset X.

    Returns
    -------
    float
        Mark-to-market portfolio value.
    """
    value = cash
    for p in longs:
        px = y if p.ticker == "Y" else x
        value += p.n_shares * px
    for p in shorts:
        px = y if p.ticker == "Y" else x
        value += (p.entry_price - px) * p.n_shares
    return value


def backtest(data: pd.DataFrame, initial_cash=None):
    """
    Execute a Kalman-filter-based pairs trading backtest with dynamic hedge ratios,
    VECM smoothing, rolling cointegration checks, and realistic transaction costs.

    Parameters
    ----------
    data : pd.DataFrame
        Two-asset price series ordered in time.
    initial_cash : float, optional
        Initial portfolio cash. If None, uses the value defined in config.

    Returns
    -------
    tuple
        (
            equity_curve : pd.Series,
            final_cash : float,
            win_rate : float,
            n_buy : int,
            n_sell : int,
            n_hold : int,
            n_closed_positions : int,
            closed_positions : list,
            total_borrow_cost : float,
            total_commission_cost : float
        )
    """
    data = data.copy()
    cash = config.capital if initial_cash is None else initial_cash

    COM = config.COM
    INVEST = config.INVEST
    ENTRY_Z = config.ENTRY_Z
    EXIT_Z = config.EXIT_Z
    STOP_Z = 3.5
    BR_daily = config.BR / 252
    WINDOW = config.TDays

    k_hr = KalmanFilter(n=2)
    k_vecm = KalmanFilter(n=1)

    longs, shorts = [], []
    closed_positions = []
    spread_history = []
    equity = []

    buy = sell = hold = 0
    total_borrow_cost = 0.0
    total_commission_cost = 0.0

    allow_entries = True

    for date, row in data.iterrows():

        y = row.iloc[0]
        x = row.iloc[1]

        w_pred, P_pred = k_hr.predict()
        k_hr.update(np.array([1, x]), y, w_pred, P_pred)
        beta = k_hr.w_t[1]

        spread = y - beta * x

        wp, Pp = k_vecm.predict()
        k_vecm.update(np.array([1]), spread, wp, Pp)
        spr_hat = k_vecm.w_t[0]
        spread_history.append(spr_hat)

        if len(spread_history) < WINDOW:
            equity.append(get_portfolio_value(cash, longs, shorts, y, x))
            continue

        mu = np.mean(spread_history[-WINDOW:])
        sd = np.std(spread_history[-WINDOW:])
        sd = sd if sd > 0 else 1e-6
        z = (spr_hat - mu) / sd

        recent = pd.Series(spread_history[-WINDOW:])
        adf_stat, pvalue, *_ = adfuller(recent)
        allow_entries = pvalue <= 0.05

        for p in shorts:
            px = y if p.ticker == "Y" else x
            daily_cost = p.n_shares * px * BR_daily
            cash -= daily_cost
            total_borrow_cost += daily_cost

        if (longs or shorts) and abs(z) > STOP_Z:

            for p in longs[:]:
                px = y if p.ticker == "Y" else x
                com = p.n_shares * px * COM
                pnl = (px - p.entry_price) * p.n_shares - com

                cash += (px * p.n_shares) - com
                total_commission_cost += com

                p.exit_price = px
                p.profit = pnl
                closed_positions.append(p)
                longs.remove(p)
                sell += 1

            for p in shorts[:]:
                px = y if p.ticker == "Y" else x
                com = p.n_shares * px * COM
                pnl = (p.entry_price - px) * p.n_shares - com

                cash += pnl
                total_commission_cost += com

                p.exit_price = px
                p.profit = pnl
                closed_positions.append(p)
                shorts.remove(p)
                sell += 1

            equity.append(get_portfolio_value(cash, longs, shorts, y, x))
            continue

        if (longs or shorts) and abs(z) < EXIT_Z:

            for p in longs[:]:
                px = y if p.ticker == "Y" else x
                com = p.n_shares * px * COM
                pnl = (px - p.entry_price) * p.n_shares - com

                cash += (px * p.n_shares) - com
                total_commission_cost += com

                p.exit_price = px
                p.profit = pnl
                closed_positions.append(p)
                longs.remove(p)
                sell += 1

            for p in shorts[:]:
                px = y if p.ticker == "Y" else x
                com = p.n_shares * px * COM
                pnl = (p.entry_price - px) * p.n_shares - com

                cash += pnl
                total_commission_cost += com

                p.exit_price = px
                p.profit = pnl
                closed_positions.append(p)
                shorts.remove(p)
                sell += 1

            equity.append(get_portfolio_value(cash, longs, shorts, y, x))
            continue

        if allow_entries and not longs and not shorts:

            capital_to_use = cash * INVEST

            if z > ENTRY_Z:
                n = int(capital_to_use / (abs(y) + abs(beta * x)))
                if n > 0:
                    comY = n * y * COM
                    comX = n * x * COM
                    costX = n * x

                    if cash >= costX + comY:
                        cash -= costX
                        longs.append(Position(n, "X", x, "LONG", date))

                        cash -= comY
                        shorts.append(Position(n, "Y", y, "SHORT", date))

                        total_commission_cost += (comY + comX)
                        buy += 1

            elif z < -ENTRY_Z:
                n = int(capital_to_use / (abs(y) + abs(beta * x)))
                if n > 0:
                    comY = n * y * COM
                    comX = n * x * COM
                    costY = n * y

                    if cash >= costY + comX:
                        cash -= costY
                        longs.append(Position(n, "Y", y, "LONG", date))

                        cash -= comX
                        shorts.append(Position(n, "X", x, "SHORT", date))

                        total_commission_cost += (comY + comX)
                        buy += 1

        else:
            hold += 1

        equity.append(get_portfolio_value(cash, longs, shorts, y, x))

    equity = pd.Series(equity, index=data.index[:len(equity)])

    win_rate = (
        sum(p.profit > 0 for p in closed_positions) / len(closed_positions)
        if closed_positions else 0
    )

    return (
        equity,
        cash,
        win_rate,
        buy,
        sell,
        hold,
        len(closed_positions),
        closed_positions,
        total_borrow_cost,
        total_commission_cost,
    )
