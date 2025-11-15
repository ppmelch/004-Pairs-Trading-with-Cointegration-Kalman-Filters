import numpy as np
import pandas as pd
from classes import config, Position
from kalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller

def get_portfolio_value(cash, longs, shorts, y, x):
    value = cash
    for p in longs:
        px = y if p.ticker == "Y" else x
        value += p.n_shares * px
    for p in shorts:
        px = y if p.ticker == "Y" else x
        value += (p.entry_price - px) * p.n_shares
    return value


def backtest(data: pd.DataFrame, initial_cash=None):

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

        # === 1. KF Hedge Ratio ===
        w_pred, P_pred = k_hr.predict()
        k_hr.update(np.array([1, x]), y, w_pred, P_pred)
        beta = k_hr.w_t[1]

        # === 2. Spread ===
        spread = y - beta * x

        # === 3. KF VECM ===
        wp, Pp = k_vecm.predict()
        k_vecm.update(np.array([1]), spread, wp, Pp)
        spr_hat = k_vecm.w_t[0]
        spread_history.append(spr_hat)

        # === 4. Z-Score ===
        if len(spread_history) < WINDOW:
            equity.append(get_portfolio_value(cash, longs, shorts, y, x))
            continue

        mu = np.mean(spread_history[-WINDOW:])
        sd = np.std(spread_history[-WINDOW:])
        sd = sd if sd > 0 else 1e-6
        z = (spr_hat - mu) / sd

        # === 5. Rolling Cointegration Protection ===
        recent = pd.Series(spread_history[-WINDOW:])
        adf_stat, pvalue, *_ = adfuller(recent)
        allow_entries = pvalue <= 0.05

        # === 6. Borrow cost ===
        for p in shorts:
            px = y if p.ticker == "Y" else x
            daily_cost = p.n_shares * px * BR_daily
            cash -= daily_cost
            total_borrow_cost += daily_cost

        # === 7. STOP LOSS ===
        if (longs or shorts) and abs(z) > STOP_Z:

            # Close longs
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

            # Close shorts
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

        # === 8. EXIT SIGNAL ===
        if (longs or shorts) and abs(z) < EXIT_Z:

            # Close longs
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

            # Close shorts
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

        # === 9. ENTRY ===
        if allow_entries and not longs and not shorts:

            capital_to_use = cash * INVEST

            # Short Y / Long X
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

            # Long Y / Short X
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
