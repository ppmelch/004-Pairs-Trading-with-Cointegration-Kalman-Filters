from libraries import *
from kalman import KalmanFilter
from classes import Position, config
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def get_portfolio_value(cash, active_long_ops, active_short_ops, y, x):

    val = cash

    # LONG POSITIONS – simple mark to market
    for pos in active_long_ops:
        if pos.ticker == "Y":
            val += pos.n_shares * y
        elif pos.ticker == "X":
            val += pos.n_shares * x

    # SHORT POSITIONS – academic PNL model: (open - current) * shares
    for pos in active_short_ops:
        if pos.ticker == "Y":
            val += (pos.entry_price - y) * pos.n_shares
        elif pos.ticker == "X":
            val += (pos.entry_price - x) * pos.n_shares

    return val



def backtest(data: pd.DataFrame):
    data = data.copy()

    # ===== PARAMS =====
    DAYS = config.TDays              # Rolling window for Johansen & normalization
    COM = config.COM                 # Commission per trade
    INVEST = config.INVEST           # Fraction of capital used per trade
    BR_daily = config.BR / DAYS            # Daily borrow cost
    theta = config.theta             # Entry Z-threshold# Exit Z-threshold (realistic; 0.05 is too tight)
    EXIT_TH = config.EXIT_TH
    cash = config.capital            # Initial capital

    # ===== KALMAN FILTERS =====
    k1 = KalmanFilter(n=2)           # Hedge ratio
    k2 = KalmanFilter(n=1)           # VECM smoothing

    # ===== STORAGE =====
    vecm_hat_list = []
    position_long = []
    position_short = []
    closed_positions = []
    equity_curve = []
    entry_idx_list = []
    exit_idx_list = []

    buy = sell = hold = 0

    # MAIN LOOP
    for idx, (date, row) in enumerate(data.iterrows()):
        y = row.iloc[0]   # Asset1
        x = row.iloc[1]   # Asset2

        # =============================
        # 1) KALMAN FILTER #1 — Hedge Ratio
        # =============================
        wp, Pp = k1.predict()
        k1.update(np.array([1, x]), y, wp, Pp)
        beta = k1.w_t[1]

        # =============================
        # 2) JOHANSEN Rolling
        # =============================
        if idx < DAYS:
            equity_curve.append(get_portfolio_value(cash, position_long, position_short, y, x))
            continue

        window_data = data.iloc[idx - DAYS:idx]
        eig = coint_johansen(window_data, 0, 1)

        e1, e2 = eig.evec[0, 0], eig.evec[1, 0]
        vecm_t = e1 * y + e2 * x

        # =============================
        # 3) KALMAN FILTER #2 — VECM smoothing
        # =============================
        w2p, P2p = k2.predict()
        k2.update(np.array([1]), vecm_t, w2p, P2p)
        vecm_hat = k2.w_t[0]
        vecm_hat_list.append(vecm_hat)

        # =============================
        # 4) NORMALIZED VECM → SIGNAL
        # =============================
        if len(vecm_hat_list) < DAYS:
            equity_curve.append(get_portfolio_value(cash, position_long, position_short, y, x))
            continue

        window_vecm = np.array(vecm_hat_list[-DAYS:])
        mu = window_vecm.mean()
        sd = window_vecm.std() if window_vecm.std() > 0 else 1
        z = (vecm_hat - mu) / sd

        # =============================
        # 5) BORROW COST for ALL open shorts
        # =============================
        for pos in position_short:
            px = x if pos.ticker == "X" else y
            cash -= pos.n_shares * px * BR_daily

        # ======================================================
        # 6) EXIT LOGIC — Mean Reversion (|z| < EXIT_TH)
        # ======================================================
        if (position_long or position_short) and abs(z) < EXIT_TH:

            # ---- Close LONGS ----
            for pos in position_long[:]:
                exit_px = x if pos.ticker == "X" else y
                pnl = (exit_px - pos.entry_price) * pos.n_shares
                cash += pos.n_shares * exit_px * (1 - COM)

                pos.exit_price = exit_px
                pos.exit_date = date
                pos.profit = pnl

                position_long.remove(pos)
                closed_positions.append(pos)

            # ---- Close SHORTS ----
            for pos in position_short[:]:
                exit_px = x if pos.ticker == "X" else y
                pnl = (pos.entry_price - exit_px) * pos.n_shares
                commission = exit_px * pos.n_shares * COM
                cash += pnl - commission

                pos.exit_price = exit_px
                pos.exit_date = date
                pos.profit = pnl - commission

                position_short.remove(pos)
                closed_positions.append(pos)

            exit_idx_list.append(idx)
            equity_curve.append(get_portfolio_value(cash, position_long, position_short, y, x))
            continue

        # ======================================================
        # 7) ENTRY LOGIC — VECM Mean Reversion Signal
        # ======================================================

        total_to_invest = cash * INVEST
        half = total_to_invest * 0.5

        # ---- LONG Y / SHORT X ----
        if z > theta and not position_long and not position_short:

            n_long = int(half // (y * (1 + COM)))
            n_short = int(n_long * abs(beta))
            cost_comm_short = n_short * x * COM
            cost_long = n_long * y * (1 + COM)

            if n_long > 0 and n_short > 0 and cash >= cost_long + cost_comm_short:

                # LONG Y
                cash -= cost_long
                buy += 1
                position_long.append(Position(
                    n_shares=n_long, ticker="Y", entry_price=y,
                    type_of_trade="LONG", entry_date=date
                ))

                # SHORT X
                cash -= cost_comm_short
                sell += 1
                position_short.append(Position(
                    n_shares=n_short, ticker="X", entry_price=x,
                    type_of_trade="SHORT", entry_date=date
                ))

                entry_idx_list.append(idx)
                equity_curve.append(get_portfolio_value(cash, position_long, position_short, y, x))
                continue

        # ---- SHORT Y / LONG X ----
        if z < -theta and not position_long and not position_short:

            n_short = int(half // (y * (1 + COM)))
            n_long = int(n_short * abs(beta))
            cost_comm_short = n_short * y * COM
            cost_long = n_long * x * (1 + COM)

            if n_long > 0 and n_short > 0 and cash >= cost_long + cost_comm_short:

                # LONG X
                cash -= cost_long
                buy += 1
                position_long.append(Position(
                    n_shares=n_long, ticker="X", entry_price=x,
                    type_of_trade="LONG", entry_date=date
                ))

                # SHORT Y
                cash -= cost_comm_short
                sell += 1
                position_short.append(Position(
                    n_shares=n_short, ticker="Y", entry_price=y,
                    type_of_trade="SHORT", entry_date=date
                ))

                entry_idx_list.append(idx)
                equity_curve.append(get_portfolio_value(cash, position_long, position_short, y, x))
                continue

        # HOLD
        hold += 1
        equity_curve.append(get_portfolio_value(cash, position_long, position_short, y, x))

    # ===== FINAL =====
    port_series = pd.Series(equity_curve, index=data.index[:len(equity_curve)])
    total_trades = len(closed_positions)
    wins = sum(1 for pos in closed_positions if pos.profit > 0)
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    return port_series, cash, win_rate, buy, sell, hold, total_trades, closed_positions
