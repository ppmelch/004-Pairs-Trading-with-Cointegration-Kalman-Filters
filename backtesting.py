from libraries import *
from kalman import KalmanFilter
from classes import Position, config
from functions import get_portfolio_value

def backtest(data: pd.DataFrame) -> pd.DataFrame:

    data = data.copy()

    # CONFIG
    TDays = config.TDays
    cash = config.capital
    COM = config.COM
    INVEST = config.INVEST
    anual_borrow = config.BR
    theta = config.theta
    daily_borrow = anual_borrow / TDays  

    # Kalman filters
    kalman_1 = KalmanFilter(n=2)   # hedge ratio
    kalman_2 = KalmanFilter(n=1)   # VECM smoothing

    vecms_hat = []
    active_long = []
    active_short = []
    all_positions = []
    equity_curve = []

    buy = sell = hold = 0

    for i, row in data.iterrows():

        p1 = row[0]   # Y
        p2 = row[1]   # X

        # Borrow cost diario
        for pos in active_short:
            price_now = p2 if pos.ticker == "X" else p1
            notional = pos.n_shares * price_now
            cash -= notional * daily_borrow

        # ==================== KALMAN 1 (hedge ratio)
        x1_t = np.array([1, p2])
        y1_t = p1

        w_pred_1, P_pred_1 = kalman_1.predict()
        kalman_1.update(x1_t, y1_t, w_pred_1, P_pred_1)

        hr = kalman_1.w_t[1]

        # ==================== KALMAN 2 (VECM smoothing)
        if i < TDays:
            equity_curve.append(get_portfolio_value(cash, active_long, active_short, p1, p2))
            continue

        window_data = data.iloc[i - TDays:i]
        eig = coint_johansen(window_data, det_order=0, k_ar_diff=1)

        e1, e2 = eig.evec[0, 0], eig.evec[1, 0]
        VECM = e1 * p1 + e2 * p2

        x2_t = np.array([1])
        y2_t = VECM

        w_pred_2, P_pred_2 = kalman_2.predict()
        kalman_2.update(x2_t, y2_t, w_pred_2, P_pred_2)

        VECM_hat = kalman_2.w_t[0]
        vecms_hat.append(VECM_hat)

        if len(vecms_hat) < TDays:
            equity_curve.append(get_portfolio_value(cash, active_long, active_short, p1, p2))
            continue

        # Señal normalizada
        sample = np.array(vecms_hat[-TDays:])
        VECM_norm = (sample[-1] - sample.mean()) / sample.std()

        # ==================== ENTRADA SHORT SPREAD
        if VECM_norm < -theta and not active_long and not active_short:

            total_to_invest = cash * INVEST
            side_cash = total_to_invest * 0.5      # 40% por activo

            # SHORT Y
            n_shares_short = int(side_cash // (p1 * (1 + COM)))

            if n_shares_short > 0:
                cash -= p1 * n_shares_short * (1 + COM)
                sell += 1
                pos = Position(
                    n_shares=n_shares_short,
                    ticker="Y",
                    entry_price=p1,
                    type_of_trade="SHORT"
                )
                active_short.append(pos)
                all_positions.append(pos)

            # LONG X
            n_shares_long = int(abs(n_shares_short * hr))
            cost_long = p2 * n_shares_long * (1 + COM)

            if cash > cost_long:
                cash -= cost_long
                buy += 1
                pos = Position(
                    n_shares=n_shares_long,
                    ticker="X",
                    entry_price=p2,
                    type_of_trade="LONG"
                )
                active_long.append(pos)
                all_positions.append(pos)
            else:
                hold += 1

            equity_curve.append(get_portfolio_value(cash, active_long, active_short, p1, p2))

        # ==================== ENTRADA LONG SPREAD
        elif VECM_norm > theta and not active_long and not active_short:

            total_to_invest = cash * INVEST
            side_cash = total_to_invest * 0.5

            # LONG Y
            n_shares_long_Y = int(side_cash // (p1 * (1 + COM)))

            if n_shares_long_Y > 0:
                cash -= p1 * n_shares_long_Y * (1 + COM)
                buy += 1
                pos = Position(
                    n_shares=n_shares_long_Y,
                    ticker="Y",
                    entry_price=p1,
                    type_of_trade="LONG"
                )
                active_long.append(pos)
                all_positions.append(pos)

            # SHORT X
            n_shares_short_X = int(abs(n_shares_long_Y * hr))
            cost_short = p2 * n_shares_short_X * (1 + COM)

            if cash > cost_short:
                cash -= cost_short
                sell += 1
                pos = Position(
                    n_shares=n_shares_short_X,
                    ticker="X",
                    entry_price=p2,
                    type_of_trade="SHORT"
                )
                active_short.append(pos)
                all_positions.append(pos)
            else:
                hold += 1

            equity_curve.append(get_portfolio_value(cash, active_long, active_short, p1, p2))

        # ==================== CIERRE DE POSICIONES
        if (active_long or active_short) and abs(VECM_norm) <= 0.5:

            for pos in active_long[:]:
                exit_price = p2 if pos.ticker == "X" else p1
                cash += pos.n_shares * exit_price * (1 - COM)
                pos.exit_price = exit_price
                active_long.remove(pos)

            for pos in active_short[:]:
                exit_price = p2 if pos.ticker == "X" else p1
                profit = (pos.entry_price - exit_price) * pos.n_shares
                cash += profit
                cash -= pos.n_shares * exit_price * COM
                pos.exit_price = exit_price
                active_short.remove(pos)

            equity_curve.append(get_portfolio_value(cash, active_long, active_short, p1, p2))

    return pd.DataFrame({
        "cash": [cash],
        "positions": [len(all_positions)],
        "buy": [buy],
        "sell": [sell],
        "hold": [hold],
        "equity_curve": [equity_curve]
    })
