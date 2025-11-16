from metrics import metrics, trade_stadistics
from libraries import pd, np
from classes import config
from backtesting import backtest
from visualization import (
    plot_normalized_data, plot_portfolio_splits, plot_spread, plot_dynamic_hedge_ratio,
    plot_single_split, plot_test_validation, plot_trade_returns,
    plot_spread_vs_vecm, plot_kalman_eigenvectors, compute_kalman_vecm
)


def results(initial_capital, cash, port_value, win_rate, buy, sell, hold,
            total_trades, positions, total_borrow, total_comm):
    """
    Print ordered results for a backtest split:
    - Capital evolution
    - Performance metrics
    - Trading statistics
    """
    final = float(port_value.iloc[-1])
    profit = final - initial_capital

    print("\n" + "=" * 60)
    print(f"Capital Inicial:        ${initial_capital:,.2f}")
    print(f"Capital Final (Cash):   ${cash:,.2f}")
    print(f"Valor Portafolio:       ${final:,.2f}")
    print(f"Profit:                 ${profit:,.2f}")
    print(f"Return %:               {(profit / initial_capital) * 100:.2f}%")

    # Performance metrics
    m = metrics(port_value)
    print("\n====================== MÉTRICAS ======================")
    for k, v in m.items():
        print(f"{k:<20}: {v:.4f}")

    # Trading statistics
    ts = trade_stadistics(positions, buy, sell, hold, total_borrow, total_comm)
    print("\n==================== TRADE STATISTICS ====================")
    for k, v in ts.items():
        print(f"{k:<30}: {v}")

    print("=" * 60 + "\n")


def backtest_pair_splits(train, test, val):
    """
    Complete backtest pipeline:
    - Train / Test / Validation
    - Kalman β(t)
    - Real spread = y - βx
    - Kalman-smoothed VECM-like signal
    - Johansen eigenvector
    - Portfolio curves
    """

    # ===================== TRAIN =====================
    print("\n========== TRAIN ==========")
    (p_train, c_train, w_train, b_train, s_train,
     h_train, tt_train, pos_train, tb_train, tc_train) = backtest(train)

    results(config.capital, c_train, p_train, w_train, b_train,
            s_train, h_train, tt_train, pos_train, tb_train, tc_train)

    # Normalized price plot
    plot_normalized_data(train)

    # Hedge ratio β(t)
    beta_series = plot_dynamic_hedge_ratio(train)

    spread = train.iloc[:, 0] - train.iloc[:, 1]
    vecm_signal = compute_kalman_vecm(spread)
    plot_spread_vs_vecm(spread, vecm_signal)


    # Johansen eigenvector for GOOGL–HD
    eig = np.array([0.314046, -0.088092])
    plot_kalman_eigenvectors(train, eig)

    plot_single_split(p_train, "Train Portfolio")

    # Kalman-smoothed spread (VECM-like)
    vecm_signal = compute_kalman_vecm(spread)
    plot_spread_vs_vecm(spread, vecm_signal)

    # ===================== TEST =====================
    print("\n========== TEST ==========")
    (p_test, c_test, w_test, b_test, s_test,
     h_test, tt_test, pos_test, tb_test, tc_test) = backtest(test)

    results(config.capital, c_test, p_test, w_test, b_test,
            s_test, h_test, tt_test, pos_test, tb_test, tc_test)

    plot_single_split(p_test, "Test Portfolio")

    # ===================== VALIDATION =====================
    print("\n======== VALIDATION =======")
    start_val = float(p_test.iloc[-1])

    (p_val, c_val, w_val, b_val, s_val,
     h_val, tt_val, pos_val, tb_val, tc_val) = backtest(val, initial_cash=start_val)

    results(start_val, c_val, p_val, w_val, b_val,
            s_val, h_val, tt_val, pos_val, tb_val, tc_val)

    plot_single_split(p_val, "Validation Portfolio")
    plot_test_validation(p_test, p_val)
    plot_trade_returns([p.profit for p in pos_val])

    # ===================== FULL BACKTEST =====================
    print("\n=== BACKTEST COMPLETO ===\n")
    plot_portfolio_splits(p_train, p_test, p_val)
