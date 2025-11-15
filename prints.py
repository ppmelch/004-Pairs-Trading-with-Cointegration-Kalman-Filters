from libraries import *
from classes import config
from metrics import metrics , trade_stadistics
from backtesting import backtest
from visualization import plot_normalized_data, plot_single_split, plot_spread, plot_test_validation

def results(
    cash: float,
    port_value,
    win_rate: float,
    buy: int,
    sell: int,
    hold: int,
    total_trades: int,
    positions=None
) -> None:

    final_value = float(port_value.iloc[-1])
    profit = final_value - config.capital
    returns = (profit / config.capital) * 100

    print("="*60)
    print(f"Capital Inicial:       ${config.capital:,.2f}")
    print(f"Capital final:         ${cash:,.2f}")
    print(f"Valor del portafolio:  ${final_value:,.2f}")
    print(f"Profit:        ${profit:,.2f} ")
    print(f"Return % : {returns:.2f}%")


    # =============== PORTFOLIO METRICS =================
    try:
        m = metrics(port_value)
        print("\n========================== MÉTRICAS ==========================")
        for k, v in m.items():
            print(f"{k:<20}: {v:.4f}" if isinstance(
                v, float) else f"{k:<20}: {v}")
        print("===============================================================\n")
    except Exception as e:
        print(f"\n⚠️ No se pudieron calcular métricas: {e}")

    # =============== TRADE STATISTICS ==================
    if positions is not None and len(positions) > 0:
        try:
            ts = trade_stadistics(
                positions,
                buy=buy,
                sell=sell,
                hold=hold,
                anual_borrow=config.BR,
                TDays=config.TDays,
                COM=config.COM
            )
            print("======================= TRADE STATS ==========================")
            for k, v in ts.items():
                print(f"{k:<25}: {v}")
            print("===============================================================\n")
        except Exception as e:
            print(f"\n⚠️ No se pudieron calcular TradeStatistics: {e}")



def backtest_pair_splits(pair_train, pair_test, pair_val):

    print("\n========== TRAIN ==========\n")

    port_train, cash, win_rate, buy, sell, hold, total_trades, positions = backtest(pair_train)
    results(cash, port_train, win_rate, buy, sell, hold, total_trades, positions)
    plot_normalized_data(pair_train)
    plot_spread(pair_train)
    plot_single_split(port_train, "Train Portfolio")


    if pair_test is not None:
        print("\n========== TEST ==========\n")

        port_test, cash, win_rate, buy, sell, hold, total_trades, positions = backtest(pair_test)
        results(cash, port_test, win_rate, buy, sell, hold, total_trades, positions)
        plot_single_split(port_test, "Test Portfolio")


    if pair_val is not None:
        print("\n========== VALIDATION ==========\n")

        port_val, cash, win_rate, buy, sell, hold, total_trades, positions = backtest(pair_val)
        results(cash, port_val, win_rate, buy, sell, hold, total_trades, positions)
        plot_single_split(port_val, "Validation Portfolio")

        if pair_test is not None:
            plot_test_validation(port_test, port_val)

