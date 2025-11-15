from metrics import metrics, trade_stadistics
from classes import config
from backtesting import backtest
from visualization import plot_normalized_data, plot_spread, plot_dynamic_hedge_ratio, plot_single_split, plot_test_validation, plot_trade_returns, plot_spread_vs_vecm, plot_trading_signals , plot_portfolio_splits

def results(initial_capital, cash, port_value, win_rate, buy, sell, hold, total_trades, positions, total_borrow, total_comm):
    final = float(port_value.iloc[-1])
    profit = final - initial_capital
    print("\n" + "="*60)
    print(f"Capital Inicial:        ${initial_capital:,.2f}")
    print(f"Capital final:          ${cash:,.2f}")
    print(f"Valor del portafolio:   ${final:,.2f}")
    print(f"Profit:                 ${profit:,.2f}")
    print(f"Return %:               {(profit/initial_capital)*100:.2f}%")
    m = metrics(port_value)
    print("\n====================== MÉTRICAS ======================")
    for k,v in m.items(): print(f"{k:<20}: {v:.4f}")
    print("\n==================== TRADE STATISTICS ====================")
    ts = trade_stadistics(positions,buy,sell,hold,total_borrow,total_comm)
    for k,v in ts.items(): print(f"{k:<30}: {v}")
    print("="*60 + "\n")

def backtest_pair_splits(train, test, val):
    print("\n========== TRAIN ==========")
    (p_train,c_train,w_train,b_train,s_train,h_train,tt_train,pos_train,tb_train,tc_train) = backtest(train)
    results(config.capital,c_train,p_train,w_train,b_train,s_train,h_train,tt_train,pos_train,tb_train,tc_train)
    plot_normalized_data(train)
    plot_spread(train)
    plot_dynamic_hedge_ratio(train)
    plot_single_split(p_train,"Train Portfolio")
    plot_spread_vs_vecm(train.iloc[:,0] - train.iloc[:,1], train.iloc[:,0] - train.iloc[:,1])  # Placeholder for VECM series
    plot_trading_signals((train.iloc[:,0] - train.iloc[:,1]).rolling(window=config.TDays).mean(), [], [], [])  # Placeholder for indices

    print("\n========== TEST ==========")
    (p_test,c_test,w_test,b_test,s_test,h_test,tt_test,pos_test,tb_test,tc_test) = backtest(test)
    results(config.capital,c_test,p_test,w_test,b_test,s_test,h_test,tt_test,pos_test,tb_test,tc_test)
    plot_single_split(p_test,"Test Portfolio")

    print("\n======== VALIDATION =======")
    start_val = float(p_test.iloc[-1])
    (p_val,c_val,w_val,b_val,s_val,h_val,tt_val,pos_val,tb_val,tc_val) = backtest(val,initial_cash=start_val)
    results(start_val,c_val,p_val,w_val,b_val,s_val,h_val,tt_val,pos_val,tb_val,tc_val)
    plot_single_split(p_val,"Validation Portfolio")
    plot_test_validation(p_test,p_val)
    plot_trade_returns([p.profit for p in pos_val])
    print("\n=== BACKTEST COMPLETO ===\n")
    plot_portfolio_splits(p_train, p_test, p_val)