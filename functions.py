from libraries import *

def get_portfolio_value(cash, long_ops, short_ops, price_y, price_x):
    """
    price_y = p1 (asset Y)
    price_x = p2 (asset X)
    """
    
    value = cash

    # LONG positions (suman valor)
    for pos in long_ops:
        if pos.ticker == "Y":
            value += pos.n_shares * price_y
        elif pos.ticker == "X":
            value += pos.n_shares * price_x

    # SHORT positions (restan valor del precio actual)
    for pos in short_ops:
        if pos.ticker == "Y":
            value -= pos.n_shares * price_y
        elif pos.ticker == "X":
            value -= pos.n_shares * price_x

    return value

