from libraries import *


@dataclass
class config:
    """
    Global configuration parameters for capital usage, commissions,
    borrow rates, and signal thresholds used in the trading engine.
    """
    capital: float = 1_000_000
    COM: float = 0.00125
    INVEST: float = 0.80
    BR: float = 0.25 / 100
    ENTRY_Z: float = 1.0
    EXIT_Z: float = 0.5
    TDays: int = 20


@dataclass
class coint_config:
    """
    Configuration parameters for cointegration analysis including rolling window,
    deterministic trend order, VAR lag selection, minimum correlation threshold,
    and ADF significance cutoff.
    """
    window: int = 252
    det_order: int = 0
    k_ar_diff: int = 1
    threshold: float = 0.7
    adf_alpha: float = 0.05


@dataclass
class Position:
    """
    Representation of a long or short position in the trading system.

    Attributes
    ----------
    n_shares : float
        Number of shares in the position.
    ticker : str
        Asset identifier ("Y" or "X").
    entry_price : float
        Price at which the position is opened.
    exit_price : float
        Price at which the position is closed.
    type_of_trade : str
        Direction of trade ("LONG" or "SHORT").
    entry_date : any
        Timestamp for entry.
    exit_date : any
        Timestamp for exit.
    profit : float
        Realized profit or loss.
    """
    n_shares: float
    ticker: str = None
    entry_price: float = None
    exit_price: float = None
    type_of_trade: str = None
    entry_date: any = None
    exit_date: any = None
    profit: float = 0.0
