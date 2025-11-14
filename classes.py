from libraries import *

@dataclass 
class config:
    capital : float = 1_000_000  # Initial Capital
    COM : float = 0.00125  # Commission per trade
    INVEST : float = 0.80 
    BR : float = 0.25 / 100 # Borrowing Rate (annualized)
    theta : float = 0.7  # Kalman smoothing parameter
    TDays : int = 252  # Trading days in a year

@dataclass
class coint_config:
    window = 252
    det_order: int = 0
    k_ar_diff: int = 1
    corr_threshold: float = 0.7
    adf_alpha : float = 0.05

@dataclass
class Position:
    n_shares: float
    ticker: str = None
    entry_price: float = None
    exit_price: float = None
    type_of_trade : str = None  # 'LONG' or 'SHORT'







