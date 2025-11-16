from libraries import *

@dataclass 
class config:
    # ===== Capital y Costos =====
    capital : float = 1_000_000                # Capital inicial
    COM : float = 0.00125                      # Comisión (0.125%)
    INVEST : float = 0.80                      # 80% del capital disponible
    BR : float = 0.25 / 100                    # Tasa anual para short (0.25%)
    
    # ===== Parámetros de señales =====
    ENTRY_Z : float = 1.0                       # Entrada cuando |z| > 1
    EXIT_Z  : float = 0.5                      # Salida cuando |z| < 0.5

    # ===== Rolling windows =====
    TDays : int = 20                            # Rolling para Johansen y Z-score

@dataclass
class coint_config:
    window = 252
    det_order: int = 0
    k_ar_diff: int = 1
    threshold: float = 0.7      # Correlación mínima
    adf_alpha : float = 0.05    # P-value máximo para considerar cointegración

@dataclass
class Position:
    n_shares: float
    ticker: str = None
    entry_price: float = None
    exit_price: float = None
    type_of_trade: str = None       # ‘LONG’ o ‘SHORT’
    entry_date: any = None
    exit_date: any = None
    profit: float = 0.0