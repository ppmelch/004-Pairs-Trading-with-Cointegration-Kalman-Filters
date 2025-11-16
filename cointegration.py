from libraries import *
from itertools import combinations
from classes import coint_config


# =====================================================
#   ROLLING CORRELATION
# =====================================================
def correlation(data: pd.DataFrame, window=coint_config.window):
    """
    Rolling correlation entre dos activos.
    """
    data = data.copy()
    corr = data.iloc[:, 0].rolling(window).corr(data.iloc[:, 1])
    return corr.rolling(window).mean()


# =====================================================
#   OLS + ADF (Engle-Granger)
# =====================================================
def OLS(data: pd.DataFrame):
    """
    Devuelve:
        - residuales del spread (y - beta*x)
        - p-value del ADF
        - residuo medio
    """
    data = data.copy().dropna()

    y = data.iloc[:, 0]
    x = sm.add_constant(data.iloc[:, 1])

    model = sm.OLS(y, x).fit()
    resid = model.resid
    mean_resid = resid.mean()

    adf_p = adfuller(resid, regression='c')[1]

    return resid, adf_p, mean_resid


# =====================================================
#   JOHANSEN TEST
# =====================================================
def johansen_test(data: pd.DataFrame,
                  det_order=coint_config.det_order,
                  k_ar_diff=coint_config.k_ar_diff):
    """
    Devuelve dict con:
        - eigenvector (primero)
        - valores críticos
        - estadístico de trazas
    """
    data = data.copy().dropna()
    res = coint_johansen(data, det_order, k_ar_diff)

    return {
        'eigenvectors': res.evec[:, 0],        # vector propio dominante
        'critical_values': res.cvt[:, 1],      # valores críticos 5%
        'trace_stat': res.lr1[0]               # estadístico trace
    }


# =====================================================
#   SELECCIÓN DE PARES
# =====================================================
def select_pairs(prices: pd.DataFrame,
                 corr_threshold: float = 0.7,
                 adf_alpha: float = 0.05):
    """
    Para cada pareja:
        - calcula correlación
        - corre Engle-Granger
        - corre Johansen
        - normaliza eigenvector
    """

    results = []
    corr_matrix = prices.corr()

    for a, b in combinations(prices.columns, 2):

        corr = corr_matrix.loc[a, b]
        if pd.isna(corr) or corr < corr_threshold:
            continue

        data_pair = prices[[a, b]].dropna()

        # ========== OLS + ADF ==========
        resid, adf_p, _ = OLS(data_pair)

        # ========== JOHANSEN ==========
        joh = johansen_test(data_pair)

        eig = joh['eigenvectors']
        beta1, beta2 = float(eig[0]), float(eig[1])

        # Normalización (beta2 = 1)
        beta1_norm = beta1 / beta2 if beta2 != 0 else np.nan
        beta2_norm = 1.0

        joh_trace = float(joh['trace_stat'])
        joh_crit = float(joh['critical_values'][0])
        johansen_ok = joh_trace > joh_crit

        results.append({
            'Asset1': a,
            'Asset2': b,
            'Correlation': float(corr),

            'ADF_pvalue': adf_p,
            'ADF_Cointegrated': adf_p < adf_alpha,

            'Johansen_stat': joh_trace,
            'Johansen_crit_95': joh_crit,
            'Johansen_Cointegrated': johansen_ok,

            'Eigenvector_1': beta1,
            'Eigenvector_2': beta2,
            'Beta1_norm': beta1_norm,
            'Beta2_norm': beta2_norm,

            'Johansen_strength': joh_trace - joh_crit
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # FILTRO estricto
    mask = (
        (df['Correlation'] >= corr_threshold) &
        (df['ADF_pvalue'] < adf_alpha) &
        (df['ADF_Cointegrated']) &
        (df['Johansen_Cointegrated'])
    )

    selected = df.loc[mask].copy()
    if selected.empty:
        return selected

    # Ranking final
    selected = selected.sort_values(
        by=['ADF_pvalue', 'Johansen_strength', 'Correlation'],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    return selected


# =====================================================
#   SELECCIONA DOS ACTIVOS
# =====================================================
def selected_pair(data: pd.DataFrame, asset1: str, asset2: str) -> pd.DataFrame:
    return data[[asset1, asset2]].dropna()
