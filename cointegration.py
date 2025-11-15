from libraries import *
from itertools import combinations
from classes import coint_config

def correlation(data, window=coint_config.window):

    data = data.copy()

    corr = data.iloc[:, 0].rolling(window=window).corr(data.iloc[:, 1])

    mean = corr.rolling(window=window).mean()

    return mean

def OLS(data: pd.DataFrame):

    data = data.copy().dropna()

    y = data.iloc[:, 0]
    x = sm.add_constant(data.iloc[:, 1])

    model = sm.OLS(y, x).fit()
    resid = model.resid
    mean_resid = resid.mean()  

    adf_pvalue = adfuller(resid, regression='c')[1]

    return resid, adf_pvalue , mean_resid


def johansen_test(data: pd.DataFrame , det_order=coint_config.det_order, k_ar_diff=coint_config.k_ar_diff):

    data = data.copy().dropna()

    result = coint_johansen(data, det_order, k_ar_diff)

    return {
        'eigenvectors': result.evec[:, 0],
        'critical_values': result.cvt[:, 1],  
        'trace_stat' : result.lr1[0],
    }


def select_pairs(prices: pd.DataFrame, corr_threshold: float = 0.7, adf_alpha: float = 0.05):
    results = []
    corr_matrix = prices.corr()

    for a, b in combinations(prices.columns, 2):
        corr = corr_matrix.loc[a, b]
        if pd.isna(corr) or corr < corr_threshold:
            continue

        data_pair = prices[[a, b]].dropna()

        # ======================
        # 1) OLS + ADF TEST
        # ======================
        resid, adf_p, _ = OLS(data_pair)

        # ======================
        # 2) JOHANSEN TEST
        # ======================
        joh = johansen_test(data_pair)

        eigenvec = joh['eigenvectors']
        beta1 = float(eigenvec[0])
        beta2 = float(eigenvec[1])

        # Normalización (beta2 = 1)
        beta1_norm = beta1 / beta2
        beta2_norm = 1.0

        joh_trace = float(joh['trace_stat'])
        joh_crit = float(joh['critical_values'][0])
        johansen_coint = joh_trace > joh_crit

        results.append({
            'Asset1': a,
            'Asset2': b,
            'Correlation': float(corr),

            'ADF_pvalue': adf_p,
            'ADF_Cointegrated': adf_p < adf_alpha,

            'Johansen_stat': joh_trace,
            'Johansen_crit_95': joh_crit,
            'Johansen_Cointegrated': johansen_coint,

            'Eigenvector_1': beta1,
            'Eigenvector_2': beta2,
            'Beta1_norm': beta1_norm,
            'Beta2_norm': beta2_norm
        })

    df = pd.DataFrame(results)

    # ======================
    #       FILTRO
    # ======================
    mask = (
        (df['ADF_pvalue'] < adf_alpha) &
        (df['Correlation'] >= corr_threshold) &
        (df['ADF_Cointegrated']) &
        (df['Johansen_Cointegrated']) &
        (df['Johansen_stat'] > df['Johansen_crit_95'])
    )

    buenos = df.loc[mask].copy()
    if buenos.empty:
        return pd.DataFrame()

    buenos['Johansen_strength'] = buenos['Johansen_stat'] - buenos['Johansen_crit_95']

    # ======================
    #    ORDEN DE COLUMNAS
    # ======================
    columnas_ordenadas = [
        'Asset1', 'Asset2',

        'Correlation',

        'ADF_pvalue', 'ADF_Cointegrated',

        'Johansen_stat', 'Johansen_crit_95',
        'Johansen_Cointegrated', 'Johansen_strength',

        'Eigenvector_1', 'Eigenvector_2',
        'Beta1_norm', 'Beta2_norm'
    ]

    buenos = buenos[columnas_ordenadas]

    # ======================
    #       RANKING
    # ======================
    top5 = buenos.sort_values(
        by=['ADF_pvalue', 'Johansen_strength', 'Correlation'],
        ascending=[True, False, False]
    ).reset_index(drop=True).head()

    return top5


def selected_pair (data: pd.DataFrame, asset1: str, asset2: str) -> pd.DataFrame:
    pair_data = data[[asset1, asset2]].dropna()
    return pair_data