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

        resid, adf_p, _ = OLS(data_pair)

        joh = johansen_test(data_pair)
        joh_trace = float(joh['trace_stat'])
        joh_crit95_r0 = float(joh['critical_values'][0])
        johansen_coint = joh_trace > joh_crit95_r0

        results.append({
            'Asset1': a,
            'Asset2': b,
            'Correlation': float(corr),
            'ADF_pvalue': adf_p,
            'ADF_Cointegrated': adf_p < adf_alpha,
            'Johansen_stat': joh_trace,
            'Johansen_crit_95': joh_crit95_r0,
            'Johansen_Cointegrated': johansen_coint
        })

    df = pd.DataFrame(results)

    # RANKING 

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

    top5 = buenos.sort_values(
        by=['ADF_pvalue', 'Johansen_strength', 'Correlation'],
        ascending=[True, False, False]
    ).reset_index(drop=True).head(10)

    return top5


