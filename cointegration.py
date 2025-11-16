from libraries import *
from classes import coint_config


def correlation(data: pd.DataFrame, window=coint_config.window):
    """
    Compute smoothed rolling correlation between two price series.

    Parameters
    ----------
    data : pd.DataFrame
        Two-column price dataset.
    window : int
        Window size for rolling correlation.

    Returns
    -------
    pd.Series
        Smoothed rolling correlation values.
    """
    data = data.copy()
    corr = data.iloc[:, 0].rolling(window).corr(data.iloc[:, 1])
    return corr.rolling(window).mean()


def OLS(data: pd.DataFrame):
    """
    Perform Engle–Granger OLS regression to estimate the hedge ratio,
    compute residuals of the spread, and evaluate stationarity via ADF.

    Parameters
    ----------
    data : pd.DataFrame
        Two-asset price series.

    Returns
    -------
    tuple
        (residuals, adf_pvalue, residual_mean)
    """
    data = data.copy().dropna()

    y = data.iloc[:, 0]
    x = sm.add_constant(data.iloc[:, 1])

    model = sm.OLS(y, x).fit()
    resid = model.resid
    mean_resid = resid.mean()

    adf_p = adfuller(resid, regression='c')[1]

    return resid, adf_p, mean_resid


def johansen_test(data: pd.DataFrame,
                  det_order=coint_config.det_order,
                  k_ar_diff=coint_config.k_ar_diff):
    """
    Apply Johansen cointegration test and extract the dominant eigenvector,
    trace statistic, and corresponding critical value.

    Parameters
    ----------
    data : pd.DataFrame
        Two-asset price data.
    det_order : int
        Deterministic trend specification.
    k_ar_diff : int
        Johansen VAR lag order.

    Returns
    -------
    dict
        {
            'eigenvectors': np.ndarray,
            'critical_values': np.ndarray,
            'trace_stat': float
        }
    """
    data = data.copy().dropna()
    res = coint_johansen(data, det_order, k_ar_diff)

    return {
        'eigenvectors': res.evec[:, 0],
        'critical_values': res.cvt[:, 1],
        'trace_stat': res.lr1[0]
    }


def select_pairs(prices: pd.DataFrame,
                 corr_threshold: float = 0.7,
                 adf_alpha: float = 0.05):
    """
    Evaluate all asset pairs and select those satisfying correlation,
    Engle–Granger, and Johansen cointegration requirements.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price matrix.
    corr_threshold : float
        Minimum acceptable correlation.
    adf_alpha : float
        Maximum ADF p-value allowed.

    Returns
    -------
    pd.DataFrame
        Ranked table of cointegrated pairs and statistics.
    """
    results = []
    corr_matrix = prices.corr()

    for a, b in combinations(prices.columns, 2):

        corr = corr_matrix.loc[a, b]
        if pd.isna(corr) or corr < corr_threshold:
            continue

        data_pair = prices[[a, b]].dropna()

        resid, adf_p, _ = OLS(data_pair)
        joh = johansen_test(data_pair)

        eig = joh['eigenvectors']
        beta1, beta2 = float(eig[0]), float(eig[1])

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

    mask = (
        (df['Correlation'] >= corr_threshold) &
        (df['ADF_pvalue'] < adf_alpha) &
        (df['ADF_Cointegrated']) &
        (df['Johansen_Cointegrated'])
    )

    selected = df.loc[mask].copy()
    if selected.empty:
        return selected

    selected = selected.sort_values(
        by=['ADF_pvalue', 'Johansen_strength', 'Correlation'],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    return selected


def selected_pair(data: pd.DataFrame, asset1: str, asset2: str) -> pd.DataFrame:
    """
    Extract a clean two-asset price series for a chosen pair.

    Parameters
    ----------
    data : pd.DataFrame
        Full price matrix.
    asset1 : str
        First asset ticker.
    asset2 : str
        Second asset ticker.

    Returns
    -------
    pd.DataFrame
        Two-column price series for the pair.
    """
    return data[[asset1, asset2]].dropna()
