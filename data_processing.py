from libraries import *


def clean_data(activos, intervalo: str = "15y") -> pd.DataFrame:
    """
    Download and preprocess daily closing prices for one or multiple tickers.

    Parameters
    ----------
    activos : list or str
        Tickers to fetch.
    intervalo : str
        Time range in compact notation (e.g., "10y", "6m").

    Returns
    -------
    pd.DataFrame
        Cleaned and aligned closing price matrix.
    """
    if isinstance(activos, str):
        tickers = [activos.strip().upper()]
    else:
        tickers = [t.strip().upper()
                   for t in activos if isinstance(t, str) and t.strip()]

    m = re.match(r"^\s*(\d+)\s*([dwmy])\s*$", intervalo.lower())
    if not m:
        raise ValueError(
            "Interval must follow the format '<int><unit>' with units in {d,w,m,y}")

    n, u = m.groups()
    delta = {"d": "days", "w": "weeks", "m": "months", "y": "years"}[u]
    start = dt.date.today() - relativedelta(**{delta: int(n)})
    end = dt.date.today() + dt.timedelta(days=1)

    datos_validos = {}
    for t in tickers:
        df = yf.download(
            t,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=False
        )
        if df is None or df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns or df["Close"].isna().all():
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            elif all(c in df.columns for c in ["Open", "High", "Low"]):
                df["Close"] = df[["Open", "High", "Low"]].mean(axis=1)

        df = df[["Close"]].dropna()
        if not df.empty:
            df = df.rename(columns={"Close": t})
            datos_validos[t] = df

    if not datos_validos:
        return pd.DataFrame()

    combined = pd.concat(datos_validos.values(), axis=1)
    combined.index.name = "Date"
    combined.sort_index(inplace=True)
    return combined.dropna(how="any")


def dataset_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into train, test, and validation partitions maintaining
    chronological order with fixed proportions 60/20/20.

    Parameters
    ----------
    data : pd.DataFrame
        Full price matrix.

    Returns
    -------
    tuple
        (train, test, validation)
    """
    train_size = int(len(data) * 0.6)
    test_size = int(len(data) * 0.2)

    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    val_data = data[train_size + test_size:]

    return train_data, test_data, val_data
