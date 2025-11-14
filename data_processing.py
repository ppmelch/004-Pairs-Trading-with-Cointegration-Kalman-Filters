from libraries import *

def clean_data(activos, intervalo: str = "15y") -> pd.DataFrame:
    # Asegurar lista de tickers
    if isinstance(activos, str):
        tickers = [activos.strip().upper()]
    else:
        tickers = [t.strip().upper() for t in activos if isinstance(t, str) and t.strip()]

    # Parsear intervalo tipo '15y'
    m = re.match(r"^\s*(\d+)\s*([dwmy])\s*$", intervalo.lower())
    if not m:
        raise ValueError("intervalo debe tener formato '<int><unidad>' con unidad en {d,w,m,y}")
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
            auto_adjust=False  # evita FutureWarning
        )

        if df is None or df.empty:
            print(f"NO HAY DATOS EN ESTE INTERVALO: {t}")
            continue

        # Aplanar multiíndice si aplica
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Fallback para 'Close'
        if "Close" not in df.columns or df["Close"].isna().all():
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            elif all(c in df.columns for c in ["Open", "High", "Low"]):
                df["Close"] = df[["Open", "High", "Low"]].mean(axis=1)

        # Mantener solo columna Close
        df = df[["Close"]].dropna()

        if df.empty:
            print(f"NO HAY DATOS EN ESTE INTERVALO: {t}")
            continue

        df = df.rename(columns={"Close": t})
        datos_validos[t] = df

    if not datos_validos:
        print("NO HAY DATOS DISPONIBLES EN ESTE INTERVALO PARA NINGÚN TICKER.")
        return pd.DataFrame()

    # Combinar todos los cierres y eliminar filas con NaN
    combined = pd.concat(datos_validos.values(), axis=1)
    combined.index.name = "Date"
    combined.sort_index(inplace=True)
    combined = combined.dropna(how="any")  

    return combined


def dataset_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered dataset into train, test, and validation partitions.

    The split uses fixed proportions: 60% train, 20% test, 20% validation.

    Parameters
    ----------
    data : pd.DataFrame
        Full, time-ordered dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, test, validation) partitions, preserving order.
    """
    train_size = int(len(data) * 0.6)
    test_size = int(len(data) * 0.2)

    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    validation_data = data[train_size + test_size:]

    return train_data, test_data, validation_data