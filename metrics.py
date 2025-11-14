from libraries import *

class Metrics:
    """Calcula métricas financieras sobre una serie de valores de portafolio."""

    @staticmethod
    def sharpe(data: pd.Series) -> float:
        """Calcula el ratio de Sharpe anualizado."""
        if data is None or data.empty:
            return 0.0
        returns = data.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        mean_ret = returns.mean()
        std_ret = returns.std()
        annual_mean = mean_ret * np.sqrt(252)  # anualización diaria
        annual_std = std_ret * np.sqrt(252)
        return annual_mean / annual_std

    @staticmethod
    def sortino(data: pd.Series) -> float:
        """Calcula el ratio de Sortino anualizado."""
        if data is None or data.empty:
            return 0.0
        returns = data.pct_change().dropna()
        downside = returns[returns < 0]
        if downside.std() == 0:
            return 0.0
        mean_ret = returns.mean()
        annual_mean = mean_ret * np.sqrt(252)
        annual_downside = downside.std() * np.sqrt(252)
        return annual_mean / annual_downside

    @staticmethod
    def max_drawdown(data: pd.Series) -> float:
        """Calcula el máximo drawdown (como valor positivo)."""
        if data is None or data.empty:
            return 0.0
        rolling_max = data.cummax()
        drawdown = (data - rolling_max) / rolling_max
        return abs(drawdown.min())

    @staticmethod
    def calmar(data: pd.Series) -> float:
        """Calcula el ratio de Calmar: retorno anual / drawdown máximo."""
        if data is None or data.empty:
            return 0.0
        returns = data.pct_change().dropna()
        annual_return = (1 + returns.mean()) ** 252 - 1
        mdd = Metrics.max_drawdown(data)
        return annual_return / mdd if mdd > 0 else 0.0

    @staticmethod
    def win_rate(data: pd.Series) -> float:
        """Porcentaje de retornos positivos."""
        if data is None or data.empty:
            return 0.0
        returns = data.pct_change().dropna()
        return (returns > 0).mean()

def metrics(port_value: pd.Series) -> dict:
    """
    Calcula métricas clave de performance del portafolio.
    """
    return {
        "Sharpe Ratio": Metrics.sharpe(port_value),
        "Sortino Ratio": Metrics.sortino(port_value),
        "Maximum Drawdown": Metrics.max_drawdown(port_value),
        "Calmar Ratio": Metrics.calmar(port_value),
        "Win Rate": Metrics.win_rate(port_value)
    }