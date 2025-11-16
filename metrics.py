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

def metrics(series):
    return {
        "Sharpe Ratio": Metrics.sharpe(series),
        "Sortino Ratio": Metrics.sortino(series),
        "Maximum Drawdown": Metrics.max_drawdown(series),
        "Calmar Ratio": Metrics.calmar(series),
        "Win Rate": Metrics.win_rate(series),
    }

def trade_stadistics(positions,buy,sell,hold,total_borrow,total_comm):
    profits = [p.profit for p in positions]
    wins = [p for p in profits if p>0]
    losses = [p for p in profits if p<0]
    return {
        "# Trades": len(positions),
        "Operations": {"buy": buy, "sell": sell, "hold": hold},
        "Avg Win": np.mean(wins) if wins else 0,
        "Avg Loss": np.mean(losses) if losses else 0,
        "Profit": sum(profits),
        "Total Borrow Cost": total_borrow,
        "Total Comission Cost": total_comm,
    }