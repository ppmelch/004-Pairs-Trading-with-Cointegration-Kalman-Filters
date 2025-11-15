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


class TradeStatistics:
    @staticmethod
    def trades_count(positions: list) -> int:
        """Cuenta el número total de trades realizados."""
        return len(positions)

    @staticmethod
    def buy_sell_hold_counts(buy: int, sell: int, hold: int) -> dict:
        """Devuelve un diccionario con los conteos de buy, sell y hold."""
        return {
            "buy": buy,
            "sell": sell,
            "hold": hold
        }

    @staticmethod
    def AVG_WIN(positions: list) -> float:
        """Calcula el promedio de ganancias por trade ganador."""
        wins = [pos.profit for pos in positions if pos.profit > 0]
        return float(np.mean(wins)) if wins else 0.0

    @staticmethod
    def AVG_LOSS(positions: list) -> float:
        """Calcula el promedio de pérdidas por trade perdedor."""
        losses = [pos.profit for pos in positions if pos.profit < 0]
        return float(np.mean(losses)) if losses else 0.0

    @staticmethod
    def PROFIT_FACTOR(positions: list) -> float:
        """
        Calcula el Profit Factor: suma de ganancias / suma de pérdidas (en valor absoluto).
        Si no hay pérdidas, devuelve inf (matemáticamente correcto).
        """
        total_wins = sum(pos.profit for pos in positions if pos.profit > 0)
        total_losses = abs(sum(pos.profit for pos in positions if pos.profit < 0))
        if total_losses == 0:
            # Podrías cambiar esto a 0.0 o np.nan si no quieres 'inf'
            return float('inf') if total_wins > 0 else 0.0
        return total_wins / total_losses

    @staticmethod
    def BORROW_COST(total_positions: list, anual_borrow: float, TDays: int) -> float:
        """
        Calcula el costo total de borrow para todas las posiciones.

        Aproximación académica:
        - Usa el notional al precio de entrada.
        - Usa days_held basado en entry_date / exit_date.
        - daily_rate = anual_borrow / TDays (coherente con el backtest).
        """
        if not total_positions:
            return 0.0

        total_cost = 0.0
        daily_rate = anual_borrow / TDays

        for pos in total_positions:
            if pos.entry_date is None or pos.exit_date is None:
                continue

            days_held = (pos.exit_date - pos.entry_date).days
            if days_held <= 0:
                continue

            notional = pos.n_shares * pos.entry_price
            cost = notional * daily_rate * days_held
            total_cost += cost

        return float(total_cost)

    @staticmethod
    def COMISSION_COST(total_positions: list, COM: float) -> float:
        """
        Calcula el costo total de comisiones para todas las posiciones.

        Entrada: n_shares * entry_price * COM
        Salida: n_shares * exit_price * COM

        Nota:
        - Esto es puramente informativo si en el backtest ya restaste comisiones
          del cash y pos.profit es neto de comisiones.
        """
        if not total_positions:
            return 0.0

        total_cost = 0.0
        for pos in total_positions:
            if pos.exit_price is None:
                continue
            entry_cost = pos.n_shares * pos.entry_price * COM
            exit_cost = pos.n_shares * pos.exit_price * COM
            total_cost += (entry_cost + exit_cost)

        return float(total_cost)


def metrics(port_value: pd.Series) -> dict:
    return {
        "Sharpe Ratio": Metrics.sharpe(port_value),
        "Sortino Ratio": Metrics.sortino(port_value),
        "Maximum Drawdown": Metrics.max_drawdown(port_value),
        "Calmar Ratio": Metrics.calmar(port_value),
        "Win Rate": Metrics.win_rate(port_value)
    }


def trade_stadistics(positions: list, buy: int, sell: int, hold: int,
                     anual_borrow: float, TDays: int, COM: float) -> dict:

    return {
        "# Trades": TradeStatistics.trades_count(positions),
        "Operations": TradeStatistics.buy_sell_hold_counts(buy, sell, hold),
        "Avg Win": TradeStatistics.AVG_WIN(positions),
        "Avg Loss": TradeStatistics.AVG_LOSS(positions),
        "Profit": TradeStatistics.PROFIT_FACTOR(positions),
        "Borrow Cost": TradeStatistics.BORROW_COST(positions, anual_borrow, TDays),
        "Comission Cost": TradeStatistics.COMISSION_COST(positions, COM)
    }