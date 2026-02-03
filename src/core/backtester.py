"""
Motor de Backtest Institucional
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, date
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from config.settings import (
    BacktestConfig,
    TransactionCostConfig,
    SlippageModel,
    RebalanceFrequency
)
from src.core.portfolio import Portfolio, PositionSide, OrderType
from src.core.risk_manager import RiskManager, RiskMetrics


logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Representa um sinal de trading"""
    symbol: str
    weight: float  # Peso alvo no portfolio (-1 a 1)
    score: float  # Score da estrategia
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> PositionSide:
        return PositionSide.LONG if self.weight > 0 else PositionSide.SHORT


@dataclass
class BacktestResult:
    """Resultado do backtest"""
    # Metricas de retorno
    total_return: float
    cagr: float
    annualized_volatility: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float

    # Tail risk
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float

    # Attribution (se benchmark disponivel)
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float

    # Trading metrics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_holding_period: float
    turnover: float

    # Custos
    total_commission: float
    total_slippage: float
    total_costs: float

    # Series
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    positions_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Risk
    risk_metrics: Optional[RiskMetrics] = None

    def to_dict(self) -> Dict:
        return {
            'total_return': self.total_return,
            'cagr': self.cagr,
            'annualized_volatility': self.annualized_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'alpha': self.alpha,
            'beta': self.beta,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'turnover': self.turnover,
            'total_costs': self.total_costs
        }


class TransactionCostModel:
    """Modelo de custos de transacao"""

    def __init__(self, config: TransactionCostConfig):
        self.config = config

    def calculate_commission(self, quantity: float, price: float) -> float:
        """Calcula comissao"""
        trade_value = quantity * price
        commission = trade_value * (self.config.commission_bps / 10000)
        return max(commission, self.config.min_commission)

    def calculate_slippage(
        self,
        quantity: float,
        price: float,
        avg_volume: Optional[float] = None
    ) -> float:
        """Calcula slippage baseado no modelo"""
        trade_value = quantity * price

        if self.config.slippage_model == SlippageModel.FIXED:
            return trade_value * (self.config.slippage_bps / 10000)

        elif self.config.slippage_model == SlippageModel.PROPORTIONAL:
            return trade_value * (self.config.slippage_bps / 10000)

        elif self.config.slippage_model == SlippageModel.SQRT:
            if avg_volume and avg_volume > 0:
                participation = quantity / avg_volume
                impact = self.config.market_impact_factor * np.sqrt(participation)
                return trade_value * impact
            else:
                return trade_value * (self.config.slippage_bps / 10000)

        return 0.0

    def calculate_total_cost(
        self,
        quantity: float,
        price: float,
        avg_volume: Optional[float] = None
    ) -> Tuple[float, float]:
        """Retorna (commission, slippage)"""
        commission = self.calculate_commission(quantity, price)
        slippage = self.calculate_slippage(quantity, price, avg_volume)
        return commission, slippage


class Backtester:
    """Motor de Backtest Institucional"""

    def __init__(
        self,
        config: BacktestConfig,
        strategy: 'BaseStrategy',
        data_provider: 'DataProvider'
    ):
        self.config = config
        self.strategy = strategy
        self.data_provider = data_provider

        self.portfolio = Portfolio(
            initial_capital=config.initial_capital,
            currency=config.currency
        )
        self.risk_manager = RiskManager(config.risk_config)
        self.cost_model = TransactionCostModel(config.transaction_costs)

        self._signals_history: List[List[Signal]] = []
        self._rebalance_dates: List[datetime] = []
        self._benchmark_returns: Optional[pd.Series] = None

    def _should_rebalance(self, current_date: datetime, last_rebalance: Optional[datetime]) -> bool:
        """Determina se deve rebalancear"""
        if last_rebalance is None:
            return True

        freq = self.config.rebalance_frequency

        if freq == RebalanceFrequency.DAILY:
            return True
        elif freq == RebalanceFrequency.WEEKLY:
            return (current_date - last_rebalance).days >= 7
        elif freq == RebalanceFrequency.MONTHLY:
            return current_date.month != last_rebalance.month
        elif freq == RebalanceFrequency.QUARTERLY:
            current_quarter = (current_date.month - 1) // 3
            last_quarter = (last_rebalance.month - 1) // 3
            return current_quarter != last_quarter or current_date.year != last_rebalance.year

        return True

    def _execute_rebalance(
        self,
        signals: List[Signal],
        prices: Dict[str, float],
        timestamp: datetime,
        volumes: Optional[Dict[str, float]] = None
    ) -> None:
        """Executa rebalanceamento do portfolio"""
        # Calcula pesos alvo
        target_weights = {s.symbol: s.weight for s in signals}
        current_weights = self.portfolio.get_position_weights()

        # Verifica limites de risco antes de executar
        for symbol, weight in target_weights.items():
            passed, alert = self.risk_manager.check_position_size(
                symbol, weight, current_weights
            )
            if not passed:
                logger.warning(f"Posicao rejeitada por risco: {alert}")
                target_weights[symbol] = min(
                    abs(weight),
                    self.config.risk_config.max_position_size
                ) * np.sign(weight)

        # Verifica alavancagem
        total_weight = sum(abs(w) for w in target_weights.values())
        if total_weight > self.config.risk_config.max_portfolio_leverage:
            scale = self.config.risk_config.max_portfolio_leverage / total_weight
            target_weights = {k: v * scale for k, v in target_weights.items()}

        # Calcula trades necessarios
        trades_needed = {}
        portfolio_value = self.portfolio.total_equity

        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            weight_diff = target_weight - current_weight

            if abs(weight_diff) > 0.001:  # Threshold minimo
                target_value = portfolio_value * weight_diff
                if symbol in prices and prices[symbol] > 0:
                    quantity = abs(target_value / prices[symbol])
                    side = PositionSide.LONG if weight_diff > 0 else PositionSide.SHORT
                    trades_needed[symbol] = (quantity, side, prices[symbol])

        # Fecha posicoes que nao estao nos targets
        for symbol in list(self.portfolio.positions.keys()):
            if symbol not in target_weights or target_weights[symbol] == 0:
                if symbol in prices:
                    price = prices[symbol]
                    volume = volumes.get(symbol) if volumes else None
                    position = self.portfolio.positions[symbol]
                    commission, slippage = self.cost_model.calculate_total_cost(
                        position.quantity, price, volume
                    )
                    self.portfolio.close_position(
                        symbol, price, commission, slippage, timestamp
                    )

        # Executa trades
        for symbol, (quantity, side, price) in trades_needed.items():
            volume = volumes.get(symbol) if volumes else None
            commission, slippage = self.cost_model.calculate_total_cost(
                quantity, price, volume
            )

            order = self.portfolio.create_order(
                symbol=symbol,
                quantity=quantity,
                side=side,
                order_type=OrderType.MARKET
            )

            self.portfolio.execute_order(
                order=order,
                fill_price=price,
                commission=commission,
                slippage=slippage,
                timestamp=timestamp
            )

    def run(
        self,
        data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> BacktestResult:
        """
        Executa backtest

        Args:
            data: DataFrame com OHLCV para cada ativo (MultiIndex ou wide format)
            benchmark_data: DataFrame com dados do benchmark (opcional)
        """
        logger.info("Iniciando backtest...")

        # Reset
        self.portfolio.reset()
        self.risk_manager.reset()
        self._signals_history.clear()
        self._rebalance_dates.clear()

        # Prepara dados
        if isinstance(data.columns, pd.MultiIndex):
            symbols = data.columns.get_level_values(0).unique().tolist()
            dates = data.index
        else:
            symbols = [col.split('_')[0] for col in data.columns if '_close' in col.lower()]
            symbols = list(set(symbols))
            dates = data.index

        # Warmup
        warmup_end_idx = min(self.config.warmup_period, len(dates) - 1)

        last_rebalance = None
        positions_history = []

        # Loop principal
        for i, current_date in enumerate(dates):
            if i < warmup_end_idx:
                continue

            timestamp = pd.Timestamp(current_date).to_pydatetime()

            # Obtem precos atuais
            current_prices = self._get_prices_at_date(data, current_date, symbols)
            volumes = self._get_volumes_at_date(data, current_date, symbols)

            # Atualiza precos no portfolio
            self.portfolio.update_prices(current_prices, timestamp)

            # Verifica drawdown
            self.risk_manager.check_drawdown(self.portfolio.total_equity)

            # Verifica se deve rebalancear
            if self._should_rebalance(timestamp, last_rebalance):
                # Obtem dados historicos para estrategia
                historical_data = data.loc[:current_date]

                # Gera sinais
                signals = self.strategy.generate_signals(
                    historical_data,
                    current_date,
                    symbols
                )

                if signals:
                    self._signals_history.append(signals)
                    self._rebalance_dates.append(timestamp)

                    # Executa rebalanceamento
                    self._execute_rebalance(
                        signals,
                        current_prices,
                        timestamp,
                        volumes
                    )

                    last_rebalance = timestamp

            # Registra posicoes
            positions_history.append({
                'date': current_date,
                'equity': self.portfolio.total_equity,
                'cash': self.portfolio.cash,
                'positions': len(self.portfolio.positions),
                **self.portfolio.get_position_weights()
            })

        # Processa benchmark
        if benchmark_data is not None:
            benchmark_close = benchmark_data['Close'] if 'Close' in benchmark_data else benchmark_data.iloc[:, 0]
            self._benchmark_returns = benchmark_close.pct_change().dropna()

        # Calcula resultados
        result = self._calculate_results(positions_history)

        logger.info(f"Backtest concluido. CAGR: {result.cagr:.2%}, Sharpe: {result.sharpe_ratio:.2f}")

        return result

    def _get_prices_at_date(
        self,
        data: pd.DataFrame,
        date: datetime,
        symbols: List[str]
    ) -> Dict[str, float]:
        """Obtem precos de fechamento na data"""
        prices = {}

        if isinstance(data.columns, pd.MultiIndex):
            for symbol in symbols:
                try:
                    if (symbol, 'Close') in data.columns:
                        price = data.loc[date, (symbol, 'Close')]
                    elif (symbol, 'close') in data.columns:
                        price = data.loc[date, (symbol, 'close')]
                    elif (symbol, 'Adj Close') in data.columns:
                        price = data.loc[date, (symbol, 'Adj Close')]
                    else:
                        continue

                    if pd.notna(price) and price > 0:
                        prices[symbol] = float(price)
                except (KeyError, TypeError):
                    continue
        else:
            for symbol in symbols:
                for col_suffix in ['_Close', '_close', '_Adj Close', '_adj_close']:
                    col = f"{symbol}{col_suffix}"
                    if col in data.columns:
                        try:
                            price = data.loc[date, col]
                            if pd.notna(price) and price > 0:
                                prices[symbol] = float(price)
                            break
                        except (KeyError, TypeError):
                            continue

        return prices

    def _get_volumes_at_date(
        self,
        data: pd.DataFrame,
        date: datetime,
        symbols: List[str]
    ) -> Dict[str, float]:
        """Obtem volumes na data"""
        volumes = {}

        if isinstance(data.columns, pd.MultiIndex):
            for symbol in symbols:
                try:
                    if (symbol, 'Volume') in data.columns:
                        vol = data.loc[date, (symbol, 'Volume')]
                    elif (symbol, 'volume') in data.columns:
                        vol = data.loc[date, (symbol, 'volume')]
                    else:
                        continue

                    if pd.notna(vol) and vol > 0:
                        volumes[symbol] = float(vol)
                except (KeyError, TypeError):
                    continue

        return volumes

    def _calculate_results(self, positions_history: List[Dict]) -> BacktestResult:
        """Calcula metricas do backtest"""
        positions_df = pd.DataFrame(positions_history)
        positions_df.set_index('date', inplace=True)

        equity_curve = positions_df['equity']
        returns = equity_curve.pct_change().dropna()

        # Metricas basicas
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        years = len(returns) / 252
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        vol = returns.std() * np.sqrt(252)

        # Risk-adjusted
        rf_daily = 0.02 / 252  # Risk-free rate
        excess_returns = returns - rf_daily
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1
        sortino = (returns.mean() * 252) / downside_std if downside_std > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()

        # Duracao do drawdown
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            dd_groups = (~in_drawdown).cumsum()
            dd_durations = in_drawdown.groupby(dd_groups).sum()
            max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0
        else:
            max_dd_duration = 0

        avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0

        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Tail risk
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        skewness = returns.skew() if len(returns) > 0 else 0
        kurtosis = returns.kurtosis() if len(returns) > 0 else 0

        # Attribution
        if self._benchmark_returns is not None and len(self._benchmark_returns) > 0:
            aligned = pd.concat([returns, self._benchmark_returns], axis=1).dropna()
            if len(aligned) > 30:
                port_ret = aligned.iloc[:, 0]
                bench_ret = aligned.iloc[:, 1]

                cov = np.cov(port_ret, bench_ret)[0, 1]
                var_bench = np.var(bench_ret)
                beta = cov / var_bench if var_bench > 0 else 1.0

                alpha = (port_ret.mean() - beta * bench_ret.mean()) * 252

                tracking_diff = port_ret - bench_ret
                tracking_error = tracking_diff.std() * np.sqrt(252)
                info_ratio = (tracking_diff.mean() * 252) / tracking_error if tracking_error > 0 else 0
            else:
                alpha, beta, info_ratio, tracking_error = 0, 1, 0, 0
        else:
            alpha, beta, info_ratio, tracking_error = 0, 1, 0, 0

        # Trading metrics
        trades_df = self.portfolio.get_trades_df()
        num_trades = len(trades_df)

        if num_trades > 0:
            total_commission = trades_df['commission'].sum()
            total_slippage = trades_df['slippage'].sum()

            # Win rate (simplificado)
            trade_pnls = []
            for symbol in trades_df['symbol'].unique():
                symbol_trades = trades_df[trades_df['symbol'] == symbol]
                if len(symbol_trades) >= 2:
                    buys = symbol_trades[symbol_trades['side'] == 'long']
                    sells = symbol_trades[symbol_trades['side'] == 'short']
                    if len(buys) > 0 and len(sells) > 0:
                        pnl = sells['price'].mean() - buys['price'].mean()
                        trade_pnls.append(pnl)

            if trade_pnls:
                wins = [p for p in trade_pnls if p > 0]
                losses = [p for p in trade_pnls if p < 0]
                win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0
                avg_win = np.mean(wins) if wins else 0
                avg_loss = abs(np.mean(losses)) if losses else 0
                profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0
            else:
                win_rate, avg_win, avg_loss, profit_factor = 0, 0, 0, 0
        else:
            total_commission, total_slippage = 0, 0
            win_rate, avg_win, avg_loss, profit_factor = 0, 0, 0, 0

        # Turnover
        if len(positions_df) > 1:
            weight_cols = [c for c in positions_df.columns if c not in ['equity', 'cash', 'positions']]
            if weight_cols:
                weight_changes = positions_df[weight_cols].diff().abs().sum(axis=1)
                turnover = weight_changes.mean() * 252  # Anualizado
            else:
                turnover = 0
        else:
            turnover = 0

        # Risk metrics
        risk_metrics = self.risk_manager.calculate_risk_metrics(
            returns,
            self._benchmark_returns,
            self.portfolio.get_position_weights(),
            self.portfolio.total_equity,
            self.portfolio.leverage
        )

        return BacktestResult(
            total_return=total_return,
            cagr=cagr,
            annualized_volatility=vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            avg_drawdown=avg_dd,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            alpha=alpha,
            beta=beta,
            information_ratio=info_ratio,
            tracking_error=tracking_error,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_holding_period=0,  # TODO: calcular
            turnover=turnover,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_costs=total_commission + total_slippage,
            equity_curve=equity_curve,
            returns=returns,
            drawdown_series=drawdown,
            positions_history=positions_df,
            trades_history=trades_df,
            risk_metrics=risk_metrics
        )

    def run_parallel(
        self,
        data: pd.DataFrame,
        parameter_sets: List[Dict],
        max_workers: int = 4
    ) -> List[BacktestResult]:
        """Executa multiplos backtests em paralelo"""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for params in parameter_sets:
                strategy_copy = self.strategy.copy_with_params(params)
                config_copy = BacktestConfig(**{
                    **self.config.__dict__,
                    **params.get('backtest', {})
                })

                backtester = Backtester(
                    config=config_copy,
                    strategy=strategy_copy,
                    data_provider=self.data_provider
                )

                future = executor.submit(backtester.run, data)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Erro no backtest paralelo: {e}")

        return results
