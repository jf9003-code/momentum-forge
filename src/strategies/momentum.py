"""
Estrategia de Momentum Institucional
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, LongOnlyStrategy, StrategyMetadata
from src.core.backtester import Signal
from config.settings import MomentumStrategyConfig


class MomentumStrategy(LongOnlyStrategy):
    """
    Estrategia de Momentum Relativo Institucional.

    Seleciona os top N ativos com melhor momentum score ponderado,
    aplicando filtros de qualidade e gestao de risco.
    """

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        config: Optional[MomentumStrategyConfig] = None
    ):
        super().__init__(parameters)
        self.config = config or MomentumStrategyConfig()

        # Override com parametros se fornecidos
        if parameters:
            if 'lookback_short' in parameters:
                self.config.lookback_short = parameters['lookback_short']
            if 'lookback_long' in parameters:
                self.config.lookback_long = parameters['lookback_long']
            if 'lookback_sma' in parameters:
                self.config.lookback_sma = parameters['lookback_sma']
            if 'weight_short' in parameters:
                self.config.weight_short = parameters['weight_short']
            if 'min_return' in parameters:
                self.config.min_return = parameters['min_return']
            if 'top_n' in parameters:
                self.config.top_n = parameters['top_n']
            if 'universe' in parameters:
                self.config.universe = parameters['universe']

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="Momentum Relativo Top-N",
            version="2.0.0",
            author="Momentum Forge",
            description="Estrategia de momentum relativo com filtros de qualidade",
            asset_classes=["equity", "etf"],
            min_history_days=max(self.config.lookback_long, self.config.lookback_sma) + 50,
            rebalance_frequency="daily",
            parameters={
                'lookback_short': self.config.lookback_short,
                'lookback_long': self.config.lookback_long,
                'lookback_sma': self.config.lookback_sma,
                'weight_short': self.config.weight_short,
                'min_return': self.config.min_return,
                'top_n': self.config.top_n
            }
        )

    def _get_price_series(
        self,
        data: pd.DataFrame,
        symbol: str,
        price_type: str = 'Close'
    ) -> Optional[pd.Series]:
        """Extrai serie de precos para um simbolo"""
        try:
            if isinstance(data.columns, pd.MultiIndex):
                # MultiIndex format (symbol, field)
                for field in [price_type, price_type.lower(), 'Adj Close', 'adj_close']:
                    if (symbol, field) in data.columns:
                        return data[(symbol, field)]
            else:
                # Wide format
                for suffix in [f'_{price_type}', f'_{price_type.lower()}', '_Adj Close', '_adj_close', '']:
                    col = f"{symbol}{suffix}"
                    if col in data.columns:
                        return data[col]
            return None
        except Exception:
            return None

    def _get_volume_series(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Optional[pd.Series]:
        """Extrai serie de volume para um simbolo"""
        try:
            if isinstance(data.columns, pd.MultiIndex):
                for field in ['Volume', 'volume']:
                    if (symbol, field) in data.columns:
                        return data[(symbol, field)]
            else:
                for suffix in ['_Volume', '_volume']:
                    col = f"{symbol}{suffix}"
                    if col in data.columns:
                        return data[col]
            return None
        except Exception:
            return None

    def calculate_momentum_score(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None
    ) -> Optional[Dict[str, float]]:
        """
        Calcula score de momentum para um ativo.

        Retorna dict com score e metricas ou None se invalido.
        """
        if prices is None or len(prices) < self.config.lookback_sma:
            return None

        try:
            current_price = prices.iloc[-1]

            if pd.isna(current_price) or current_price <= 0:
                return None

            # Retorno curto prazo (3 meses)
            if len(prices) >= self.config.lookback_short:
                price_short = prices.iloc[-self.config.lookback_short]
                if pd.notna(price_short) and price_short > 0:
                    return_short = (current_price / price_short) - 1
                else:
                    return None
            else:
                return None

            # Retorno longo prazo (6 meses)
            if len(prices) >= self.config.lookback_long:
                price_long = prices.iloc[-self.config.lookback_long]
                if pd.notna(price_long) and price_long > 0:
                    return_long = (current_price / price_long) - 1
                else:
                    return_long = return_short
            else:
                return_long = return_short

            # SMA 200
            sma = prices.rolling(window=self.config.lookback_sma).mean().iloc[-1]
            above_sma = current_price > sma if pd.notna(sma) else True

            # Filtro de retorno minimo
            if return_short < self.config.min_return:
                return None

            # Filtro de tendencia (acima da SMA)
            if not above_sma:
                return None

            # Score ponderado
            score = (
                self.config.weight_short * return_short +
                (1 - self.config.weight_short) * return_long
            )

            # Volatilidade (para ajuste de risco)
            returns = prices.pct_change().dropna()
            volatility = returns.tail(63).std() * np.sqrt(252) if len(returns) >= 63 else 0.2

            # Volume medio (para liquidez)
            avg_volume = volume.tail(20).mean() if volume is not None and len(volume) >= 20 else 0

            return {
                'score': score,
                'return_short': return_short,
                'return_long': return_long,
                'volatility': volatility,
                'avg_volume': avg_volume,
                'above_sma': above_sma,
                'current_price': current_price
            }

        except Exception:
            return None

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_date: datetime,
        universe: List[str]
    ) -> List[Signal]:
        """Gera sinais de momentum para a data atual"""

        # Usa universo da config se nao fornecido
        if not universe:
            universe = self.config.universe

        # Calcula scores para cada ativo
        scores = {}

        for symbol in universe:
            prices = self._get_price_series(data, symbol)
            volume = self._get_volume_series(data, symbol)

            result = self.calculate_momentum_score(prices, volume)

            if result is not None:
                scores[symbol] = result

        if not scores:
            return []

        # Ordena por score
        sorted_symbols = sorted(
            scores.keys(),
            key=lambda s: scores[s]['score'],
            reverse=True
        )

        # Seleciona top N
        top_symbols = sorted_symbols[:self.config.top_n]

        # Cria sinais com pesos iguais
        weight_per_asset = 1.0 / len(top_symbols) if top_symbols else 0

        signals = []
        for symbol in top_symbols:
            score_data = scores[symbol]

            signal = Signal(
                symbol=symbol,
                weight=weight_per_asset,
                score=score_data['score'],
                timestamp=pd.Timestamp(current_date).to_pydatetime(),
                metadata={
                    'return_short': score_data['return_short'],
                    'return_long': score_data['return_long'],
                    'volatility': score_data['volatility'],
                    'rank': top_symbols.index(symbol) + 1
                }
            )
            signals.append(signal)

        return signals

    def copy_with_params(self, parameters: Dict[str, Any]) -> 'MomentumStrategy':
        """Cria copia com novos parametros"""
        new_params = {**self._parameters, **parameters}
        return MomentumStrategy(parameters=new_params, config=self.config)


class RiskAdjustedMomentumStrategy(MomentumStrategy):
    """
    Variante com ajuste de risco (volatility scaling).

    Pondera ativos pelo inverso da volatilidade para
    risk parity dentro do portfolio momentum.
    """

    @property
    def metadata(self) -> StrategyMetadata:
        base = super().metadata
        return StrategyMetadata(
            name="Momentum Risk-Adjusted",
            version="2.0.0",
            author="Momentum Forge",
            description="Momentum com ponderacao por volatilidade inversa",
            asset_classes=base.asset_classes,
            min_history_days=base.min_history_days,
            rebalance_frequency=base.rebalance_frequency,
            parameters=base.parameters
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_date: datetime,
        universe: List[str]
    ) -> List[Signal]:
        """Gera sinais com ponderacao por volatilidade"""

        # Obtem sinais base
        base_signals = super().generate_signals(data, current_date, universe)

        if not base_signals:
            return []

        # Calcula pesos por volatilidade inversa
        inv_vols = {}
        for signal in base_signals:
            vol = signal.metadata.get('volatility', 0.2)
            inv_vols[signal.symbol] = 1.0 / vol if vol > 0 else 1.0

        total_inv_vol = sum(inv_vols.values())

        # Ajusta pesos
        adjusted_signals = []
        for signal in base_signals:
            new_weight = inv_vols[signal.symbol] / total_inv_vol

            adjusted_signals.append(Signal(
                symbol=signal.symbol,
                weight=new_weight,
                score=signal.score,
                timestamp=signal.timestamp,
                metadata={
                    **signal.metadata,
                    'original_weight': signal.weight,
                    'vol_adjusted': True
                }
            ))

        return adjusted_signals


class DualMomentumStrategy(LongOnlyStrategy):
    """
    Estrategia Dual Momentum (Absolute + Relative).

    Combina momentum absoluto (vs cash/bonds) com
    momentum relativo (entre ativos).
    """

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        config: Optional[MomentumStrategyConfig] = None,
        safe_asset: str = "SHY",  # Short-term treasury
        absolute_threshold: float = 0.0
    ):
        super().__init__(parameters)
        self.config = config or MomentumStrategyConfig()
        self.safe_asset = safe_asset
        self.absolute_threshold = absolute_threshold

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="Dual Momentum",
            version="2.0.0",
            author="Momentum Forge",
            description="Momentum absoluto + relativo com protecao em cash",
            asset_classes=["equity", "etf", "bond"],
            min_history_days=max(self.config.lookback_long, self.config.lookback_sma) + 50,
            rebalance_frequency="monthly",
            parameters={
                'lookback_short': self.config.lookback_short,
                'safe_asset': self.safe_asset,
                'absolute_threshold': self.absolute_threshold
            }
        )

    def _get_price_series(
        self,
        data: pd.DataFrame,
        symbol: str,
        price_type: str = 'Close'
    ) -> Optional[pd.Series]:
        """Extrai serie de precos para um simbolo"""
        try:
            if isinstance(data.columns, pd.MultiIndex):
                for field in [price_type, price_type.lower(), 'Adj Close', 'adj_close']:
                    if (symbol, field) in data.columns:
                        return data[(symbol, field)]
            else:
                for suffix in [f'_{price_type}', f'_{price_type.lower()}', '_Adj Close', '_adj_close', '']:
                    col = f"{symbol}{suffix}"
                    if col in data.columns:
                        return data[col]
            return None
        except Exception:
            return None

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_date: datetime,
        universe: List[str]
    ) -> List[Signal]:
        """Gera sinais de dual momentum"""

        if not universe:
            universe = self.config.universe

        # Calcula retornos para cada ativo
        returns = {}

        for symbol in universe + [self.safe_asset]:
            prices = self._get_price_series(data, symbol)

            if prices is None or len(prices) < self.config.lookback_short:
                continue

            current = prices.iloc[-1]
            past = prices.iloc[-self.config.lookback_short]

            if pd.notna(current) and pd.notna(past) and past > 0:
                returns[symbol] = (current / past) - 1

        if not returns:
            return []

        # Momentum absoluto: melhor ativo vs safe asset
        safe_return = returns.get(self.safe_asset, 0)

        # Filtra ativos com momentum absoluto positivo
        valid_assets = {
            s: r for s, r in returns.items()
            if s != self.safe_asset and r > self.absolute_threshold
        }

        if not valid_assets:
            # Vai para ativo seguro
            if self.safe_asset in returns:
                return [Signal(
                    symbol=self.safe_asset,
                    weight=1.0,
                    score=safe_return,
                    timestamp=pd.Timestamp(current_date).to_pydatetime(),
                    metadata={'reason': 'absolute_momentum_negative'}
                )]
            return []

        # Momentum relativo: seleciona melhor entre validos
        best_symbol = max(valid_assets, key=valid_assets.get)
        best_return = valid_assets[best_symbol]

        # Se melhor ativo perder para safe asset, vai para safe
        if best_return < safe_return:
            return [Signal(
                symbol=self.safe_asset,
                weight=1.0,
                score=safe_return,
                timestamp=pd.Timestamp(current_date).to_pydatetime(),
                metadata={'reason': 'relative_momentum_negative'}
            )]

        return [Signal(
            symbol=best_symbol,
            weight=1.0,
            score=best_return,
            timestamp=pd.Timestamp(current_date).to_pydatetime(),
            metadata={
                'return': best_return,
                'safe_return': safe_return,
                'reason': 'momentum_positive'
            }
        )]
