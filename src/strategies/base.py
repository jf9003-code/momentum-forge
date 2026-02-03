"""
Classe base abstrata para estrategias
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from dataclasses import dataclass, field

from src.core.backtester import Signal


@dataclass
class StrategyMetadata:
    """Metadados da estrategia"""
    name: str
    version: str
    author: str
    description: str
    asset_classes: List[str] = field(default_factory=list)
    min_history_days: int = 252
    rebalance_frequency: str = "daily"
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Classe base abstrata para todas as estrategias.

    Todas as estrategias devem herdar desta classe e implementar
    o metodo generate_signals.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self._parameters = parameters or {}
        self._is_initialized = False
        self._last_signals: List[Signal] = []

    @property
    @abstractmethod
    def metadata(self) -> StrategyMetadata:
        """Retorna metadados da estrategia"""
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        """Retorna parametros atuais"""
        return self._parameters.copy()

    @property
    def name(self) -> str:
        """Nome da estrategia"""
        return self.metadata.name

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        current_date: datetime,
        universe: List[str]
    ) -> List[Signal]:
        """
        Gera sinais de trading para a data atual.

        Args:
            data: DataFrame com dados historicos ate current_date
            current_date: Data atual do backtest
            universe: Lista de simbolos disponiveis

        Returns:
            Lista de Signal com pesos alvo para cada ativo
        """
        pass

    def initialize(self, data: pd.DataFrame) -> None:
        """
        Inicializa a estrategia com dados historicos.
        Chamado uma vez antes do inicio do backtest.
        """
        self._is_initialized = True

    def on_trade(self, symbol: str, quantity: float, price: float) -> None:
        """Callback quando um trade e executado"""
        pass

    def on_rebalance(self, date: datetime, signals: List[Signal]) -> None:
        """Callback apos rebalanceamento"""
        self._last_signals = signals

    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Atualiza parametros da estrategia"""
        self._parameters.update(parameters)

    def copy_with_params(self, parameters: Dict[str, Any]) -> 'BaseStrategy':
        """Cria copia da estrategia com novos parametros"""
        new_params = {**self._parameters, **parameters}
        return self.__class__(new_params)

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Obtem valor de um parametro"""
        return self._parameters.get(key, default)

    def validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """Valida se dados contem colunas necessarias"""
        if isinstance(data.columns, pd.MultiIndex):
            return True  # MultiIndex e mais flexivel

        for col in required_columns:
            if col not in data.columns:
                return False
        return True

    @staticmethod
    def calculate_returns(
        prices: pd.Series,
        periods: int = 1
    ) -> pd.Series:
        """Calcula retornos para N periodos"""
        return prices.pct_change(periods)

    @staticmethod
    def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
        """Calcula media movel simples"""
        return prices.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
        """Calcula media movel exponencial"""
        return prices.ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_volatility(
        returns: pd.Series,
        window: int = 20,
        annualize: bool = True
    ) -> pd.Series:
        """Calcula volatilidade rolling"""
        vol = returns.rolling(window=window).std()
        if annualize:
            vol = vol * (252 ** 0.5)
        return vol

    @staticmethod
    def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Normaliza pesos para somar 1"""
        total = sum(abs(w) for w in weights.values())
        if total == 0:
            return weights
        return {k: v / total for k, v in weights.items()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self._parameters})"


class LongOnlyStrategy(BaseStrategy):
    """Classe base para estrategias apenas long"""

    def normalize_signals(self, signals: List[Signal]) -> List[Signal]:
        """Normaliza sinais para serem apenas long e somarem 1"""
        # Filtra apenas sinais positivos
        long_signals = [s for s in signals if s.weight > 0]

        if not long_signals:
            return []

        # Normaliza pesos
        total_weight = sum(s.weight for s in long_signals)

        return [
            Signal(
                symbol=s.symbol,
                weight=s.weight / total_weight,
                score=s.score,
                timestamp=s.timestamp,
                metadata=s.metadata
            )
            for s in long_signals
        ]


class LongShortStrategy(BaseStrategy):
    """Classe base para estrategias long/short"""

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        gross_exposure: float = 2.0,
        net_exposure: float = 0.0
    ):
        super().__init__(parameters)
        self.gross_exposure = gross_exposure
        self.net_exposure = net_exposure

    def balance_signals(
        self,
        long_signals: List[Signal],
        short_signals: List[Signal]
    ) -> List[Signal]:
        """Balanceia sinais long/short para exposicao desejada"""
        if not long_signals and not short_signals:
            return []

        # Normaliza cada lado
        long_weight = (self.gross_exposure + self.net_exposure) / 2
        short_weight = (self.gross_exposure - self.net_exposure) / 2

        balanced = []

        if long_signals:
            long_total = sum(s.weight for s in long_signals)
            for s in long_signals:
                balanced.append(Signal(
                    symbol=s.symbol,
                    weight=(s.weight / long_total) * long_weight,
                    score=s.score,
                    timestamp=s.timestamp,
                    metadata=s.metadata
                ))

        if short_signals:
            short_total = sum(abs(s.weight) for s in short_signals)
            for s in short_signals:
                balanced.append(Signal(
                    symbol=s.symbol,
                    weight=-(abs(s.weight) / short_total) * short_weight,
                    score=s.score,
                    timestamp=s.timestamp,
                    metadata=s.metadata
                ))

        return balanced
