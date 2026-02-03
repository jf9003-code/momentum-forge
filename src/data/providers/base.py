"""
Classe base abstrata para provedores de dados
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import date, datetime
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class DataFrequency(Enum):
    """Frequencia dos dados"""
    TICK = "tick"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"


class AssetClass(Enum):
    """Classes de ativos"""
    EQUITY = "equity"
    ETF = "etf"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    BOND = "bond"
    INDEX = "index"
    OPTION = "option"
    FUTURE = "future"


@dataclass
class AssetInfo:
    """Informacoes sobre um ativo"""
    symbol: str
    name: str
    asset_class: AssetClass
    exchange: str
    currency: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    market_cap: Optional[float] = None


@dataclass
class DataQuality:
    """Metricas de qualidade dos dados"""
    total_rows: int
    missing_rows: int
    missing_percentage: float
    gaps_detected: int
    outliers_detected: int
    start_date: date
    end_date: date
    is_valid: bool
    issues: List[str]


class DataProvider(ABC):
    """
    Classe base abstrata para provedores de dados.

    Todos os provedores (Yahoo, Bloomberg, Refinitiv, etc)
    devem herdar desta classe.
    """

    def __init__(self, name: str, rate_limit: int = 10):
        self.name = name
        self.rate_limit = rate_limit  # Requests por segundo
        self._cache: Dict[str, pd.DataFrame] = {}

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Optional[Union[str, date, datetime]] = None,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """
        Obtem dados OHLCV para um simbolo.

        Args:
            symbol: Simbolo do ativo
            start_date: Data inicial
            end_date: Data final (opcional, default=hoje)
            frequency: Frequencia dos dados

        Returns:
            DataFrame com colunas: Open, High, Low, Close, Volume, Adj Close
        """
        pass

    @abstractmethod
    def get_ohlcv_multiple(
        self,
        symbols: List[str],
        start_date: Union[str, date, datetime],
        end_date: Optional[Union[str, date, datetime]] = None,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """
        Obtem dados OHLCV para multiplos simbolos.

        Returns:
            DataFrame com MultiIndex (symbol, field) nas colunas
        """
        pass

    @abstractmethod
    def get_asset_info(self, symbol: str) -> Optional[AssetInfo]:
        """Obtem informacoes sobre um ativo"""
        pass

    def search_symbols(
        self,
        query: str,
        asset_class: Optional[AssetClass] = None,
        limit: int = 10
    ) -> List[AssetInfo]:
        """Busca simbolos por nome/ticker"""
        return []

    def get_available_symbols(
        self,
        asset_class: Optional[AssetClass] = None
    ) -> List[str]:
        """Lista simbolos disponiveis"""
        return []

    def validate_data(self, data: pd.DataFrame) -> DataQuality:
        """Valida qualidade dos dados"""
        issues = []
        total = len(data)
        missing = data.isnull().sum().sum()

        # Detecta gaps (dias faltantes)
        if isinstance(data.index, pd.DatetimeIndex):
            expected_days = pd.date_range(data.index.min(), data.index.max(), freq='B')
            gaps = len(expected_days) - len(data)
            if gaps > len(expected_days) * 0.1:
                issues.append(f"Muitos gaps detectados: {gaps} dias faltantes")
        else:
            gaps = 0

        # Detecta outliers simples (Z-score > 5)
        outliers = 0
        if 'Close' in data.columns:
            returns = data['Close'].pct_change().dropna()
            z_scores = (returns - returns.mean()) / returns.std()
            outliers = (z_scores.abs() > 5).sum()
            if outliers > 0:
                issues.append(f"{outliers} outliers detectados")

        # Valida precos
        if 'Close' in data.columns:
            if (data['Close'] <= 0).any():
                issues.append("Precos negativos ou zero encontrados")
            if data['Close'].isna().any():
                issues.append("Precos de fechamento faltantes")

        # Valida volume
        if 'Volume' in data.columns:
            if (data['Volume'] < 0).any():
                issues.append("Volumes negativos encontrados")

        return DataQuality(
            total_rows=total,
            missing_rows=int(missing),
            missing_percentage=missing / (total * len(data.columns)) if total > 0 else 0,
            gaps_detected=gaps,
            outliers_detected=outliers,
            start_date=data.index.min().date() if hasattr(data.index.min(), 'date') else data.index.min(),
            end_date=data.index.max().date() if hasattr(data.index.max(), 'date') else data.index.max(),
            is_valid=len(issues) == 0,
            issues=issues
        )

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Limpa e corrige dados"""
        df = data.copy()

        # Forward fill para gaps pequenos
        df = df.ffill(limit=5)

        # Remove linhas com muitos NaN
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=int(threshold))

        # Corrige valores negativos de volume
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].clip(lower=0)

        # Remove duplicatas
        df = df[~df.index.duplicated(keep='first')]

        # Ordena por data
        df = df.sort_index()

        return df

    def cache_key(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: DataFrequency
    ) -> str:
        """Gera chave de cache"""
        return f"{self.name}_{symbol}_{start_date}_{end_date}_{frequency.value}"

    def clear_cache(self) -> None:
        """Limpa cache"""
        self._cache.clear()
