"""
Provedor de dados Yahoo Finance
"""
from typing import Dict, List, Optional, Union
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import logging
import time

try:
    import yfinance as yf
except ImportError:
    yf = None

from src.data.providers.base import (
    DataProvider,
    DataFrequency,
    AssetClass,
    AssetInfo,
    DataQuality
)


logger = logging.getLogger(__name__)


# Mapeamento de frequencias para yfinance
FREQUENCY_MAP = {
    DataFrequency.MINUTE_1: '1m',
    DataFrequency.MINUTE_5: '5m',
    DataFrequency.MINUTE_15: '15m',
    DataFrequency.MINUTE_30: '30m',
    DataFrequency.HOUR_1: '1h',
    DataFrequency.DAILY: '1d',
    DataFrequency.WEEKLY: '1wk',
    DataFrequency.MONTHLY: '1mo',
}

# Universos pre-definidos
UNIVERSES = {
    'etf_diversified': [
        'SPY', 'QQQ', 'IWM', 'EFA', 'EEM',
        'TLT', 'IEF', 'LQD', 'HYG',
        'GLD', 'SLV', 'USO', 'DBA',
        'VNQ', 'XLF', 'XLE', 'XLK', 'XLV'
    ],
    'sectors': [
        'XLB', 'XLC', 'XLE', 'XLF', 'XLI',
        'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY'
    ],
    'factors': [
        'MTUM', 'VLUE', 'SIZE', 'QUAL', 'USMV'
    ],
    'international': [
        'EFA', 'EEM', 'VEA', 'VWO', 'IEFA', 'IEMG'
    ],
    'bonds': [
        'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP', 'AGG', 'BND'
    ],
    'commodities': [
        'GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBC'
    ]
}


class YahooDataProvider(DataProvider):
    """
    Provedor de dados usando Yahoo Finance.

    Limitacoes:
    - Dados intraday limitados a 7-60 dias
    - Rate limiting pode ocorrer
    - Dados podem ter atrasos
    """

    def __init__(self, rate_limit: int = 2):
        super().__init__(name="Yahoo Finance", rate_limit=rate_limit)
        self._last_request_time = 0

        if yf is None:
            raise ImportError("yfinance nao esta instalado. Execute: pip install yfinance")

    def _rate_limit_wait(self):
        """Espera para respeitar rate limit"""
        elapsed = time.time() - self._last_request_time
        wait_time = 1.0 / self.rate_limit

        if elapsed < wait_time:
            time.sleep(wait_time - elapsed)

        self._last_request_time = time.time()

    def _parse_date(self, d: Union[str, date, datetime]) -> datetime:
        """Converte para datetime"""
        if isinstance(d, str):
            return pd.to_datetime(d)
        elif isinstance(d, date) and not isinstance(d, datetime):
            return datetime.combine(d, datetime.min.time())
        return d

    def get_ohlcv(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Optional[Union[str, date, datetime]] = None,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Obtem dados OHLCV para um simbolo"""
        self._rate_limit_wait()

        start = self._parse_date(start_date)
        end = self._parse_date(end_date) if end_date else datetime.now()

        interval = FREQUENCY_MAP.get(frequency, '1d')

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start,
                end=end + timedelta(days=1),
                interval=interval,
                auto_adjust=False
            )

            if df.empty:
                logger.warning(f"Nenhum dado encontrado para {symbol}")
                return pd.DataFrame()

            # Padroniza colunas
            df = df.rename(columns={
                'Adj Close': 'Adj Close',
                'Stock Splits': 'Splits',
                'Dividends': 'Dividends'
            })

            # Mantem apenas colunas relevantes
            cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            df = df[[c for c in cols if c in df.columns]]

            # Adiciona Adj Close se nao existir
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']

            logger.debug(f"Obtidos {len(df)} registros para {symbol}")

            return df

        except Exception as e:
            logger.error(f"Erro ao obter dados para {symbol}: {e}")
            return pd.DataFrame()

    def get_ohlcv_multiple(
        self,
        symbols: List[str],
        start_date: Union[str, date, datetime],
        end_date: Optional[Union[str, date, datetime]] = None,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Obtem dados OHLCV para multiplos simbolos"""
        self._rate_limit_wait()

        start = self._parse_date(start_date)
        end = self._parse_date(end_date) if end_date else datetime.now()

        interval = FREQUENCY_MAP.get(frequency, '1d')

        try:
            # Download em batch
            df = yf.download(
                symbols,
                start=start,
                end=end + timedelta(days=1),
                interval=interval,
                auto_adjust=False,
                group_by='ticker',
                threads=True,
                progress=False
            )

            if df.empty:
                logger.warning("Nenhum dado encontrado para os simbolos")
                return pd.DataFrame()

            # Se apenas um simbolo, reorganiza
            if len(symbols) == 1:
                symbol = symbols[0]
                df.columns = pd.MultiIndex.from_product([[symbol], df.columns])

            # Garante formato correto do MultiIndex
            if not isinstance(df.columns, pd.MultiIndex):
                # Tenta reconstruir MultiIndex
                new_cols = []
                for col in df.columns:
                    if isinstance(col, tuple):
                        new_cols.append(col)
                    else:
                        new_cols.append((symbols[0], col))
                df.columns = pd.MultiIndex.from_tuples(new_cols)

            logger.debug(f"Obtidos dados para {len(symbols)} simbolos")

            return df

        except Exception as e:
            logger.error(f"Erro ao obter dados multiplos: {e}")
            return pd.DataFrame()

    def get_asset_info(self, symbol: str) -> Optional[AssetInfo]:
        """Obtem informacoes sobre um ativo"""
        self._rate_limit_wait()

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or 'symbol' not in info:
                return None

            # Determina classe do ativo
            quote_type = info.get('quoteType', '').upper()
            asset_class_map = {
                'EQUITY': AssetClass.EQUITY,
                'ETF': AssetClass.ETF,
                'CRYPTOCURRENCY': AssetClass.CRYPTO,
                'CURRENCY': AssetClass.FOREX,
                'FUTURE': AssetClass.FUTURE,
                'INDEX': AssetClass.INDEX,
                'OPTION': AssetClass.OPTION,
            }
            asset_class = asset_class_map.get(quote_type, AssetClass.EQUITY)

            return AssetInfo(
                symbol=info.get('symbol', symbol),
                name=info.get('shortName', info.get('longName', symbol)),
                asset_class=asset_class,
                exchange=info.get('exchange', 'Unknown'),
                currency=info.get('currency', 'USD'),
                sector=info.get('sector'),
                industry=info.get('industry'),
                country=info.get('country'),
                market_cap=info.get('marketCap')
            )

        except Exception as e:
            logger.error(f"Erro ao obter info para {symbol}: {e}")
            return None

    def search_symbols(
        self,
        query: str,
        asset_class: Optional[AssetClass] = None,
        limit: int = 10
    ) -> List[AssetInfo]:
        """Busca simbolos (limitado no Yahoo)"""
        # Yahoo nao tem API de busca publica
        # Retorna match simples nos universos conhecidos
        results = []
        query_upper = query.upper()

        for universe_name, symbols in UNIVERSES.items():
            for symbol in symbols:
                if query_upper in symbol:
                    info = self.get_asset_info(symbol)
                    if info:
                        if asset_class is None or info.asset_class == asset_class:
                            results.append(info)
                            if len(results) >= limit:
                                return results

        return results

    def get_available_symbols(
        self,
        asset_class: Optional[AssetClass] = None
    ) -> List[str]:
        """Lista simbolos disponiveis nos universos pre-definidos"""
        all_symbols = set()

        for symbols in UNIVERSES.values():
            all_symbols.update(symbols)

        if asset_class is None:
            return sorted(list(all_symbols))

        # Filtra por classe (requer chamadas adicionais)
        filtered = []
        for symbol in all_symbols:
            info = self.get_asset_info(symbol)
            if info and info.asset_class == asset_class:
                filtered.append(symbol)

        return sorted(filtered)

    def get_universe(self, universe_name: str) -> List[str]:
        """Obtem lista de simbolos de um universo pre-definido"""
        return UNIVERSES.get(universe_name, [])

    def list_universes(self) -> List[str]:
        """Lista universos disponiveis"""
        return list(UNIVERSES.keys())

    def get_benchmark_data(
        self,
        benchmark: str = "SPY",
        start_date: Union[str, date, datetime] = None,
        end_date: Optional[Union[str, date, datetime]] = None
    ) -> pd.DataFrame:
        """Obtem dados do benchmark"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365*10)

        return self.get_ohlcv(
            symbol=benchmark,
            start_date=start_date,
            end_date=end_date,
            frequency=DataFrequency.DAILY
        )

    def get_risk_free_rate(
        self,
        start_date: Union[str, date, datetime],
        end_date: Optional[Union[str, date, datetime]] = None,
        proxy: str = "^IRX"  # 13-week T-bill
    ) -> pd.Series:
        """Obtem taxa livre de risco (proxy)"""
        df = self.get_ohlcv(
            symbol=proxy,
            start_date=start_date,
            end_date=end_date,
            frequency=DataFrequency.DAILY
        )

        if df.empty:
            return pd.Series(dtype=float)

        # Converte para taxa diaria
        return (df['Close'] / 100) / 252


class YahooDataProviderCached(YahooDataProvider):
    """
    Versao com cache persistente do Yahoo provider.

    Usa cache em memoria para evitar requisicoes repetidas.
    """

    def __init__(self, rate_limit: int = 2, cache_hours: int = 24):
        super().__init__(rate_limit=rate_limit)
        self.cache_hours = cache_hours
        self._cache_timestamps: Dict[str, datetime] = {}

    def _is_cache_valid(self, key: str) -> bool:
        """Verifica se cache ainda e valido"""
        if key not in self._cache_timestamps:
            return False

        age = datetime.now() - self._cache_timestamps[key]
        return age.total_seconds() < self.cache_hours * 3600

    def get_ohlcv(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Optional[Union[str, date, datetime]] = None,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Obtem dados com cache"""
        start = self._parse_date(start_date)
        end = self._parse_date(end_date) if end_date else datetime.now()

        cache_key = self.cache_key(symbol, start.date(), end.date(), frequency)

        if cache_key in self._cache and self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit para {symbol}")
            return self._cache[cache_key].copy()

        # Busca dados
        df = super().get_ohlcv(symbol, start_date, end_date, frequency)

        if not df.empty:
            self._cache[cache_key] = df.copy()
            self._cache_timestamps[cache_key] = datetime.now()

        return df

    def get_ohlcv_multiple(
        self,
        symbols: List[str],
        start_date: Union[str, date, datetime],
        end_date: Optional[Union[str, date, datetime]] = None,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Obtem dados multiplos com cache"""
        start = self._parse_date(start_date)
        end = self._parse_date(end_date) if end_date else datetime.now()

        cache_key = f"multi_{'_'.join(sorted(symbols))}_{start.date()}_{end.date()}_{frequency.value}"

        if cache_key in self._cache and self._is_cache_valid(cache_key):
            logger.debug("Cache hit para multiplos simbolos")
            return self._cache[cache_key].copy()

        df = super().get_ohlcv_multiple(symbols, start_date, end_date, frequency)

        if not df.empty:
            self._cache[cache_key] = df.copy()
            self._cache_timestamps[cache_key] = datetime.now()

        return df
