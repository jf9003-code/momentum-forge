"""
Testes unitarios para estrategias
"""
import pytest
from datetime import datetime
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies.base import BaseStrategy, StrategyMetadata
from src.strategies.momentum import MomentumStrategy, RiskAdjustedMomentumStrategy
from src.strategies.factory import StrategyFactory, StrategyType, get_strategy_presets
from config.settings import MomentumStrategyConfig


class TestMomentumStrategy:
    """Testes para estrategia de Momentum"""

    @pytest.fixture
    def sample_data(self):
        """Cria dados de exemplo para testes"""
        dates = pd.date_range('2020-01-01', periods=300, freq='B')
        np.random.seed(42)

        data = {}
        for symbol in ['SPY', 'QQQ', 'IWM']:
            prices = 100 * np.cumprod(1 + np.random.randn(300) * 0.01)
            data[(symbol, 'Close')] = prices
            data[(symbol, 'Volume')] = np.random.randint(1000000, 10000000, 300)

        df = pd.DataFrame(data, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        return df

    def test_strategy_creation(self):
        """Testa criacao de estrategia"""
        config = MomentumStrategyConfig(
            lookback_short=63,
            lookback_long=126,
            top_n=2
        )
        strategy = MomentumStrategy(config=config)

        assert strategy.config.lookback_short == 63
        assert strategy.config.top_n == 2

    def test_strategy_metadata(self):
        """Testa metadados da estrategia"""
        strategy = MomentumStrategy()

        metadata = strategy.metadata
        assert metadata.name is not None
        assert metadata.version is not None
        assert len(metadata.asset_classes) > 0

    def test_generate_signals(self, sample_data):
        """Testa geracao de sinais"""
        config = MomentumStrategyConfig(
            lookback_short=63,
            lookback_long=126,
            top_n=2,
            universe=['SPY', 'QQQ', 'IWM']
        )
        strategy = MomentumStrategy(config=config)

        current_date = sample_data.index[-1]
        signals = strategy.generate_signals(
            sample_data,
            current_date,
            ['SPY', 'QQQ', 'IWM']
        )

        # Deve retornar ate top_n sinais
        assert len(signals) <= config.top_n

        # Pesos devem somar 1
        if signals:
            total_weight = sum(s.weight for s in signals)
            assert abs(total_weight - 1.0) < 0.01

    def test_momentum_score_calculation(self, sample_data):
        """Testa calculo de score de momentum"""
        strategy = MomentumStrategy()

        prices = sample_data[('SPY', 'Close')]
        volume = sample_data[('SPY', 'Volume')]

        result = strategy.calculate_momentum_score(prices, volume)

        if result is not None:
            assert 'score' in result
            assert 'return_short' in result
            assert 'volatility' in result

    def test_copy_with_params(self):
        """Testa copia de estrategia com novos parametros"""
        strategy = MomentumStrategy()
        new_strategy = strategy.copy_with_params({'top_n': 5})

        assert new_strategy is not strategy


class TestRiskAdjustedMomentum:
    """Testes para estrategia Risk-Adjusted"""

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range('2020-01-01', periods=300, freq='B')
        np.random.seed(42)

        data = {}
        for symbol in ['SPY', 'QQQ', 'IWM']:
            prices = 100 * np.cumprod(1 + np.random.randn(300) * 0.01)
            data[(symbol, 'Close')] = prices
            data[(symbol, 'Volume')] = np.random.randint(1000000, 10000000, 300)

        df = pd.DataFrame(data, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        return df

    def test_risk_adjusted_signals(self, sample_data):
        """Testa sinais ajustados por volatilidade"""
        config = MomentumStrategyConfig(
            lookback_short=63,
            lookback_long=126,
            top_n=2,
            universe=['SPY', 'QQQ', 'IWM']
        )
        strategy = RiskAdjustedMomentumStrategy(config=config)

        signals = strategy.generate_signals(
            sample_data,
            sample_data.index[-1],
            ['SPY', 'QQQ', 'IWM']
        )

        if signals:
            # Verifica que pesos foram ajustados
            assert all('vol_adjusted' in s.metadata for s in signals)


class TestStrategyFactory:
    """Testes para factory de estrategias"""

    def test_create_momentum(self):
        """Testa criacao de estrategia momentum"""
        strategy = StrategyFactory.create(StrategyType.MOMENTUM)
        assert isinstance(strategy, MomentumStrategy)

    def test_create_risk_adjusted(self):
        """Testa criacao de estrategia risk-adjusted"""
        strategy = StrategyFactory.create(StrategyType.MOMENTUM_RISK_ADJUSTED)
        assert isinstance(strategy, RiskAdjustedMomentumStrategy)

    def test_create_with_params(self):
        """Testa criacao com parametros"""
        strategy = StrategyFactory.create(
            StrategyType.MOMENTUM,
            parameters={'top_n': 5}
        )
        assert strategy is not None

    def test_list_available(self):
        """Testa listagem de estrategias"""
        available = StrategyFactory.list_available()
        assert len(available) > 0
        assert all('name' in s for s in available)


class TestStrategyPresets:
    """Testes para presets de estrategias"""

    def test_get_presets(self):
        """Testa obtencao de presets"""
        presets = get_strategy_presets()
        assert len(presets) > 0
        assert 'conservative' in presets
        assert 'aggressive' in presets

    def test_preset_structure(self):
        """Testa estrutura dos presets"""
        presets = get_strategy_presets()

        for name, preset in presets.items():
            assert 'strategy_type' in preset
            assert 'parameters' in preset
            assert 'description' in preset


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
