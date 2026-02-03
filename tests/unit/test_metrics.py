"""
Testes unitarios para metricas de performance
"""
import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analytics.metrics import MetricsCalculator, PerformanceMetrics, compare_strategies


class TestMetricsCalculator:
    """Testes para calculadora de metricas"""

    @pytest.fixture
    def sample_returns(self):
        """Cria retornos de exemplo"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='B')
        returns = pd.Series(
            np.random.randn(252) * 0.01 + 0.0003,  # Media positiva
            index=dates
        )
        return returns

    @pytest.fixture
    def calculator(self):
        return MetricsCalculator()

    def test_total_return(self, calculator, sample_returns):
        """Testa calculo de retorno total"""
        total = calculator.total_return(sample_returns)
        assert isinstance(total, float)

    def test_cagr(self, calculator, sample_returns):
        """Testa calculo de CAGR"""
        cagr = calculator.cagr(sample_returns)
        assert isinstance(cagr, float)

    def test_annualized_volatility(self, calculator, sample_returns):
        """Testa volatilidade anualizada"""
        vol = calculator.annualized_volatility(sample_returns)
        assert vol > 0

    def test_sharpe_ratio(self, calculator, sample_returns):
        """Testa Sharpe Ratio"""
        sharpe = calculator.sharpe_ratio(sample_returns)
        assert isinstance(sharpe, float)

    def test_sortino_ratio(self, calculator, sample_returns):
        """Testa Sortino Ratio"""
        sortino = calculator.sortino_ratio(sample_returns)
        assert isinstance(sortino, float)

    def test_calmar_ratio(self, calculator, sample_returns):
        """Testa Calmar Ratio"""
        calmar = calculator.calmar_ratio(sample_returns)
        assert isinstance(calmar, float)

    def test_var(self, calculator, sample_returns):
        """Testa Value at Risk"""
        var_95 = calculator.var(sample_returns, 0.95)
        var_99 = calculator.var(sample_returns, 0.99)

        assert var_95 < 0  # VaR e negativo
        assert var_99 < var_95  # 99% VaR mais extremo

    def test_cvar(self, calculator, sample_returns):
        """Testa Conditional VaR"""
        cvar = calculator.cvar(sample_returns, 0.95)
        var = calculator.var(sample_returns, 0.95)

        assert cvar <= var  # CVaR mais extremo que VaR

    def test_drawdown_metrics(self, calculator, sample_returns):
        """Testa metricas de drawdown"""
        dd_metrics = calculator.drawdown_metrics(sample_returns)

        assert 'max_drawdown' in dd_metrics
        assert 'max_duration' in dd_metrics
        assert 'ulcer_index' in dd_metrics
        assert dd_metrics['max_drawdown'] <= 0

    def test_calculate_all(self, calculator, sample_returns):
        """Testa calculo de todas as metricas"""
        metrics = calculator.calculate_all(sample_returns)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return != 0 or len(sample_returns) < 2
        assert metrics.annualized_volatility >= 0

    def test_monthly_metrics(self, calculator, sample_returns):
        """Testa metricas mensais"""
        monthly = calculator.monthly_metrics(sample_returns)

        assert 'positive_months' in monthly
        assert 'negative_months' in monthly
        assert 'win_rate' in monthly

    def test_rolling_metrics(self, calculator, sample_returns):
        """Testa metricas rolling"""
        rolling = calculator.rolling_metrics(sample_returns, window=63)

        assert 'return' in rolling.columns
        assert 'volatility' in rolling.columns
        assert 'sharpe' in rolling.columns

    def test_benchmark_metrics(self, calculator, sample_returns):
        """Testa metricas vs benchmark"""
        benchmark = sample_returns * 0.8 + 0.0001  # Benchmark correlacionado

        metrics = calculator.benchmark_metrics(sample_returns, benchmark)

        assert 'alpha' in metrics
        assert 'beta' in metrics
        assert 'r_squared' in metrics
        assert 'tracking_error' in metrics


class TestCompareStrategies:
    """Testes para comparacao de estrategias"""

    def test_compare(self):
        """Testa comparacao de multiplas estrategias"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='B')

        returns_dict = {
            'Strategy A': pd.Series(np.random.randn(252) * 0.01, index=dates),
            'Strategy B': pd.Series(np.random.randn(252) * 0.015, index=dates)
        }

        comparison = compare_strategies(returns_dict)

        assert len(comparison) == 2
        assert 'CAGR' in comparison.columns
        assert 'Sharpe' in comparison.columns


class TestPerformanceMetrics:
    """Testes para dataclass PerformanceMetrics"""

    def test_to_dict(self):
        """Testa conversao para dicionario"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='B')
        returns = pd.Series(np.random.randn(252) * 0.01, index=dates)

        calculator = MetricsCalculator()
        metrics = calculator.calculate_all(returns)

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert 'Retorno Total' in result
        assert 'Sharpe Ratio' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
