"""
Walk-Forward Analysis e Monte Carlo Simulation
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

from src.core.backtester import Backtester, BacktestResult
from src.strategies.base import BaseStrategy
from src.optimization.optimizer import GridSearchOptimizer, ParameterSpace, OptimizationResult


logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Janela de walk-forward"""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    optimal_params: Dict[str, Any]
    in_sample_result: Optional[BacktestResult]
    out_sample_result: Optional[BacktestResult]


@dataclass
class WalkForwardResult:
    """Resultado completo do walk-forward"""
    windows: List[WalkForwardWindow]
    combined_returns: pd.Series
    combined_metrics: Dict[str, float]
    in_sample_sharpe: float
    out_sample_sharpe: float
    efficiency_ratio: float
    robustness_score: float
    param_stability: Dict[str, float]


class WalkForwardAnalyzer:
    """
    Analise Walk-Forward.

    Otimiza em periodos in-sample e valida em periodos out-of-sample.
    """

    def __init__(
        self,
        train_period_days: int = 252,
        test_period_days: int = 63,
        step_days: Optional[int] = None,
        anchored: bool = False,
        optimization_objective: str = "sharpe_ratio"
    ):
        """
        Args:
            train_period_days: Dias de treinamento
            test_period_days: Dias de teste
            step_days: Dias de avanço (default = test_period_days)
            anchored: Se True, treino sempre comeca do inicio
            optimization_objective: Metrica a otimizar
        """
        self.train_period = train_period_days
        self.test_period = test_period_days
        self.step = step_days or test_period_days
        self.anchored = anchored
        self.objective = optimization_objective

    def generate_windows(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Gera janelas de train/test"""
        windows = []
        dates = data.index

        start_idx = self.train_period
        end_idx = len(dates)

        i = 0
        while start_idx + self.test_period <= end_idx:
            if self.anchored:
                train_start = dates[0]
            else:
                train_start = dates[start_idx - self.train_period]

            train_end = dates[start_idx - 1]
            test_start = dates[start_idx]
            test_end_idx = min(start_idx + self.test_period - 1, end_idx - 1)
            test_end = dates[test_end_idx]

            windows.append((train_start, train_end, test_start, test_end))

            start_idx += self.step
            i += 1

        return windows

    def run(
        self,
        backtester: Backtester,
        data: pd.DataFrame,
        param_spaces: List[ParameterSpace]
    ) -> WalkForwardResult:
        """
        Executa analise walk-forward.

        Args:
            backtester: Backtester configurado
            data: Dados completos
            param_spaces: Espacos de parametros

        Returns:
            WalkForwardResult
        """
        windows = self.generate_windows(data)
        logger.info(f"Walk-forward com {len(windows)} janelas")

        results = []
        all_test_returns = []
        in_sample_sharpes = []
        out_sample_sharpes = []
        param_history = []

        optimizer = GridSearchOptimizer(
            objective=self.objective,
            higher_is_better=True
        )

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Janela {i + 1}/{len(windows)}: "
                       f"Train {train_start.date()}-{train_end.date()}, "
                       f"Test {test_start.date()}-{test_end.date()}")

            # Dados de treino
            train_data = data[train_start:train_end]

            # Otimiza em treino
            opt_result = optimizer.optimize(
                backtester=backtester,
                data=train_data,
                param_spaces=param_spaces,
                max_combinations=100  # Limita para performance
            )

            optimal_params = opt_result.best_params
            param_history.append(optimal_params)

            # Backtest in-sample com parametros otimos
            strategy = backtester.strategy.copy_with_params(optimal_params)
            backtester.strategy = strategy
            in_sample_result = backtester.run(train_data)
            in_sample_sharpes.append(in_sample_result.sharpe_ratio)

            # Backtest out-of-sample
            test_data = data[test_start:test_end]
            out_sample_result = backtester.run(test_data)
            out_sample_sharpes.append(out_sample_result.sharpe_ratio)

            # Coleta retornos OOS
            all_test_returns.append(out_sample_result.returns)

            # Registra janela
            results.append(WalkForwardWindow(
                window_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                optimal_params=optimal_params,
                in_sample_result=in_sample_result,
                out_sample_result=out_sample_result
            ))

        # Combina retornos OOS
        combined_returns = pd.concat(all_test_returns)
        combined_returns = combined_returns[~combined_returns.index.duplicated(keep='first')]
        combined_returns = combined_returns.sort_index()

        # Calcula metricas combinadas
        total_return = (1 + combined_returns).prod() - 1
        ann_return = combined_returns.mean() * 252
        ann_vol = combined_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        combined_metrics = {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe
        }

        # Efficiency ratio (OOS / IS)
        avg_is = np.mean(in_sample_sharpes)
        avg_oos = np.mean(out_sample_sharpes)
        efficiency = avg_oos / avg_is if avg_is != 0 else 0

        # Robustness score (baseado em consistencia)
        positive_oos = sum(1 for s in out_sample_sharpes if s > 0)
        robustness = positive_oos / len(out_sample_sharpes) if out_sample_sharpes else 0

        # Estabilidade de parametros
        param_stability = self._calculate_param_stability(param_history)

        return WalkForwardResult(
            windows=results,
            combined_returns=combined_returns,
            combined_metrics=combined_metrics,
            in_sample_sharpe=avg_is,
            out_sample_sharpe=avg_oos,
            efficiency_ratio=efficiency,
            robustness_score=robustness,
            param_stability=param_stability
        )

    def _calculate_param_stability(
        self,
        param_history: List[Dict]
    ) -> Dict[str, float]:
        """Calcula estabilidade de cada parametro"""
        if not param_history:
            return {}

        stability = {}
        param_names = param_history[0].keys()

        for param in param_names:
            values = [h[param] for h in param_history if param in h]
            if len(values) > 1:
                # Coeficiente de variacao (menor = mais estavel)
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else 0
                stability[param] = 1 - min(cv, 1)  # Inverte para score
            else:
                stability[param] = 1.0

        return stability


@dataclass
class MonteCarloResult:
    """Resultado de simulacao Monte Carlo"""
    n_simulations: int
    mean_return: float
    median_return: float
    std_return: float
    var_95: float
    var_99: float
    probability_positive: float
    probability_target: float
    percentiles: Dict[int, float]
    simulation_paths: Optional[np.ndarray] = None


class MonteCarloSimulator:
    """
    Simulacao Monte Carlo para analise de robustez.
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        n_days: int = 252,
        random_seed: Optional[int] = None
    ):
        self.n_simulations = n_simulations
        self.n_days = n_days
        if random_seed:
            np.random.seed(random_seed)

    def simulate_returns(
        self,
        historical_returns: pd.Series,
        method: str = "bootstrap"
    ) -> MonteCarloResult:
        """
        Simula caminhos de retorno.

        Args:
            historical_returns: Retornos historicos
            method: 'bootstrap', 'parametric', ou 'block_bootstrap'

        Returns:
            MonteCarloResult
        """
        returns = historical_returns.dropna().values

        if method == "bootstrap":
            paths = self._bootstrap(returns)
        elif method == "parametric":
            paths = self._parametric(returns)
        elif method == "block_bootstrap":
            paths = self._block_bootstrap(returns)
        else:
            raise ValueError(f"Metodo desconhecido: {method}")

        # Calcula retornos cumulativos finais
        cumulative = np.cumprod(1 + paths, axis=1)
        final_returns = cumulative[:, -1] - 1

        # Estatisticas
        percentiles = {
            5: np.percentile(final_returns, 5),
            10: np.percentile(final_returns, 10),
            25: np.percentile(final_returns, 25),
            50: np.percentile(final_returns, 50),
            75: np.percentile(final_returns, 75),
            90: np.percentile(final_returns, 90),
            95: np.percentile(final_returns, 95)
        }

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            mean_return=np.mean(final_returns),
            median_return=np.median(final_returns),
            std_return=np.std(final_returns),
            var_95=np.percentile(final_returns, 5),
            var_99=np.percentile(final_returns, 1),
            probability_positive=np.mean(final_returns > 0),
            probability_target=np.mean(final_returns > 0.10),  # 10% target
            percentiles=percentiles,
            simulation_paths=cumulative
        )

    def _bootstrap(self, returns: np.ndarray) -> np.ndarray:
        """Bootstrap simples com reposicao"""
        paths = np.zeros((self.n_simulations, self.n_days))

        for i in range(self.n_simulations):
            indices = np.random.randint(0, len(returns), self.n_days)
            paths[i] = returns[indices]

        return paths

    def _parametric(self, returns: np.ndarray) -> np.ndarray:
        """Simulacao parametrica (normal)"""
        mean = np.mean(returns)
        std = np.std(returns)

        return np.random.normal(mean, std, (self.n_simulations, self.n_days))

    def _block_bootstrap(
        self,
        returns: np.ndarray,
        block_size: int = 21
    ) -> np.ndarray:
        """Bootstrap em blocos para preservar autocorrelacao"""
        n_blocks = int(np.ceil(self.n_days / block_size))
        paths = np.zeros((self.n_simulations, self.n_days))

        for i in range(self.n_simulations):
            path = []
            while len(path) < self.n_days:
                start = np.random.randint(0, len(returns) - block_size)
                block = returns[start:start + block_size]
                path.extend(block)

            paths[i] = np.array(path[:self.n_days])

        return paths

    def simulate_drawdown(
        self,
        historical_returns: pd.Series,
        n_simulations: int = 1000
    ) -> Dict:
        """Simula distribuicao de drawdowns"""
        returns = historical_returns.dropna().values
        max_drawdowns = []

        for _ in range(n_simulations):
            # Bootstrap de retornos
            indices = np.random.randint(0, len(returns), self.n_days)
            sim_returns = returns[indices]

            # Calcula drawdown
            cumulative = np.cumprod(1 + sim_returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak
            max_drawdowns.append(drawdown.min())

        return {
            'mean_max_dd': np.mean(max_drawdowns),
            'median_max_dd': np.median(max_drawdowns),
            'worst_dd_95': np.percentile(max_drawdowns, 5),
            'worst_dd_99': np.percentile(max_drawdowns, 1),
            'probability_dd_10': np.mean(np.array(max_drawdowns) < -0.10),
            'probability_dd_20': np.mean(np.array(max_drawdowns) < -0.20)
        }


def cross_validate_strategy(
    backtester: Backtester,
    data: pd.DataFrame,
    n_splits: int = 5,
    shuffle: bool = False
) -> Dict:
    """
    Validacao cruzada k-fold para estrategia.

    Args:
        backtester: Backtester configurado
        data: Dados completos
        n_splits: Numero de splits
        shuffle: Embaralhar dados

    Returns:
        Dicionario com resultados
    """
    n = len(data)
    fold_size = n // n_splits

    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    results = []

    for i in range(n_splits):
        # Define fold de teste
        test_start = i * fold_size
        test_end = test_start + fold_size if i < n_splits - 1 else n

        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

        # Dados de treino e teste
        train_data = data.iloc[train_indices].sort_index()
        test_data = data.iloc[test_indices].sort_index()

        # Backtest
        try:
            train_result = backtester.run(train_data)
            test_result = backtester.run(test_data)

            results.append({
                'fold': i,
                'train_sharpe': train_result.sharpe_ratio,
                'test_sharpe': test_result.sharpe_ratio,
                'train_cagr': train_result.cagr,
                'test_cagr': test_result.cagr,
                'train_max_dd': train_result.max_drawdown,
                'test_max_dd': test_result.max_drawdown
            })

        except Exception as e:
            logger.warning(f"Erro no fold {i}: {e}")

    if not results:
        return {'error': 'Nenhum fold completado'}

    df = pd.DataFrame(results)

    return {
        'n_splits': n_splits,
        'results': results,
        'mean_train_sharpe': df['train_sharpe'].mean(),
        'mean_test_sharpe': df['test_sharpe'].mean(),
        'std_test_sharpe': df['test_sharpe'].std(),
        'mean_train_cagr': df['train_cagr'].mean(),
        'mean_test_cagr': df['test_cagr'].mean(),
        'overfit_ratio': df['train_sharpe'].mean() / df['test_sharpe'].mean()
        if df['test_sharpe'].mean() != 0 else float('inf')
    }
