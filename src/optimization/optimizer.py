"""
Otimizacao de Estrategias
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from src.core.backtester import Backtester, BacktestResult
from src.strategies.base import BaseStrategy


logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """Espaco de parametros para otimizacao"""
    name: str
    values: List[Any]
    param_type: str = "discrete"  # discrete, continuous, integer

    def sample(self, n: int = 1) -> List[Any]:
        """Amostra valores do espaco"""
        if self.param_type == "discrete":
            return list(np.random.choice(self.values, size=min(n, len(self.values)), replace=False))
        elif self.param_type == "continuous":
            low, high = min(self.values), max(self.values)
            return list(np.random.uniform(low, high, n))
        elif self.param_type == "integer":
            low, high = min(self.values), max(self.values)
            return list(np.random.randint(low, high + 1, n))
        return self.values[:n]


@dataclass
class OptimizationResult:
    """Resultado da otimizacao"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict]
    objective: str
    optimization_time: float
    total_combinations: int
    evaluated_combinations: int


class GridSearchOptimizer:
    """Otimizador por busca em grade"""

    def __init__(
        self,
        objective: str = "sharpe_ratio",
        higher_is_better: bool = True
    ):
        self.objective = objective
        self.higher_is_better = higher_is_better

    def optimize(
        self,
        backtester: Backtester,
        data: pd.DataFrame,
        param_spaces: List[ParameterSpace],
        max_combinations: Optional[int] = None
    ) -> OptimizationResult:
        """
        Executa otimizacao por grid search.

        Args:
            backtester: Backtester configurado
            data: Dados para backtest
            param_spaces: Espacos de parametros
            max_combinations: Limite de combinacoes

        Returns:
            OptimizationResult
        """
        start_time = datetime.now()

        # Gera combinacoes
        param_names = [p.name for p in param_spaces]
        param_values = [p.values for p in param_spaces]
        combinations = list(product(*param_values))

        if max_combinations and len(combinations) > max_combinations:
            indices = np.random.choice(len(combinations), max_combinations, replace=False)
            combinations = [combinations[i] for i in indices]

        logger.info(f"Avaliando {len(combinations)} combinacoes...")

        results = []
        best_score = float('-inf') if self.higher_is_better else float('inf')
        best_params = None

        for combo in combinations:
            params = dict(zip(param_names, combo))

            try:
                # Cria estrategia com parametros
                strategy = backtester.strategy.copy_with_params(params)
                backtester.strategy = strategy

                # Executa backtest
                result = backtester.run(data)

                # Obtem score
                score = getattr(result, self.objective, 0)

                results.append({
                    'params': params,
                    'score': score,
                    'sharpe': result.sharpe_ratio,
                    'cagr': result.cagr,
                    'max_dd': result.max_drawdown
                })

                # Atualiza melhor
                if self.higher_is_better:
                    if score > best_score:
                        best_score = score
                        best_params = params
                else:
                    if score < best_score:
                        best_score = score
                        best_params = params

            except Exception as e:
                logger.warning(f"Erro com parametros {params}: {e}")
                continue

        elapsed = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=results,
            objective=self.objective,
            optimization_time=elapsed,
            total_combinations=len(list(product(*param_values))),
            evaluated_combinations=len(results)
        )


class RandomSearchOptimizer:
    """Otimizador por busca aleatoria"""

    def __init__(
        self,
        objective: str = "sharpe_ratio",
        higher_is_better: bool = True,
        n_iterations: int = 100
    ):
        self.objective = objective
        self.higher_is_better = higher_is_better
        self.n_iterations = n_iterations

    def optimize(
        self,
        backtester: Backtester,
        data: pd.DataFrame,
        param_spaces: List[ParameterSpace]
    ) -> OptimizationResult:
        """Executa otimizacao por random search"""
        start_time = datetime.now()

        results = []
        best_score = float('-inf') if self.higher_is_better else float('inf')
        best_params = None

        for i in range(self.n_iterations):
            # Amostra parametros
            params = {p.name: p.sample(1)[0] for p in param_spaces}

            try:
                strategy = backtester.strategy.copy_with_params(params)
                backtester.strategy = strategy

                result = backtester.run(data)
                score = getattr(result, self.objective, 0)

                results.append({
                    'params': params,
                    'score': score,
                    'sharpe': result.sharpe_ratio,
                    'cagr': result.cagr,
                    'max_dd': result.max_drawdown
                })

                if self.higher_is_better:
                    if score > best_score:
                        best_score = score
                        best_params = params
                else:
                    if score < best_score:
                        best_score = score
                        best_params = params

                if (i + 1) % 10 == 0:
                    logger.info(f"Iteracao {i + 1}/{self.n_iterations}, Melhor score: {best_score:.4f}")

            except Exception as e:
                logger.warning(f"Erro na iteracao {i}: {e}")
                continue

        elapsed = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=results,
            objective=self.objective,
            optimization_time=elapsed,
            total_combinations=self.n_iterations,
            evaluated_combinations=len(results)
        )


class BayesianOptimizer:
    """
    Otimizador Bayesiano (simplificado).

    Usa aquisicao baseada em UCB (Upper Confidence Bound).
    """

    def __init__(
        self,
        objective: str = "sharpe_ratio",
        n_iterations: int = 50,
        n_initial: int = 10,
        exploration_weight: float = 2.0
    ):
        self.objective = objective
        self.n_iterations = n_iterations
        self.n_initial = n_initial
        self.exploration_weight = exploration_weight

    def optimize(
        self,
        backtester: Backtester,
        data: pd.DataFrame,
        param_spaces: List[ParameterSpace]
    ) -> OptimizationResult:
        """Executa otimizacao bayesiana simplificada"""
        start_time = datetime.now()

        # Fase inicial: amostragem aleatoria
        results = []
        param_history = []
        score_history = []

        for i in range(self.n_initial):
            params = {p.name: p.sample(1)[0] for p in param_spaces}

            try:
                strategy = backtester.strategy.copy_with_params(params)
                backtester.strategy = strategy
                result = backtester.run(data)
                score = getattr(result, self.objective, 0)

                results.append({
                    'params': params,
                    'score': score,
                    'iteration': i,
                    'phase': 'initial'
                })

                param_history.append(list(params.values()))
                score_history.append(score)

            except Exception:
                continue

        # Fase de otimizacao
        for i in range(self.n_initial, self.n_iterations):
            # Seleciona proximo ponto usando UCB simplificado
            best_params = self._select_next_point(
                param_spaces, param_history, score_history
            )

            try:
                strategy = backtester.strategy.copy_with_params(best_params)
                backtester.strategy = strategy
                result = backtester.run(data)
                score = getattr(result, self.objective, 0)

                results.append({
                    'params': best_params,
                    'score': score,
                    'iteration': i,
                    'phase': 'optimization'
                })

                param_history.append(list(best_params.values()))
                score_history.append(score)

            except Exception:
                continue

        # Encontra melhor
        if results:
            best_result = max(results, key=lambda x: x['score'])
            best_params = best_result['params']
            best_score = best_result['score']
        else:
            best_params = {}
            best_score = 0

        elapsed = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results,
            objective=self.objective,
            optimization_time=elapsed,
            total_combinations=self.n_iterations,
            evaluated_combinations=len(results)
        )

    def _select_next_point(
        self,
        param_spaces: List[ParameterSpace],
        param_history: List[List],
        score_history: List[float]
    ) -> Dict:
        """Seleciona proximo ponto usando UCB simplificado"""
        if not score_history:
            return {p.name: p.sample(1)[0] for p in param_spaces}

        # Amostra candidatos
        n_candidates = 100
        candidates = []

        for _ in range(n_candidates):
            params = {p.name: p.sample(1)[0] for p in param_spaces}
            candidates.append(params)

        # Calcula UCB para cada candidato
        mean_score = np.mean(score_history)
        std_score = np.std(score_history) + 1e-6

        best_ucb = float('-inf')
        best_candidate = candidates[0]

        for candidate in candidates:
            # Distancia minima para pontos ja avaliados
            candidate_values = np.array(list(candidate.values()))
            min_dist = float('inf')

            for history_values in param_history:
                dist = np.linalg.norm(candidate_values - np.array(history_values))
                min_dist = min(min_dist, dist)

            # UCB = mean + exploration * uncertainty
            ucb = mean_score + self.exploration_weight * (min_dist / (std_score + 1))

            if ucb > best_ucb:
                best_ucb = ucb
                best_candidate = candidate

        return best_candidate


def sensitivity_analysis(
    backtester: Backtester,
    data: pd.DataFrame,
    base_params: Dict[str, Any],
    param_to_vary: str,
    values: List[Any],
    objective: str = "sharpe_ratio"
) -> pd.DataFrame:
    """
    Analise de sensibilidade para um parametro.

    Args:
        backtester: Backtester configurado
        data: Dados para backtest
        base_params: Parametros base
        param_to_vary: Parametro a variar
        values: Valores a testar
        objective: Metrica objetivo

    Returns:
        DataFrame com resultados
    """
    results = []

    for value in values:
        params = {**base_params, param_to_vary: value}

        try:
            strategy = backtester.strategy.copy_with_params(params)
            backtester.strategy = strategy
            result = backtester.run(data)

            results.append({
                param_to_vary: value,
                'sharpe_ratio': result.sharpe_ratio,
                'cagr': result.cagr,
                'max_drawdown': result.max_drawdown,
                'sortino_ratio': result.sortino_ratio,
                objective: getattr(result, objective, 0)
            })

        except Exception as e:
            logger.warning(f"Erro com {param_to_vary}={value}: {e}")

    return pd.DataFrame(results)
