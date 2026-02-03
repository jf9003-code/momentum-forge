"""
Atribuicao de Performance
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class AttributionResult:
    """Resultado da atribuicao de performance"""
    total_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    active_return: float
    tracking_error: float
    information_ratio: float
    by_sector: Dict[str, Dict[str, float]]
    by_asset: Dict[str, Dict[str, float]]


class BrinsonAttribution:
    """
    Atribuicao Brinson-Fachler.

    Decompoe retornos ativos em:
    - Efeito Alocacao: impacto de pesos diferentes do benchmark
    - Efeito Selecao: impacto de retornos diferentes dentro de cada setor
    - Efeito Interacao: efeito combinado
    """

    def __init__(
        self,
        portfolio_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        sector_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            portfolio_weights: Pesos do portfolio por ativo/periodo
            portfolio_returns: Retornos do portfolio por ativo/periodo
            benchmark_weights: Pesos do benchmark
            benchmark_returns: Retornos do benchmark
            sector_mapping: Mapeamento ativo -> setor
        """
        self.port_weights = portfolio_weights
        self.port_returns = portfolio_returns
        self.bench_weights = benchmark_weights
        self.bench_returns = benchmark_returns
        self.sector_mapping = sector_mapping or {}

    def calculate(self) -> AttributionResult:
        """Calcula atribuicao de performance"""
        # Alinha dados
        common_dates = (
            self.port_weights.index
            .intersection(self.port_returns.index)
            .intersection(self.bench_weights.index)
            .intersection(self.bench_returns.index)
        )

        common_assets = (
            set(self.port_weights.columns)
            .intersection(self.port_returns.columns)
            .intersection(self.bench_weights.columns)
            .intersection(self.bench_returns.columns)
        )

        if len(common_dates) == 0 or len(common_assets) == 0:
            return self._empty_result()

        # Filtra dados
        pw = self.port_weights.loc[common_dates, list(common_assets)]
        pr = self.port_returns.loc[common_dates, list(common_assets)]
        bw = self.bench_weights.loc[common_dates, list(common_assets)]
        br = self.bench_returns.loc[common_dates, list(common_assets)]

        # Retornos totais
        port_total = (pw * pr).sum(axis=1)
        bench_total = (bw * br).sum(axis=1)

        total_port_return = (1 + port_total).prod() - 1
        total_bench_return = (1 + bench_total).prod() - 1
        active_return = total_port_return - total_bench_return

        # Atribuicao por ativo
        by_asset = {}
        total_allocation = 0
        total_selection = 0
        total_interaction = 0

        for asset in common_assets:
            # Medias dos periodos
            wp = pw[asset].mean()
            wb = bw[asset].mean()
            rp = pr[asset].mean() * 252
            rb = br[asset].mean() * 252
            rb_total = bench_total.mean() * 252

            # Efeitos Brinson
            allocation = (wp - wb) * (rb - rb_total)
            selection = wb * (rp - rb)
            interaction = (wp - wb) * (rp - rb)

            by_asset[asset] = {
                'portfolio_weight': wp,
                'benchmark_weight': wb,
                'portfolio_return': rp,
                'benchmark_return': rb,
                'allocation_effect': allocation,
                'selection_effect': selection,
                'interaction_effect': interaction,
                'total_effect': allocation + selection + interaction
            }

            total_allocation += allocation
            total_selection += selection
            total_interaction += interaction

        # Atribuicao por setor
        by_sector = {}
        if self.sector_mapping:
            sectors = set(self.sector_mapping.values())
            for sector in sectors:
                sector_assets = [a for a in common_assets if self.sector_mapping.get(a) == sector]
                if not sector_assets:
                    continue

                sector_allocation = sum(by_asset[a]['allocation_effect'] for a in sector_assets)
                sector_selection = sum(by_asset[a]['selection_effect'] for a in sector_assets)
                sector_interaction = sum(by_asset[a]['interaction_effect'] for a in sector_assets)

                by_sector[sector] = {
                    'allocation_effect': sector_allocation,
                    'selection_effect': sector_selection,
                    'interaction_effect': sector_interaction,
                    'total_effect': sector_allocation + sector_selection + sector_interaction
                }

        # Tracking error e IR
        active_returns = port_total - bench_total
        tracking_error = active_returns.std() * np.sqrt(252)
        info_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0

        return AttributionResult(
            total_return=total_port_return,
            allocation_effect=total_allocation,
            selection_effect=total_selection,
            interaction_effect=total_interaction,
            active_return=active_return,
            tracking_error=tracking_error,
            information_ratio=info_ratio,
            by_sector=by_sector,
            by_asset=by_asset
        )

    def _empty_result(self) -> AttributionResult:
        return AttributionResult(
            total_return=0,
            allocation_effect=0,
            selection_effect=0,
            interaction_effect=0,
            active_return=0,
            tracking_error=0,
            information_ratio=0,
            by_sector={},
            by_asset={}
        )


class FactorAttribution:
    """
    Atribuicao baseada em fatores (Fama-French, etc).

    Decompoe retornos em exposicoes a fatores de risco.
    """

    def __init__(self, factors: Optional[Dict[str, pd.Series]] = None):
        """
        Args:
            factors: Dicionario de fatores {nome: retornos}
        """
        self.factors = factors or {}

    def set_factors(self, factors: Dict[str, pd.Series]) -> None:
        """Define fatores"""
        self.factors = factors

    def add_factor(self, name: str, returns: pd.Series) -> None:
        """Adiciona um fator"""
        self.factors[name] = returns

    def calculate(
        self,
        portfolio_returns: pd.Series,
        risk_free_rate: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calcula atribuicao por fatores.

        Args:
            portfolio_returns: Retornos do portfolio
            risk_free_rate: Taxa livre de risco (opcional)

        Returns:
            Dicionario com betas, alpha e estatisticas
        """
        if not self.factors:
            return {'error': 'Nenhum fator definido'}

        # Constroi DataFrame de fatores
        factor_df = pd.DataFrame(self.factors)

        # Alinha com retornos do portfolio
        aligned = pd.concat([portfolio_returns, factor_df], axis=1).dropna()

        if len(aligned) < 30:
            return {'error': 'Dados insuficientes'}

        y = aligned.iloc[:, 0]
        X = aligned.iloc[:, 1:]

        # Subtrai risk-free se fornecido
        if risk_free_rate is not None:
            rf_aligned = risk_free_rate.reindex(aligned.index).fillna(0)
            y = y - rf_aligned

        # Adiciona constante para alpha
        X_with_const = X.copy()
        X_with_const['const'] = 1

        # Regressao OLS
        try:
            # Resolve: (X'X)^-1 X'y
            XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
            betas = XtX_inv @ X_with_const.T @ y

            # Residuos
            y_pred = X_with_const @ betas
            residuals = y - y_pred

            # R-squared
            ss_res = (residuals ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Erro padrao
            n = len(y)
            k = len(X_with_const.columns)
            mse = ss_res / (n - k)
            se = np.sqrt(np.diag(XtX_inv) * mse)

            # T-stats
            t_stats = betas / se

            # P-values
            p_values = 2 * (1 - stats.t.cdf(abs(t_stats), n - k))

        except np.linalg.LinAlgError:
            return {'error': 'Erro na regressao (matriz singular)'}

        # Monta resultado
        result = {
            'alpha': betas[-1] * 252,  # Anualizado
            'alpha_t_stat': t_stats[-1],
            'alpha_p_value': p_values[-1],
            'r_squared': r_squared,
            'residual_vol': residuals.std() * np.sqrt(252),
            'factors': {}
        }

        for i, factor_name in enumerate(X.columns):
            result['factors'][factor_name] = {
                'beta': betas[i],
                't_stat': t_stats[i],
                'p_value': p_values[i],
                'contribution': betas[i] * X[factor_name].mean() * 252
            }

        return result


class RiskAttribution:
    """Atribuicao de risco do portfolio"""

    def __init__(self, returns: pd.DataFrame, weights: pd.Series):
        """
        Args:
            returns: Retornos de cada ativo
            weights: Pesos do portfolio
        """
        self.returns = returns
        self.weights = weights

    def calculate(self) -> Dict:
        """Calcula atribuicao de risco"""
        # Alinha dados
        common = self.returns.columns.intersection(self.weights.index)
        returns = self.returns[common]
        weights = self.weights[common]

        # Normaliza pesos
        weights = weights / weights.sum()

        # Matriz de covariancia
        cov_matrix = returns.cov() * 252

        # Variancia do portfolio
        port_var = weights @ cov_matrix @ weights
        port_vol = np.sqrt(port_var)

        # Contribuicao marginal ao risco (MCR)
        mcr = (cov_matrix @ weights) / port_vol

        # Contribuicao de risco (CR)
        cr = weights * mcr

        # Contribuicao percentual
        cr_pct = cr / port_vol

        # Correlacao media
        corr_matrix = returns.corr()
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

        result = {
            'portfolio_volatility': port_vol,
            'portfolio_variance': port_var,
            'average_correlation': avg_corr,
            'diversification_ratio': (weights * returns.std() * np.sqrt(252)).sum() / port_vol,
            'assets': {}
        }

        for asset in common:
            result['assets'][asset] = {
                'weight': weights[asset],
                'volatility': returns[asset].std() * np.sqrt(252),
                'marginal_contribution': mcr[asset],
                'risk_contribution': cr[asset],
                'risk_contribution_pct': cr_pct[asset]
            }

        return result


def calculate_turnover(
    weights_history: pd.DataFrame,
    method: str = 'absolute'
) -> pd.Series:
    """
    Calcula turnover do portfolio.

    Args:
        weights_history: DataFrame com historico de pesos
        method: 'absolute' ou 'one_sided'

    Returns:
        Serie com turnover por periodo
    """
    weight_changes = weights_history.diff().abs()

    if method == 'absolute':
        turnover = weight_changes.sum(axis=1)
    else:  # one_sided
        turnover = weight_changes.sum(axis=1) / 2

    return turnover


def calculate_concentration(weights: pd.Series) -> Dict:
    """
    Calcula metricas de concentracao.

    Args:
        weights: Pesos do portfolio

    Returns:
        Dicionario com metricas de concentracao
    """
    weights = weights[weights > 0]  # Remove pesos zero
    weights = weights / weights.sum()  # Normaliza

    # HHI - Herfindahl-Hirschman Index
    hhi = (weights ** 2).sum()

    # Numero efetivo de ativos
    effective_n = 1 / hhi if hhi > 0 else len(weights)

    # Top N concentracao
    sorted_weights = weights.sort_values(ascending=False)
    top_1 = sorted_weights.iloc[0] if len(sorted_weights) > 0 else 0
    top_3 = sorted_weights.iloc[:3].sum() if len(sorted_weights) >= 3 else sorted_weights.sum()
    top_5 = sorted_weights.iloc[:5].sum() if len(sorted_weights) >= 5 else sorted_weights.sum()

    # Gini coefficient
    n = len(weights)
    if n > 1:
        sorted_w = np.sort(weights.values)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_w) - (n + 1) * np.sum(sorted_w)) / (n * np.sum(sorted_w))
    else:
        gini = 0

    return {
        'hhi': hhi,
        'effective_n': effective_n,
        'actual_n': len(weights),
        'top_1_weight': top_1,
        'top_3_weight': top_3,
        'top_5_weight': top_5,
        'gini_coefficient': gini
    }
