"""
Metricas de Performance Institucionais
"""
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Metricas completas de performance"""
    # Retornos
    total_return: float
    cagr: float
    mtd_return: float
    ytd_return: float
    annualized_return: float

    # Volatilidade
    annualized_volatility: float
    downside_volatility: float
    upside_volatility: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    avg_drawdown_duration: float
    ulcer_index: float
    pain_index: float

    # Tail Risk
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    skewness: float
    kurtosis: float

    # Benchmark
    alpha: float
    beta: float
    r_squared: float
    tracking_error: float
    up_capture: float
    down_capture: float

    # Consistencia
    positive_months: int
    negative_months: int
    best_month: float
    worst_month: float
    avg_monthly_return: float
    monthly_win_rate: float

    # Periodos
    start_date: str
    end_date: str
    total_days: int

    def to_dict(self) -> Dict:
        """Converte para dicionario"""
        return {
            'Retorno Total': f"{self.total_return:.2%}",
            'CAGR': f"{self.cagr:.2%}",
            'Volatilidade': f"{self.annualized_volatility:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{self.sortino_ratio:.2f}",
            'Max Drawdown': f"{self.max_drawdown:.2%}",
            'VaR 95%': f"{self.var_95:.2%}",
            'Alpha': f"{self.alpha:.2%}",
            'Beta': f"{self.beta:.2f}",
            'Win Rate Mensal': f"{self.monthly_win_rate:.1%}"
        }


class MetricsCalculator:
    """Calculadora de metricas de performance"""

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        mar: float = 0.0  # Minimum Acceptable Return
    ):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.mar = mar
        self._rf_daily = risk_free_rate / periods_per_year

    def calculate_all(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """
        Calcula todas as metricas de performance.

        Args:
            returns: Serie de retornos diarios
            benchmark_returns: Retornos do benchmark (opcional)

        Returns:
            PerformanceMetrics com todas as metricas
        """
        if len(returns) < 2:
            return self._empty_metrics()

        returns = returns.dropna()

        # Retornos
        total_return = self.total_return(returns)
        cagr = self.cagr(returns)
        ann_return = self.annualized_return(returns)

        # MTD e YTD
        mtd = self.mtd_return(returns)
        ytd = self.ytd_return(returns)

        # Volatilidade
        ann_vol = self.annualized_volatility(returns)
        down_vol = self.downside_volatility(returns)
        up_vol = self.upside_volatility(returns)

        # Risk-adjusted
        sharpe = self.sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns)
        calmar = self.calmar_ratio(returns)
        omega = self.omega_ratio(returns)

        # Drawdown
        dd_metrics = self.drawdown_metrics(returns)

        # Tail risk
        var_95 = self.var(returns, 0.95)
        var_99 = self.var(returns, 0.99)
        cvar_95 = self.cvar(returns, 0.95)
        cvar_99 = self.cvar(returns, 0.99)
        skew = returns.skew()
        kurt = returns.kurtosis()

        # Benchmark metrics
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            bench_metrics = self.benchmark_metrics(returns, benchmark_returns)
            alpha = bench_metrics['alpha']
            beta = bench_metrics['beta']
            r_sq = bench_metrics['r_squared']
            te = bench_metrics['tracking_error']
            ir = bench_metrics['information_ratio']
            up_cap = bench_metrics['up_capture']
            down_cap = bench_metrics['down_capture']
        else:
            alpha, beta, r_sq, te, ir = 0, 1, 0, 0, 0
            up_cap, down_cap = 1, 1

        # Consistencia mensal
        monthly = self.monthly_metrics(returns)

        return PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            mtd_return=mtd,
            ytd_return=ytd,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            downside_volatility=down_vol,
            upside_volatility=up_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            information_ratio=ir,
            max_drawdown=dd_metrics['max_drawdown'],
            max_drawdown_duration=dd_metrics['max_duration'],
            avg_drawdown=dd_metrics['avg_drawdown'],
            avg_drawdown_duration=dd_metrics['avg_duration'],
            ulcer_index=dd_metrics['ulcer_index'],
            pain_index=dd_metrics['pain_index'],
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            skewness=skew,
            kurtosis=kurt,
            alpha=alpha,
            beta=beta,
            r_squared=r_sq,
            tracking_error=te,
            up_capture=up_cap,
            down_capture=down_cap,
            positive_months=monthly['positive_months'],
            negative_months=monthly['negative_months'],
            best_month=monthly['best_month'],
            worst_month=monthly['worst_month'],
            avg_monthly_return=monthly['avg_monthly'],
            monthly_win_rate=monthly['win_rate'],
            start_date=str(returns.index.min().date()),
            end_date=str(returns.index.max().date()),
            total_days=len(returns)
        )

    def total_return(self, returns: pd.Series) -> float:
        """Retorno total acumulado"""
        return (1 + returns).prod() - 1

    def cagr(self, returns: pd.Series) -> float:
        """Compound Annual Growth Rate"""
        total = self.total_return(returns)
        years = len(returns) / self.periods_per_year

        if years <= 0:
            return 0.0

        return (1 + total) ** (1 / years) - 1

    def annualized_return(self, returns: pd.Series) -> float:
        """Retorno anualizado"""
        return returns.mean() * self.periods_per_year

    def mtd_return(self, returns: pd.Series) -> float:
        """Month-to-date return"""
        if len(returns) == 0:
            return 0.0

        last_date = returns.index.max()
        month_start = last_date.replace(day=1)
        mtd_returns = returns[returns.index >= month_start]

        return self.total_return(mtd_returns)

    def ytd_return(self, returns: pd.Series) -> float:
        """Year-to-date return"""
        if len(returns) == 0:
            return 0.0

        last_date = returns.index.max()
        year_start = last_date.replace(month=1, day=1)
        ytd_returns = returns[returns.index >= year_start]

        return self.total_return(ytd_returns)

    def annualized_volatility(self, returns: pd.Series) -> float:
        """Volatilidade anualizada"""
        return returns.std() * np.sqrt(self.periods_per_year)

    def downside_volatility(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Volatilidade downside"""
        downside = returns[returns < threshold]
        if len(downside) < 2:
            return 0.0
        return downside.std() * np.sqrt(self.periods_per_year)

    def upside_volatility(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Volatilidade upside"""
        upside = returns[returns > threshold]
        if len(upside) < 2:
            return 0.0
        return upside.std() * np.sqrt(self.periods_per_year)

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Sharpe Ratio"""
        excess = returns - self._rf_daily
        if returns.std() == 0:
            return 0.0
        return (excess.mean() / returns.std()) * np.sqrt(self.periods_per_year)

    def sortino_ratio(self, returns: pd.Series) -> float:
        """Sortino Ratio"""
        excess_return = returns.mean() - self._rf_daily
        downside_vol = self.downside_volatility(returns, self._rf_daily)

        if downside_vol == 0:
            return 0.0

        return (excess_return * self.periods_per_year) / downside_vol

    def calmar_ratio(self, returns: pd.Series) -> float:
        """Calmar Ratio (CAGR / Max Drawdown)"""
        dd_metrics = self.drawdown_metrics(returns)
        max_dd = abs(dd_metrics['max_drawdown'])

        if max_dd == 0:
            return 0.0

        return self.cagr(returns) / max_dd

    def omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Omega Ratio"""
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())

        if losses == 0:
            return np.inf if gains > 0 else 0.0

        return gains / losses

    def var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Value at Risk (historico)"""
        return np.percentile(returns, (1 - confidence) * 100)

    def cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall)"""
        var = self.var(returns, confidence)
        tail = returns[returns <= var]
        return tail.mean() if len(tail) > 0 else var

    def drawdown_metrics(self, returns: pd.Series) -> Dict:
        """Calcula metricas de drawdown"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak

        # Max drawdown
        max_dd = drawdown.min()

        # Duracao do drawdown
        is_dd = drawdown < 0
        dd_groups = (~is_dd).cumsum()

        if is_dd.any():
            dd_durations = []
            for group_id in dd_groups[is_dd].unique():
                duration = (dd_groups == group_id).sum()
                dd_durations.append(duration)

            max_duration = max(dd_durations) if dd_durations else 0
            avg_duration = np.mean(dd_durations) if dd_durations else 0
        else:
            max_duration = 0
            avg_duration = 0

        # Average drawdown
        avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0

        # Ulcer Index
        ulcer = np.sqrt((drawdown ** 2).mean())

        # Pain Index
        pain = abs(drawdown).mean()

        return {
            'max_drawdown': max_dd,
            'max_duration': int(max_duration),
            'avg_drawdown': avg_dd,
            'avg_duration': avg_duration,
            'ulcer_index': ulcer,
            'pain_index': pain,
            'drawdown_series': drawdown
        }

    def benchmark_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict:
        """Calcula metricas relativas ao benchmark"""
        # Alinha series
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 30:
            return {
                'alpha': 0, 'beta': 1, 'r_squared': 0,
                'tracking_error': 0, 'information_ratio': 0,
                'up_capture': 1, 'down_capture': 1
            }

        port = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]

        # Regressao
        slope, intercept, r_value, _, _ = stats.linregress(bench, port)
        beta = slope
        alpha = intercept * self.periods_per_year
        r_squared = r_value ** 2

        # Tracking error e Information ratio
        active = port - bench
        tracking_error = active.std() * np.sqrt(self.periods_per_year)
        info_ratio = (active.mean() * self.periods_per_year) / tracking_error if tracking_error > 0 else 0

        # Capture ratios
        up_periods = bench > 0
        down_periods = bench < 0

        if up_periods.sum() > 0:
            up_capture = port[up_periods].mean() / bench[up_periods].mean()
        else:
            up_capture = 1.0

        if down_periods.sum() > 0:
            down_capture = port[down_periods].mean() / bench[down_periods].mean()
        else:
            down_capture = 1.0

        return {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'tracking_error': tracking_error,
            'information_ratio': info_ratio,
            'up_capture': up_capture,
            'down_capture': down_capture
        }

    def monthly_metrics(self, returns: pd.Series) -> Dict:
        """Calcula metricas mensais"""
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

        positive = (monthly > 0).sum()
        negative = (monthly < 0).sum()

        return {
            'positive_months': positive,
            'negative_months': negative,
            'best_month': monthly.max() if len(monthly) > 0 else 0,
            'worst_month': monthly.min() if len(monthly) > 0 else 0,
            'avg_monthly': monthly.mean() if len(monthly) > 0 else 0,
            'win_rate': positive / len(monthly) if len(monthly) > 0 else 0
        }

    def rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 252
    ) -> pd.DataFrame:
        """Calcula metricas rolling"""
        rolling_return = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        )
        rolling_vol = returns.rolling(window).std() * np.sqrt(self.periods_per_year)
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: self.sharpe_ratio(x)
        )

        cumulative = (1 + returns).cumprod()
        peak = cumulative.rolling(window).max()
        rolling_dd = (cumulative - peak) / peak

        return pd.DataFrame({
            'return': rolling_return,
            'volatility': rolling_vol,
            'sharpe': rolling_sharpe,
            'drawdown': rolling_dd
        })

    def _empty_metrics(self) -> PerformanceMetrics:
        """Retorna metricas vazias"""
        return PerformanceMetrics(
            total_return=0, cagr=0, mtd_return=0, ytd_return=0,
            annualized_return=0, annualized_volatility=0,
            downside_volatility=0, upside_volatility=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            omega_ratio=0, information_ratio=0,
            max_drawdown=0, max_drawdown_duration=0,
            avg_drawdown=0, avg_drawdown_duration=0,
            ulcer_index=0, pain_index=0,
            var_95=0, var_99=0, cvar_95=0, cvar_99=0,
            skewness=0, kurtosis=0,
            alpha=0, beta=1, r_squared=0, tracking_error=0,
            up_capture=1, down_capture=1,
            positive_months=0, negative_months=0,
            best_month=0, worst_month=0,
            avg_monthly_return=0, monthly_win_rate=0,
            start_date='', end_date='', total_days=0
        )


def compare_strategies(
    returns_dict: Dict[str, pd.Series],
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Compara metricas de multiplas estrategias.

    Args:
        returns_dict: Dicionario {nome: retornos}
        benchmark_returns: Retornos do benchmark
        risk_free_rate: Taxa livre de risco

    Returns:
        DataFrame com comparacao
    """
    calculator = MetricsCalculator(risk_free_rate=risk_free_rate)
    results = []

    for name, returns in returns_dict.items():
        metrics = calculator.calculate_all(returns, benchmark_returns)
        results.append({
            'Strategy': name,
            'CAGR': metrics.cagr,
            'Volatility': metrics.annualized_volatility,
            'Sharpe': metrics.sharpe_ratio,
            'Sortino': metrics.sortino_ratio,
            'Max DD': metrics.max_drawdown,
            'Calmar': metrics.calmar_ratio,
            'Alpha': metrics.alpha,
            'Beta': metrics.beta,
            'Win Rate': metrics.monthly_win_rate
        })

    return pd.DataFrame(results).set_index('Strategy')
