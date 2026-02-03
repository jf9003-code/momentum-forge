"""
Geracao de Relatorios
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

from src.analytics.metrics import MetricsCalculator, PerformanceMetrics


@dataclass
class ReportSection:
    """Secao de um relatorio"""
    title: str
    content: Dict[str, Any]
    charts: Optional[List[Dict]] = None


class ReportGenerator:
    """Gerador de relatorios de performance"""

    def __init__(
        self,
        title: str = "Performance Report",
        subtitle: str = "",
        author: str = "Momentum Forge"
    ):
        self.title = title
        self.subtitle = subtitle
        self.author = author
        self.sections: List[ReportSection] = []
        self.generated_at = datetime.now()

    def add_section(self, section: ReportSection) -> None:
        """Adiciona secao ao relatorio"""
        self.sections.append(section)

    def generate_summary(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> ReportSection:
        """Gera secao de resumo executivo"""
        calculator = MetricsCalculator()
        metrics = calculator.calculate_all(returns, benchmark_returns)

        content = {
            'Periodo': f"{metrics.start_date} a {metrics.end_date}",
            'Total de Dias': metrics.total_days,
            'Retorno Total': f"{metrics.total_return:.2%}",
            'CAGR': f"{metrics.cagr:.2%}",
            'Volatilidade': f"{metrics.annualized_volatility:.2%}",
            'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{metrics.sortino_ratio:.2f}",
            'Max Drawdown': f"{metrics.max_drawdown:.2%}",
            'Calmar Ratio': f"{metrics.calmar_ratio:.2f}"
        }

        if benchmark_returns is not None:
            content.update({
                'Alpha': f"{metrics.alpha:.2%}",
                'Beta': f"{metrics.beta:.2f}",
                'Information Ratio': f"{metrics.information_ratio:.2f}"
            })

        return ReportSection(title="Resumo Executivo", content=content)

    def generate_risk_section(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> ReportSection:
        """Gera secao de metricas de risco"""
        calculator = MetricsCalculator()
        metrics = calculator.calculate_all(returns, benchmark_returns)

        content = {
            'Volatilidade Anualizada': f"{metrics.annualized_volatility:.2%}",
            'Volatilidade Downside': f"{metrics.downside_volatility:.2%}",
            'VaR 95% (Diario)': f"{metrics.var_95:.2%}",
            'VaR 99% (Diario)': f"{metrics.var_99:.2%}",
            'CVaR 95%': f"{metrics.cvar_95:.2%}",
            'Max Drawdown': f"{metrics.max_drawdown:.2%}",
            'Duracao Max DD (dias)': metrics.max_drawdown_duration,
            'Ulcer Index': f"{metrics.ulcer_index:.4f}",
            'Skewness': f"{metrics.skewness:.2f}",
            'Kurtosis': f"{metrics.kurtosis:.2f}"
        }

        return ReportSection(title="Metricas de Risco", content=content)

    def generate_monthly_table(self, returns: pd.Series) -> ReportSection:
        """Gera tabela de retornos mensais"""
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

        # Pivot por ano e mes
        if len(monthly) > 0:
            df = pd.DataFrame({
                'Year': monthly.index.year,
                'Month': monthly.index.month,
                'Return': monthly.values
            })

            pivot = df.pivot(index='Year', columns='Month', values='Return')
            pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

            # Adiciona retorno anual
            yearly = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
            pivot['YTD'] = yearly.values[:len(pivot)]

            content = {
                'table': pivot.round(4).to_dict()
            }
        else:
            content = {'table': {}}

        return ReportSection(title="Retornos Mensais", content=content)

    def generate_drawdown_analysis(self, returns: pd.Series) -> ReportSection:
        """Gera analise de drawdowns"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak

        # Top 5 drawdowns
        is_dd = drawdown < 0
        dd_groups = (~is_dd).cumsum()

        drawdowns = []
        for group_id in dd_groups[is_dd].unique():
            mask = dd_groups == group_id
            dd_period = drawdown[mask]

            if len(dd_period) > 0:
                drawdowns.append({
                    'start': str(dd_period.index[0].date()),
                    'end': str(dd_period.index[-1].date()),
                    'depth': float(dd_period.min()),
                    'duration': len(dd_period)
                })

        # Ordena por profundidade
        drawdowns = sorted(drawdowns, key=lambda x: x['depth'])[:5]

        content = {
            'worst_drawdowns': drawdowns,
            'current_drawdown': float(drawdown.iloc[-1]) if len(drawdown) > 0 else 0,
            'recovery_status': 'Recovered' if drawdown.iloc[-1] == 0 else 'In Drawdown'
        }

        return ReportSection(title="Analise de Drawdowns", content=content)

    def generate_rolling_analysis(
        self,
        returns: pd.Series,
        window: int = 252
    ) -> ReportSection:
        """Gera analise de metricas rolling"""
        if len(returns) < window:
            return ReportSection(
                title="Analise Rolling",
                content={'note': f'Dados insuficientes (minimo {window} dias)'}
            )

        # Sharpe rolling
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0
        )

        # Volatilidade rolling
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)

        content = {
            'window_days': window,
            'current_rolling_sharpe': float(rolling_sharpe.iloc[-1]) if len(rolling_sharpe) > 0 else 0,
            'current_rolling_vol': float(rolling_vol.iloc[-1]) if len(rolling_vol) > 0 else 0,
            'max_rolling_sharpe': float(rolling_sharpe.max()),
            'min_rolling_sharpe': float(rolling_sharpe.min()),
            'avg_rolling_sharpe': float(rolling_sharpe.mean())
        }

        return ReportSection(title="Analise Rolling", content=content)

    def generate_full_report(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        positions_history: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Gera relatorio completo"""
        self.sections.clear()

        # Adiciona secoes
        self.add_section(self.generate_summary(returns, benchmark_returns))
        self.add_section(self.generate_risk_section(returns, benchmark_returns))
        self.add_section(self.generate_monthly_table(returns))
        self.add_section(self.generate_drawdown_analysis(returns))
        self.add_section(self.generate_rolling_analysis(returns))

        return self.to_dict()

    def to_dict(self) -> Dict:
        """Converte relatorio para dicionario"""
        return {
            'title': self.title,
            'subtitle': self.subtitle,
            'author': self.author,
            'generated_at': self.generated_at.isoformat(),
            'sections': [
                {
                    'title': s.title,
                    'content': s.content,
                    'charts': s.charts
                }
                for s in self.sections
            ]
        }

    def to_json(self, indent: int = 2) -> str:
        """Converte para JSON"""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Converte para Markdown"""
        lines = [
            f"# {self.title}",
            f"*{self.subtitle}*" if self.subtitle else "",
            f"**Gerado em:** {self.generated_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Por:** {self.author}",
            "",
            "---",
            ""
        ]

        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")

            for key, value in section.content.items():
                if key == 'table':
                    # Renderiza tabela
                    if isinstance(value, dict) and value:
                        df = pd.DataFrame(value)
                        lines.append(df.to_markdown())
                elif key == 'worst_drawdowns':
                    lines.append("### Top 5 Drawdowns")
                    for i, dd in enumerate(value, 1):
                        lines.append(f"{i}. {dd['start']} a {dd['end']}: "
                                   f"{dd['depth']:.2%} ({dd['duration']} dias)")
                else:
                    lines.append(f"- **{key}:** {value}")

            lines.append("")

        return "\n".join(lines)


def generate_tearsheet(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Strategy Tearsheet"
) -> Dict:
    """
    Gera tearsheet completo no estilo institucional.

    Args:
        returns: Serie de retornos diarios
        benchmark_returns: Retornos do benchmark (opcional)
        title: Titulo do relatorio

    Returns:
        Dicionario com todas as metricas e dados para visualizacao
    """
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all(returns, benchmark_returns)

    # Prepara dados para graficos
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak

    # Distribuicao de retornos
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    return {
        'title': title,
        'metrics': metrics.to_dict(),
        'full_metrics': {
            'returns': {
                'total': metrics.total_return,
                'cagr': metrics.cagr,
                'mtd': metrics.mtd_return,
                'ytd': metrics.ytd_return
            },
            'risk': {
                'volatility': metrics.annualized_volatility,
                'downside_vol': metrics.downside_volatility,
                'max_drawdown': metrics.max_drawdown,
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95
            },
            'risk_adjusted': {
                'sharpe': metrics.sharpe_ratio,
                'sortino': metrics.sortino_ratio,
                'calmar': metrics.calmar_ratio,
                'omega': metrics.omega_ratio
            },
            'benchmark': {
                'alpha': metrics.alpha,
                'beta': metrics.beta,
                'info_ratio': metrics.information_ratio,
                'tracking_error': metrics.tracking_error,
                'up_capture': metrics.up_capture,
                'down_capture': metrics.down_capture
            },
            'consistency': {
                'positive_months': metrics.positive_months,
                'negative_months': metrics.negative_months,
                'win_rate': metrics.monthly_win_rate,
                'best_month': metrics.best_month,
                'worst_month': metrics.worst_month
            }
        },
        'charts_data': {
            'equity_curve': cumulative.to_dict(),
            'drawdown': drawdown.to_dict(),
            'monthly_returns': monthly.to_dict(),
            'returns_distribution': returns.to_dict()
        },
        'period': {
            'start': metrics.start_date,
            'end': metrics.end_date,
            'days': metrics.total_days
        }
    }
