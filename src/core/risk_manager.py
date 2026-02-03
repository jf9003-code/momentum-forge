"""
Gestao de Risco Institucional
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from enum import Enum

from config.settings import RiskConfig


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(Enum):
    POSITION_SIZE = "position_size"
    SECTOR_CONCENTRATION = "sector_concentration"
    LEVERAGE = "leverage"
    DRAWDOWN = "drawdown"
    VAR_BREACH = "var_breach"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"


@dataclass
class RiskAlert:
    """Alerta de risco"""
    timestamp: datetime
    risk_type: RiskType
    level: RiskLevel
    message: str
    current_value: float
    limit: float
    symbol: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.level.value.upper()}] {self.risk_type.value}: {self.message}"


@dataclass
class RiskMetrics:
    """Metricas de risco do portfolio"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    current_drawdown: float
    volatility_annual: float
    beta: float
    correlation_avg: float
    leverage: float
    concentration_hhi: float
    liquidity_days: float


class RiskManager:
    """Gerenciador de Risco Institucional"""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.alerts: List[RiskAlert] = []
        self.risk_history: List[Tuple[datetime, RiskMetrics]] = []
        self._peak_equity = 0.0

    def check_position_size(
        self,
        symbol: str,
        proposed_weight: float,
        current_weights: Dict[str, float]
    ) -> Tuple[bool, Optional[RiskAlert]]:
        """Verifica se tamanho da posicao esta dentro dos limites"""
        total_weight = current_weights.get(symbol, 0.0) + proposed_weight

        if abs(total_weight) > self.config.max_position_size:
            alert = RiskAlert(
                timestamp=datetime.now(),
                risk_type=RiskType.POSITION_SIZE,
                level=RiskLevel.HIGH,
                message=f"Posicao em {symbol} excede limite de {self.config.max_position_size:.1%}",
                current_value=total_weight,
                limit=self.config.max_position_size,
                symbol=symbol
            )
            self.alerts.append(alert)
            return False, alert

        return True, None

    def check_sector_exposure(
        self,
        sector_exposures: Dict[str, float],
        portfolio_value: float
    ) -> Tuple[bool, List[RiskAlert]]:
        """Verifica exposicao por setor"""
        alerts = []
        passed = True

        for sector, exposure in sector_exposures.items():
            weight = abs(exposure) / portfolio_value if portfolio_value > 0 else 0

            if weight > self.config.max_sector_exposure:
                alert = RiskAlert(
                    timestamp=datetime.now(),
                    risk_type=RiskType.SECTOR_CONCENTRATION,
                    level=RiskLevel.MEDIUM,
                    message=f"Exposicao ao setor {sector} ({weight:.1%}) excede limite",
                    current_value=weight,
                    limit=self.config.max_sector_exposure
                )
                alerts.append(alert)
                self.alerts.append(alert)
                passed = False

        return passed, alerts

    def check_leverage(self, current_leverage: float) -> Tuple[bool, Optional[RiskAlert]]:
        """Verifica nivel de alavancagem"""
        if current_leverage > self.config.max_portfolio_leverage:
            alert = RiskAlert(
                timestamp=datetime.now(),
                risk_type=RiskType.LEVERAGE,
                level=RiskLevel.HIGH,
                message=f"Alavancagem ({current_leverage:.2f}x) excede limite",
                current_value=current_leverage,
                limit=self.config.max_portfolio_leverage
            )
            self.alerts.append(alert)
            return False, alert

        return True, None

    def check_drawdown(
        self,
        current_equity: float,
        peak_equity: Optional[float] = None
    ) -> Tuple[bool, Optional[RiskAlert]]:
        """Verifica nivel de drawdown"""
        if peak_equity is not None:
            self._peak_equity = max(self._peak_equity, peak_equity)
        self._peak_equity = max(self._peak_equity, current_equity)

        if self._peak_equity == 0:
            return True, None

        current_dd = (self._peak_equity - current_equity) / self._peak_equity

        if current_dd > self.config.max_drawdown_limit:
            level = RiskLevel.CRITICAL if current_dd > self.config.max_drawdown_limit * 1.5 else RiskLevel.HIGH
            alert = RiskAlert(
                timestamp=datetime.now(),
                risk_type=RiskType.DRAWDOWN,
                level=level,
                message=f"Drawdown ({current_dd:.1%}) excede limite de {self.config.max_drawdown_limit:.1%}",
                current_value=current_dd,
                limit=self.config.max_drawdown_limit
            )
            self.alerts.append(alert)
            return False, alert

        return True, None

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """Calcula Value at Risk"""
        if len(returns) < 30:
            return 0.0

        if method == 'historical':
            var = np.percentile(returns, (1 - confidence) * 100)
        elif method == 'parametric':
            mu = returns.mean()
            sigma = returns.std()
            var = mu + sigma * stats.norm.ppf(1 - confidence)
        elif method == 'cornish_fisher':
            mu = returns.mean()
            sigma = returns.std()
            skew = returns.skew()
            kurt = returns.kurtosis()
            z = stats.norm.ppf(1 - confidence)
            z_cf = (z + (z**2 - 1) * skew / 6 +
                    (z**3 - 3*z) * (kurt - 3) / 24 -
                    (2*z**3 - 5*z) * skew**2 / 36)
            var = mu + sigma * z_cf
        else:
            var = np.percentile(returns, (1 - confidence) * 100)

        return var

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """Calcula Conditional VaR (Expected Shortfall)"""
        if len(returns) < 30:
            return 0.0

        var = self.calculate_var(returns, confidence)
        cvar = returns[returns <= var].mean()

        return cvar if not np.isnan(cvar) else var

    def check_var(
        self,
        returns: pd.Series,
        portfolio_value: float
    ) -> Tuple[bool, Optional[RiskAlert]]:
        """Verifica se VaR esta dentro do limite"""
        var = self.calculate_var(returns, self.config.var_confidence)
        var_amount = abs(var) * portfolio_value
        limit_amount = self.config.var_limit * portfolio_value

        if var_amount > limit_amount:
            alert = RiskAlert(
                timestamp=datetime.now(),
                risk_type=RiskType.VAR_BREACH,
                level=RiskLevel.HIGH,
                message=f"VaR ({abs(var):.2%}) excede limite de {self.config.var_limit:.2%}",
                current_value=abs(var),
                limit=self.config.var_limit
            )
            self.alerts.append(alert)
            return False, alert

        return True, None

    def calculate_correlation_matrix(
        self,
        returns_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calcula matriz de correlacao"""
        return returns_df.corr()

    def check_correlation(
        self,
        returns_df: pd.DataFrame
    ) -> Tuple[bool, List[RiskAlert]]:
        """Verifica correlacoes altas entre posicoes"""
        if returns_df.empty or len(returns_df.columns) < 2:
            return True, []

        corr_matrix = self.calculate_correlation_matrix(returns_df)
        alerts = []
        passed = True

        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > self.config.correlation_threshold:
                        alert = RiskAlert(
                            timestamp=datetime.now(),
                            risk_type=RiskType.CORRELATION,
                            level=RiskLevel.MEDIUM,
                            message=f"Alta correlacao entre {col1} e {col2}: {corr:.2f}",
                            current_value=corr,
                            limit=self.config.correlation_threshold
                        )
                        alerts.append(alert)
                        self.alerts.append(alert)
                        passed = False

        return passed, alerts

    def calculate_liquidity_risk(
        self,
        position_values: Dict[str, float],
        adv_values: Dict[str, float]
    ) -> Dict[str, float]:
        """Calcula dias para liquidar cada posicao"""
        days_to_liquidate = {}

        for symbol, value in position_values.items():
            adv = adv_values.get(symbol, 0)
            if adv > 0:
                max_daily_liquidation = adv * self.config.min_liquidity_adv
                days = abs(value) / max_daily_liquidation if max_daily_liquidation > 0 else float('inf')
                days_to_liquidate[symbol] = days
            else:
                days_to_liquidate[symbol] = float('inf')

        return days_to_liquidate

    def check_liquidity(
        self,
        position_values: Dict[str, float],
        adv_values: Dict[str, float],
        max_days: float = 5.0
    ) -> Tuple[bool, List[RiskAlert]]:
        """Verifica risco de liquidez"""
        days_to_liq = self.calculate_liquidity_risk(position_values, adv_values)
        alerts = []
        passed = True

        for symbol, days in days_to_liq.items():
            if days > max_days:
                alert = RiskAlert(
                    timestamp=datetime.now(),
                    risk_type=RiskType.LIQUIDITY,
                    level=RiskLevel.MEDIUM if days < max_days * 2 else RiskLevel.HIGH,
                    message=f"Posicao em {symbol} levaria {days:.1f} dias para liquidar",
                    current_value=days,
                    limit=max_days,
                    symbol=symbol
                )
                alerts.append(alert)
                self.alerts.append(alert)
                passed = False

        return passed, alerts

    def calculate_hhi(self, weights: Dict[str, float]) -> float:
        """Calcula Herfindahl-Hirschman Index (concentracao)"""
        if not weights:
            return 0.0

        return sum(w**2 for w in weights.values())

    def calculate_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calcula beta do portfolio vs benchmark"""
        if len(portfolio_returns) < 30 or len(benchmark_returns) < 30:
            return 1.0

        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 30:
            return 1.0

        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
        var = np.var(aligned.iloc[:, 1])

        return cov / var if var > 0 else 1.0

    def calculate_risk_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        position_weights: Optional[Dict[str, float]] = None,
        current_equity: float = 0.0,
        leverage: float = 1.0
    ) -> RiskMetrics:
        """Calcula todas as metricas de risco"""
        if len(portfolio_returns) < 30:
            return RiskMetrics(
                var_95=0.0,
                var_99=0.0,
                cvar_95=0.0,
                cvar_99=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                volatility_annual=0.0,
                beta=1.0,
                correlation_avg=0.0,
                leverage=leverage,
                concentration_hhi=0.0,
                liquidity_days=0.0
            )

        # VaR e CVaR
        var_95 = self.calculate_var(portfolio_returns, 0.95)
        var_99 = self.calculate_var(portfolio_returns, 0.99)
        cvar_95 = self.calculate_cvar(portfolio_returns, 0.95)
        cvar_99 = self.calculate_cvar(portfolio_returns, 0.99)

        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        current_dd = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0

        # Volatilidade
        vol_annual = portfolio_returns.std() * np.sqrt(252)

        # Beta
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            beta = self.calculate_beta(portfolio_returns, benchmark_returns)
        else:
            beta = 1.0

        # Concentracao
        hhi = self.calculate_hhi(position_weights) if position_weights else 0.0

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            volatility_annual=vol_annual,
            beta=beta,
            correlation_avg=0.0,
            leverage=leverage,
            concentration_hhi=hhi,
            liquidity_days=0.0
        )

    def run_stress_test(
        self,
        portfolio_returns: pd.Series,
        scenarios: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Executa stress tests em cenarios historicos"""
        if scenarios is None:
            scenarios = {
                'black_monday_1987': -0.226,
                'asian_crisis_1997': -0.073,
                'ltcm_1998': -0.085,
                'dot_com_2000': -0.089,
                'gfc_2008': -0.170,
                'flash_crash_2010': -0.061,
                'covid_2020': -0.128,
                'fed_taper_2022': -0.043
            }

        portfolio_vol = portfolio_returns.std()
        market_vol = 0.01  # Aproximacao vol diaria mercado

        stress_results = {}
        for scenario_name, market_shock in scenarios.items():
            portfolio_shock = market_shock * (portfolio_vol / market_vol)
            stress_results[scenario_name] = min(portfolio_shock, market_shock)

        return stress_results

    def get_risk_summary(self) -> Dict:
        """Retorna resumo de risco"""
        recent_alerts = [a for a in self.alerts[-100:]]

        alerts_by_type = {}
        for alert in recent_alerts:
            risk_type = alert.risk_type.value
            if risk_type not in alerts_by_type:
                alerts_by_type[risk_type] = 0
            alerts_by_type[risk_type] += 1

        critical_count = len([a for a in recent_alerts if a.level == RiskLevel.CRITICAL])
        high_count = len([a for a in recent_alerts if a.level == RiskLevel.HIGH])

        overall_level = RiskLevel.LOW
        if critical_count > 0:
            overall_level = RiskLevel.CRITICAL
        elif high_count > 2:
            overall_level = RiskLevel.HIGH
        elif high_count > 0 or len(recent_alerts) > 5:
            overall_level = RiskLevel.MEDIUM

        return {
            'overall_risk_level': overall_level.value,
            'total_alerts': len(recent_alerts),
            'critical_alerts': critical_count,
            'high_alerts': high_count,
            'alerts_by_type': alerts_by_type,
            'recent_alerts': [str(a) for a in recent_alerts[-10:]]
        }

    def reset(self) -> None:
        """Reseta estado do risk manager"""
        self.alerts.clear()
        self.risk_history.clear()
        self._peak_equity = 0.0
