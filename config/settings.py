"""
Configuracoes globais da plataforma Momentum Forge
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import date


class ExecutionAlgorithm(Enum):
    MARKET = "market"
    TWAP = "twap"
    VWAP = "vwap"
    IMPLEMENTATION_SHORTFALL = "is"


class RebalanceFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class SlippageModel(Enum):
    FIXED = "fixed"
    PROPORTIONAL = "proportional"
    SQRT = "sqrt"  # Square root market impact


@dataclass
class TransactionCostConfig:
    """Configuracao de custos de transacao"""
    commission_bps: float = 5.0  # Basis points
    min_commission: float = 1.0  # Minimo por ordem
    slippage_model: SlippageModel = SlippageModel.PROPORTIONAL
    slippage_bps: float = 2.0
    market_impact_factor: float = 0.1
    borrow_cost_annual: float = 0.5  # Para short selling


@dataclass
class RiskConfig:
    """Configuracao de gestao de risco"""
    max_position_size: float = 0.10  # Max 10% por ativo
    max_sector_exposure: float = 0.30  # Max 30% por setor
    max_portfolio_leverage: float = 1.0  # Sem alavancagem por padrao
    max_drawdown_limit: float = 0.15  # Stop loss em 15% DD
    var_confidence: float = 0.95  # VaR 95%
    var_limit: float = 0.02  # Limite VaR 2%
    correlation_threshold: float = 0.7  # Alerta correlacao alta
    min_liquidity_adv: float = 0.01  # Max 1% do ADV


@dataclass
class BacktestConfig:
    """Configuracao do backtest"""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    initial_capital: float = 100000.0
    currency: str = "USD"
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    transaction_costs: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    benchmark: Optional[str] = "SPY"
    warmup_period: int = 252  # Dias para calculo de indicadores


@dataclass
class MomentumStrategyConfig:
    """Configuracao especifica da estrategia de momentum"""
    lookback_short: int = 63  # 3 meses
    lookback_long: int = 126  # 6 meses
    lookback_sma: int = 200  # SMA filter
    weight_short: float = 0.6  # Peso momentum curto
    min_return: float = -0.05  # Retorno minimo
    top_n: int = 2  # Numero de ativos
    universe: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        "TLT", "IEF", "LQD", "HYG",
        "GLD", "SLV", "USO", "DBA",
        "VNQ", "XLF", "XLE", "XLK", "XLV"
    ])
    exclude_bottom: bool = True  # Excluir piores performers


@dataclass
class ExecutionConfig:
    """Configuracao de execucao"""
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    max_participation_rate: float = 0.10  # Max 10% do volume
    urgency: float = 0.5  # 0=paciente, 1=urgente
    allow_partial_fills: bool = True
    timeout_seconds: int = 300


@dataclass
class MonitoringConfig:
    """Configuracao de monitoramento"""
    log_level: str = "INFO"
    log_file: str = "logs/momentum_forge.log"
    metrics_enabled: bool = True
    alert_email: Optional[str] = None
    alert_slack_webhook: Optional[str] = None


@dataclass
class PlatformConfig:
    """Configuracao completa da plataforma"""
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    strategy: MomentumStrategyConfig = field(default_factory=MomentumStrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlatformConfig':
        """Cria configuracao a partir de dicionario"""
        return cls(
            backtest=BacktestConfig(**data.get('backtest', {})),
            strategy=MomentumStrategyConfig(**data.get('strategy', {})),
            execution=ExecutionConfig(**data.get('execution', {})),
            monitoring=MonitoringConfig(**data.get('monitoring', {}))
        )
