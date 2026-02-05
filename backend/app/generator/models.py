"""
D5 ROBUST MASTER - Generator Models
Core data structures for strategy specification
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Union
import uuid
import json


class StrategyType(Enum):
    """Strategy archetypes with distinct logic profiles"""
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    VOLATILITY_EXPANSION = "volatility_expansion"
    TIME_BASED = "time_based"


class Timeframe(Enum):
    """Trading timeframes with max lookback limits"""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"

    @property
    def max_lookback(self) -> int:
        """Maximum indicator lookback for this timeframe"""
        limits = {
            "M1": 60, "M5": 100, "M15": 150, "M30": 200,
            "H1": 300, "H4": 400, "D1": 500, "W1": 200
        }
        return limits.get(self.value, 200)

    @property
    def minutes(self) -> int:
        """Timeframe in minutes"""
        mins = {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 60, "H4": 240, "D1": 1440, "W1": 10080
        }
        return mins.get(self.value, 60)


class MarketType(Enum):
    """Market regime context"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"
    ANY = "any"


class Indicator(Enum):
    """Available indicators"""
    # Oscillators
    RSI = "RSI"
    STOCH_K = "STOCH_K"
    STOCH_D = "STOCH_D"
    CCI = "CCI"
    WILLIAMS_R = "WILLIAMS_R"
    MFI = "MFI"

    # Momentum
    ROC = "ROC"
    MOM = "MOM"
    TRIX = "TRIX"
    ADX = "ADX"

    # Moving Averages
    SMA = "SMA"
    EMA = "EMA"
    WMA = "WMA"
    DEMA = "DEMA"
    TEMA = "TEMA"

    # Volatility
    ATR = "ATR"
    STDDEV = "STDDEV"
    BBANDS_UPPER = "BBANDS_UPPER"
    BBANDS_LOWER = "BBANDS_LOWER"
    BBANDS_WIDTH = "BBANDS_WIDTH"
    KELTNER_UPPER = "KELTNER_UPPER"
    KELTNER_LOWER = "KELTNER_LOWER"

    # Price
    CLOSE = "CLOSE"
    OPEN = "OPEN"
    HIGH = "HIGH"
    LOW = "LOW"
    DONCHIAN_HIGH = "DONCHIAN_HIGH"
    DONCHIAN_LOW = "DONCHIAN_LOW"

    # Volume
    VOLUME = "VOLUME"
    OBV = "OBV"
    VWAP = "VWAP"


class RelationalOperator(Enum):
    """Comparison operators for conditions"""
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "=="
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"


class EntryPattern(Enum):
    """Entry logic patterns"""
    THRESHOLD = "threshold"           # RSI < 30
    CROSSOVER = "crossover"           # FastMA crosses SlowMA
    BREAK_LEVEL = "break_level"       # Price breaks N-period high
    PULLBACK = "pullback"             # Price pulls back to MA
    TIME_TRIGGER = "time_trigger"     # Time-based entry
    VOLATILITY_SQUEEZE = "volatility_squeeze"  # Bollinger squeeze


class ExitPattern(Enum):
    """Exit logic patterns"""
    FIXED_RR = "fixed_rr"             # Fixed risk-reward
    VOLATILITY_EXIT = "volatility_exit"  # ATR-based
    TIME_EXIT = "time_exit"           # Fixed holding period
    SIGNAL_BASED = "signal_based"     # Opposite signal
    TRAILING_STOP = "trailing_stop"   # Trailing stop


class RiskModel(Enum):
    """Position sizing models"""
    FIXED_RISK = "fixed_risk"         # Fixed % of capital per trade
    VOLATILITY_RISK = "volatility_risk"  # ATR-based sizing
    KELLY = "kelly"                   # Kelly criterion


@dataclass
class IndicatorConfig:
    """Configuration for a single indicator"""
    indicator: Indicator
    period: Optional[int] = None
    source: str = "close"

    def to_dict(self) -> Dict:
        return {
            "indicator": self.indicator.value,
            "period": self.period,
            "source": self.source
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "IndicatorConfig":
        return cls(
            indicator=Indicator(data["indicator"]),
            period=data.get("period"),
            source=data.get("source", "close")
        )


@dataclass
class Condition:
    """A single condition in entry/exit logic"""
    left: IndicatorConfig
    operator: RelationalOperator
    right: Union[IndicatorConfig, float, int, str]

    def to_dict(self) -> Dict:
        return {
            "left": self.left.to_dict(),
            "operator": self.operator.value,
            "right": self.right.to_dict() if isinstance(self.right, IndicatorConfig) else self.right
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Condition":
        right = data["right"]
        if isinstance(right, dict):
            right = IndicatorConfig.from_dict(right)
        return cls(
            left=IndicatorConfig.from_dict(data["left"]),
            operator=RelationalOperator(data["operator"]),
            right=right
        )

    def __str__(self) -> str:
        left_str = f"{self.left.indicator.value}({self.left.period})" if self.left.period else self.left.indicator.value
        if isinstance(self.right, IndicatorConfig):
            right_str = f"{self.right.indicator.value}({self.right.period})" if self.right.period else self.right.indicator.value
        else:
            right_str = str(self.right)
        return f"{left_str} {self.operator.value} {right_str}"


@dataclass
class EntryLogic:
    """Entry logic specification"""
    pattern: EntryPattern
    conditions: List[Condition]
    direction: str = "long"  # "long" or "short"
    logic_operator: str = "AND"  # "AND" or "OR"

    def to_dict(self) -> Dict:
        return {
            "pattern": self.pattern.value,
            "conditions": [c.to_dict() for c in self.conditions],
            "direction": self.direction,
            "logic_operator": self.logic_operator
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EntryLogic":
        return cls(
            pattern=EntryPattern(data["pattern"]),
            conditions=[Condition.from_dict(c) for c in data.get("conditions", [])],
            direction=data.get("direction", "long"),
            logic_operator=data.get("logic_operator", "AND")
        )


@dataclass
class ExitLogic:
    """Exit logic specification"""
    pattern: ExitPattern
    risk_reward: Optional[float] = None
    atr_sl_mult: Optional[float] = None
    atr_tp_mult: Optional[float] = None
    time_exit_bars: Optional[int] = None
    trailing_stop_pct: Optional[float] = None
    stop_loss: Optional[Condition] = None

    def to_dict(self) -> Dict:
        return {
            "pattern": self.pattern.value,
            "risk_reward": self.risk_reward,
            "atr_sl_mult": self.atr_sl_mult,
            "atr_tp_mult": self.atr_tp_mult,
            "time_exit_bars": self.time_exit_bars,
            "trailing_stop_pct": self.trailing_stop_pct,
            "stop_loss": self.stop_loss.to_dict() if self.stop_loss else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ExitLogic":
        return cls(
            pattern=ExitPattern(data["pattern"]),
            risk_reward=data.get("risk_reward"),
            atr_sl_mult=data.get("atr_sl_mult"),
            atr_tp_mult=data.get("atr_tp_mult"),
            time_exit_bars=data.get("time_exit_bars"),
            trailing_stop_pct=data.get("trailing_stop_pct"),
            stop_loss=Condition.from_dict(data["stop_loss"]) if data.get("stop_loss") else None
        )


@dataclass
class RiskConfig:
    """Risk management configuration"""
    model: RiskModel = RiskModel.FIXED_RISK
    risk_percent: Optional[float] = 2.0
    atr_period: Optional[int] = 14
    atr_factor: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "model": self.model.value,
            "risk_percent": self.risk_percent,
            "atr_period": self.atr_period,
            "atr_factor": self.atr_factor
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RiskConfig":
        return cls(
            model=RiskModel(data.get("model", "fixed_risk")),
            risk_percent=data.get("risk_percent", 2.0),
            atr_period=data.get("atr_period", 14),
            atr_factor=data.get("atr_factor")
        )


@dataclass
class StrategySpec:
    """
    Complete strategy specification.
    This is a STRUCTURAL HYPOTHESIS, not a performance result.
    """
    strategy_type: StrategyType
    template_id: str
    market_context: MarketType
    timeframe: Timeframe
    entry_logic: EntryLogic
    exit_logic: ExitLogic
    risk_config: RiskConfig
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary"""
        return {
            "id": self.id,
            "strategy_type": self.strategy_type.value,
            "template_id": self.template_id,
            "market_context": self.market_context.value,
            "timeframe": self.timeframe.value,
            "entry_logic": self.entry_logic.to_dict(),
            "exit_logic": self.exit_logic.to_dict(),
            "risk_config": self.risk_config.to_dict(),
            "parameters": self.parameters,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StrategySpec":
        """Create from dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:12]),
            strategy_type=StrategyType(data["strategy_type"]),
            template_id=data["template_id"],
            market_context=MarketType(data.get("market_context", "any")),
            timeframe=Timeframe(data["timeframe"]),
            entry_logic=EntryLogic.from_dict(data["entry_logic"]),
            exit_logic=ExitLogic.from_dict(data["exit_logic"]),
            risk_config=RiskConfig.from_dict(data.get("risk_config", {})),
            parameters=data.get("parameters", {}),
            metadata=data.get("metadata", {})
        )

    def to_json(self) -> str:
        """Export as JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def get_entry_summary(self) -> str:
        """Get human-readable entry summary"""
        if self.entry_logic.conditions:
            return str(self.entry_logic.conditions[0])
        return self.entry_logic.pattern.value

    def get_exit_summary(self) -> str:
        """Get human-readable exit summary"""
        parts = []
        if self.exit_logic.atr_sl_mult:
            parts.append(f"SL: ATR×{self.exit_logic.atr_sl_mult}")
        if self.exit_logic.atr_tp_mult:
            parts.append(f"TP: ATR×{self.exit_logic.atr_tp_mult}")
        if self.exit_logic.time_exit_bars:
            parts.append(f"Exit: {self.exit_logic.time_exit_bars} bars")
        return " | ".join(parts) if parts else self.exit_logic.pattern.value
