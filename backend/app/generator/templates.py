"""
Strategy Templates - Institutional Pattern
Top-down composition templates for strategy generation
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import copy


class TemplateType(Enum):
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    VOLATILITY_EXPANSION = "volatility_expansion"
    TIME_BASED = "time_based"


@dataclass
class IndicatorSlot:
    """Defines a slot for an indicator in a template"""
    name: str
    category: str  # trend, momentum, volatility, volume
    required: bool = True
    allowed_indicators: List[str] = field(default_factory=list)
    default_indicator: Optional[str] = None
    parameter_ranges: Dict[str, tuple] = field(default_factory=dict)


@dataclass
class ConditionSlot:
    """Defines a slot for a condition in entry/exit logic"""
    name: str
    indicator_slot: str  # Reference to IndicatorSlot
    operators: List[str] = field(default_factory=list)
    compare_to_options: List[str] = field(default_factory=list)
    required: bool = True


@dataclass
class StrategyTemplate:
    """Complete strategy template definition"""
    template_id: str
    name: str
    strategy_type: TemplateType
    description: str

    # Indicator slots
    indicator_slots: List[IndicatorSlot] = field(default_factory=list)

    # Entry logic slots
    entry_slots: List[ConditionSlot] = field(default_factory=list)
    entry_logic_type: str = "AND"  # AND or OR

    # Exit logic slots
    exit_slots: List[ConditionSlot] = field(default_factory=list)
    exit_logic_type: str = "OR"

    # Risk configuration constraints
    risk_constraints: Dict[str, Any] = field(default_factory=dict)

    # Timeframe constraints
    allowed_timeframes: List[str] = field(default_factory=list)

    # Market type constraints
    allowed_markets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'template_id': self.template_id,
            'name': self.name,
            'strategy_type': self.strategy_type.value,
            'description': self.description,
            'indicator_slots': [
                {
                    'name': s.name,
                    'category': s.category,
                    'required': s.required,
                    'allowed_indicators': s.allowed_indicators,
                    'default_indicator': s.default_indicator,
                    'parameter_ranges': s.parameter_ranges
                }
                for s in self.indicator_slots
            ],
            'entry_slots': [
                {
                    'name': s.name,
                    'indicator_slot': s.indicator_slot,
                    'operators': s.operators,
                    'compare_to_options': s.compare_to_options,
                    'required': s.required
                }
                for s in self.entry_slots
            ],
            'entry_logic_type': self.entry_logic_type,
            'exit_slots': [
                {
                    'name': s.name,
                    'indicator_slot': s.indicator_slot,
                    'operators': s.operators,
                    'compare_to_options': s.compare_to_options,
                    'required': s.required
                }
                for s in self.exit_slots
            ],
            'exit_logic_type': self.exit_logic_type,
            'risk_constraints': self.risk_constraints,
            'allowed_timeframes': self.allowed_timeframes,
            'allowed_markets': self.allowed_markets
        }


# ============================================================================
# INSTITUTIONAL STRATEGY TEMPLATES
# ============================================================================

MEAN_REVERSION_RSI = StrategyTemplate(
    template_id="mr_rsi_v1",
    name="Mean Reversion RSI",
    strategy_type=TemplateType.MEAN_REVERSION,
    description="RSI-based mean reversion with Bollinger Band confirmation",
    indicator_slots=[
        IndicatorSlot(
            name="oscillator",
            category="momentum",
            required=True,
            allowed_indicators=["RSI", "Stochastic", "CCI", "Williams_R"],
            default_indicator="RSI",
            parameter_ranges={"period": (7, 21), "overbought": (70, 85), "oversold": (15, 30)}
        ),
        IndicatorSlot(
            name="band",
            category="volatility",
            required=True,
            allowed_indicators=["Bollinger", "Keltner", "Donchian"],
            default_indicator="Bollinger",
            parameter_ranges={"period": (15, 30), "std_dev": (1.5, 2.5)}
        ),
        IndicatorSlot(
            name="trend_filter",
            category="trend",
            required=False,
            allowed_indicators=["SMA", "EMA", "WMA"],
            default_indicator="SMA",
            parameter_ranges={"period": (50, 200)}
        )
    ],
    entry_slots=[
        ConditionSlot(
            name="oscillator_oversold",
            indicator_slot="oscillator",
            operators=["<", "<=", "crosses_below"],
            compare_to_options=["oversold_level", "fixed_value"]
        ),
        ConditionSlot(
            name="price_at_band",
            indicator_slot="band",
            operators=["<", "<=", "touches"],
            compare_to_options=["lower_band"]
        )
    ],
    entry_logic_type="AND",
    exit_slots=[
        ConditionSlot(
            name="oscillator_overbought",
            indicator_slot="oscillator",
            operators=[">", ">=", "crosses_above"],
            compare_to_options=["overbought_level", "midline"]
        ),
        ConditionSlot(
            name="price_at_upper",
            indicator_slot="band",
            operators=[">", ">="],
            compare_to_options=["upper_band", "middle_band"]
        )
    ],
    exit_logic_type="OR",
    risk_constraints={
        "min_risk_reward": 1.5,
        "max_stop_atr": 2.5,
        "position_sizing": ["fixed_fractional", "volatility_based"]
    },
    allowed_timeframes=["M15", "M30", "H1", "H4"],
    allowed_markets=["forex", "indices", "crypto"]
)


MOMENTUM_MACD = StrategyTemplate(
    template_id="mom_macd_v1",
    name="Momentum MACD Trend",
    strategy_type=TemplateType.MOMENTUM,
    description="MACD-based momentum following with trend filter",
    indicator_slots=[
        IndicatorSlot(
            name="momentum",
            category="momentum",
            required=True,
            allowed_indicators=["MACD", "PPO", "ROC"],
            default_indicator="MACD",
            parameter_ranges={
                "fast_period": (8, 15),
                "slow_period": (21, 34),
                "signal_period": (7, 12)
            }
        ),
        IndicatorSlot(
            name="trend",
            category="trend",
            required=True,
            allowed_indicators=["EMA", "SMA", "DEMA", "TEMA"],
            default_indicator="EMA",
            parameter_ranges={"period": (20, 50)}
        ),
        IndicatorSlot(
            name="trend_long",
            category="trend",
            required=False,
            allowed_indicators=["EMA", "SMA"],
            default_indicator="EMA",
            parameter_ranges={"period": (100, 200)}
        ),
        IndicatorSlot(
            name="volume",
            category="volume",
            required=False,
            allowed_indicators=["Volume_SMA", "OBV", "VWAP"],
            default_indicator="Volume_SMA",
            parameter_ranges={"period": (10, 20)}
        )
    ],
    entry_slots=[
        ConditionSlot(
            name="macd_cross",
            indicator_slot="momentum",
            operators=["crosses_above", ">"],
            compare_to_options=["signal_line", "zero_line"]
        ),
        ConditionSlot(
            name="trend_confirm",
            indicator_slot="trend",
            operators=[">", "crosses_above"],
            compare_to_options=["price", "trend_long"]
        )
    ],
    entry_logic_type="AND",
    exit_slots=[
        ConditionSlot(
            name="macd_reversal",
            indicator_slot="momentum",
            operators=["crosses_below", "<"],
            compare_to_options=["signal_line", "zero_line"]
        ),
        ConditionSlot(
            name="trend_break",
            indicator_slot="trend",
            operators=["<", "crosses_below"],
            compare_to_options=["price"]
        )
    ],
    exit_logic_type="OR",
    risk_constraints={
        "min_risk_reward": 2.0,
        "max_stop_atr": 2.0,
        "position_sizing": ["fixed_fractional", "kelly"]
    },
    allowed_timeframes=["M30", "H1", "H4", "D1"],
    allowed_markets=["forex", "indices", "stocks", "crypto"]
)


BREAKOUT_DONCHIAN = StrategyTemplate(
    template_id="brk_donchian_v1",
    name="Donchian Breakout",
    strategy_type=TemplateType.BREAKOUT,
    description="Classic Donchian channel breakout with ATR filter",
    indicator_slots=[
        IndicatorSlot(
            name="channel",
            category="volatility",
            required=True,
            allowed_indicators=["Donchian", "Keltner", "Price_Channel"],
            default_indicator="Donchian",
            parameter_ranges={"period": (15, 55)}
        ),
        IndicatorSlot(
            name="volatility",
            category="volatility",
            required=True,
            allowed_indicators=["ATR", "True_Range", "Standard_Deviation"],
            default_indicator="ATR",
            parameter_ranges={"period": (10, 20)}
        ),
        IndicatorSlot(
            name="volume_filter",
            category="volume",
            required=False,
            allowed_indicators=["Volume_SMA", "OBV"],
            default_indicator="Volume_SMA",
            parameter_ranges={"period": (10, 20)}
        )
    ],
    entry_slots=[
        ConditionSlot(
            name="breakout",
            indicator_slot="channel",
            operators=["breaks_above", ">"],
            compare_to_options=["upper_band", "previous_high"]
        ),
        ConditionSlot(
            name="volatility_expand",
            indicator_slot="volatility",
            operators=[">"],
            compare_to_options=["atr_threshold", "atr_average"]
        )
    ],
    entry_logic_type="AND",
    exit_slots=[
        ConditionSlot(
            name="channel_exit",
            indicator_slot="channel",
            operators=["<", "breaks_below"],
            compare_to_options=["middle_band", "lower_band"]
        ),
        ConditionSlot(
            name="volatility_contract",
            indicator_slot="volatility",
            operators=["<"],
            compare_to_options=["atr_threshold"]
        )
    ],
    exit_logic_type="OR",
    risk_constraints={
        "min_risk_reward": 2.5,
        "max_stop_atr": 1.5,
        "position_sizing": ["volatility_based", "fixed_fractional"]
    },
    allowed_timeframes=["H1", "H4", "D1"],
    allowed_markets=["forex", "futures", "crypto"]
)


VOLATILITY_EXPANSION = StrategyTemplate(
    template_id="vol_expansion_v1",
    name="Volatility Expansion",
    strategy_type=TemplateType.VOLATILITY_EXPANSION,
    description="Trade volatility expansion with squeeze detection",
    indicator_slots=[
        IndicatorSlot(
            name="squeeze",
            category="volatility",
            required=True,
            allowed_indicators=["Bollinger_Squeeze", "Keltner_Squeeze", "TTM_Squeeze"],
            default_indicator="Bollinger_Squeeze",
            parameter_ranges={"bb_period": (20, 20), "kc_period": (20, 20), "kc_mult": (1.0, 2.0)}
        ),
        IndicatorSlot(
            name="momentum",
            category="momentum",
            required=True,
            allowed_indicators=["Momentum", "ROC", "Linear_Regression_Slope"],
            default_indicator="Momentum",
            parameter_ranges={"period": (10, 20)}
        ),
        IndicatorSlot(
            name="atr",
            category="volatility",
            required=True,
            allowed_indicators=["ATR"],
            default_indicator="ATR",
            parameter_ranges={"period": (14, 14)}
        )
    ],
    entry_slots=[
        ConditionSlot(
            name="squeeze_fire",
            indicator_slot="squeeze",
            operators=["fires", "=="],
            compare_to_options=["squeeze_off", "expansion_start"]
        ),
        ConditionSlot(
            name="momentum_direction",
            indicator_slot="momentum",
            operators=[">", "crosses_above"],
            compare_to_options=["zero_line", "previous_value"]
        )
    ],
    entry_logic_type="AND",
    exit_slots=[
        ConditionSlot(
            name="momentum_reversal",
            indicator_slot="momentum",
            operators=["<", "crosses_below"],
            compare_to_options=["zero_line", "previous_value"]
        ),
        ConditionSlot(
            name="squeeze_return",
            indicator_slot="squeeze",
            operators=["=="],
            compare_to_options=["squeeze_on"]
        )
    ],
    exit_logic_type="OR",
    risk_constraints={
        "min_risk_reward": 2.0,
        "max_stop_atr": 2.0,
        "position_sizing": ["volatility_based"]
    },
    allowed_timeframes=["M30", "H1", "H4"],
    allowed_markets=["forex", "indices", "futures"]
)


TIME_BASED_SESSION = StrategyTemplate(
    template_id="time_session_v1",
    name="Session-Based Trading",
    strategy_type=TemplateType.TIME_BASED,
    description="Trade specific market sessions with time filters",
    indicator_slots=[
        IndicatorSlot(
            name="session",
            category="time",
            required=True,
            allowed_indicators=["Session_Filter", "Time_Range"],
            default_indicator="Session_Filter",
            parameter_ranges={
                "session": ["london", "new_york", "asia", "overlap"],
                "start_offset": (-60, 60),
                "end_offset": (-60, 60)
            }
        ),
        IndicatorSlot(
            name="range",
            category="volatility",
            required=True,
            allowed_indicators=["Session_Range", "Asian_Range", "Daily_Range"],
            default_indicator="Session_Range",
            parameter_ranges={"lookback": (1, 5)}
        ),
        IndicatorSlot(
            name="trend",
            category="trend",
            required=False,
            allowed_indicators=["EMA", "SMA"],
            default_indicator="EMA",
            parameter_ranges={"period": (20, 50)}
        )
    ],
    entry_slots=[
        ConditionSlot(
            name="session_active",
            indicator_slot="session",
            operators=["=="],
            compare_to_options=["session_start", "session_active"]
        ),
        ConditionSlot(
            name="range_breakout",
            indicator_slot="range",
            operators=["breaks_above", "breaks_below"],
            compare_to_options=["range_high", "range_low"]
        )
    ],
    entry_logic_type="AND",
    exit_slots=[
        ConditionSlot(
            name="session_end",
            indicator_slot="session",
            operators=["=="],
            compare_to_options=["session_end", "time_limit"]
        ),
        ConditionSlot(
            name="time_exit",
            indicator_slot="session",
            operators=[">="],
            compare_to_options=["max_bars", "fixed_time"]
        )
    ],
    exit_logic_type="OR",
    risk_constraints={
        "min_risk_reward": 1.5,
        "max_stop_atr": 1.5,
        "time_exit_bars": 20,
        "position_sizing": ["fixed_percentage", "fixed_fractional"]
    },
    allowed_timeframes=["M15", "M30", "H1"],
    allowed_markets=["forex", "indices"]
)


# Template Registry
TEMPLATE_REGISTRY: Dict[str, StrategyTemplate] = {
    "mr_rsi_v1": MEAN_REVERSION_RSI,
    "mom_macd_v1": MOMENTUM_MACD,
    "brk_donchian_v1": BREAKOUT_DONCHIAN,
    "vol_expansion_v1": VOLATILITY_EXPANSION,
    "time_session_v1": TIME_BASED_SESSION,
}

# Map strategy types to templates
TYPE_TO_TEMPLATES: Dict[str, List[str]] = {
    "mean_reversion": ["mr_rsi_v1"],
    "momentum": ["mom_macd_v1"],
    "breakout": ["brk_donchian_v1"],
    "volatility_expansion": ["vol_expansion_v1"],
    "time_based": ["time_session_v1"],
}


def get_template(template_id: str) -> Optional[StrategyTemplate]:
    """Get template by ID"""
    return TEMPLATE_REGISTRY.get(template_id)


def get_templates_for_type(strategy_type: str) -> List[StrategyTemplate]:
    """Get all templates for a strategy type"""
    template_ids = TYPE_TO_TEMPLATES.get(strategy_type, [])
    return [TEMPLATE_REGISTRY[tid] for tid in template_ids if tid in TEMPLATE_REGISTRY]


def get_all_templates() -> List[StrategyTemplate]:
    """Get all available templates"""
    return list(TEMPLATE_REGISTRY.values())


def clone_template(template_id: str) -> Optional[StrategyTemplate]:
    """Create a deep copy of a template for customization"""
    template = get_template(template_id)
    if template:
        return copy.deepcopy(template)
    return None
