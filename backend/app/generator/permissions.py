"""
D5 ROBUST MASTER - Strategy Permissions
Defines what indicators, patterns, and parameters are allowed for each strategy type.
This enforces structural validity at the archetype level.
"""

from dataclasses import dataclass, field
from typing import List, Set, FrozenSet
from .models import (
    StrategyType, Indicator, EntryPattern, ExitPattern, MarketType
)


@dataclass(frozen=True)
class StrategyProfile:
    """
    Permission profile for a strategy type.
    Defines structural constraints that make sense for the archetype.
    """
    strategy_type: StrategyType
    allowed_entry_patterns: FrozenSet[EntryPattern]
    allowed_exit_patterns: FrozenSet[ExitPattern]
    allowed_indicators: FrozenSet[Indicator]
    forbidden_indicators: FrozenSet[Indicator]
    allowed_market_types: FrozenSet[MarketType]

    # Risk constraints
    min_risk_reward: float = 1.0
    max_risk_reward: float = 5.0
    min_atr_mult: float = 1.0
    max_atr_mult: float = 5.0


# ============================================
# STRATEGY PROFILES
# ============================================

STRATEGY_PROFILES = {
    StrategyType.MEAN_REVERSION: StrategyProfile(
        strategy_type=StrategyType.MEAN_REVERSION,
        allowed_entry_patterns=frozenset([
            EntryPattern.THRESHOLD,
            EntryPattern.PULLBACK,
        ]),
        allowed_exit_patterns=frozenset([
            ExitPattern.FIXED_RR,
            ExitPattern.VOLATILITY_EXIT,
            ExitPattern.TIME_EXIT,
            ExitPattern.SIGNAL_BASED,
        ]),
        allowed_indicators=frozenset([
            Indicator.RSI, Indicator.STOCH_K, Indicator.STOCH_D,
            Indicator.CCI, Indicator.WILLIAMS_R, Indicator.MFI,
            Indicator.BBANDS_UPPER, Indicator.BBANDS_LOWER,
            Indicator.SMA, Indicator.EMA, Indicator.ATR,
            Indicator.CLOSE, Indicator.HIGH, Indicator.LOW,
        ]),
        forbidden_indicators=frozenset([
            Indicator.ADX,  # Trend indicator - doesn't fit mean reversion
            Indicator.DONCHIAN_HIGH, Indicator.DONCHIAN_LOW,  # Breakout
        ]),
        allowed_market_types=frozenset([
            MarketType.RANGING,
            MarketType.QUIET,
            MarketType.ANY,
        ]),
        min_risk_reward=1.0,
        max_risk_reward=3.0,
        min_atr_mult=1.0,
        max_atr_mult=3.0,
    ),

    StrategyType.MOMENTUM: StrategyProfile(
        strategy_type=StrategyType.MOMENTUM,
        allowed_entry_patterns=frozenset([
            EntryPattern.CROSSOVER,
            EntryPattern.THRESHOLD,
            EntryPattern.BREAK_LEVEL,
        ]),
        allowed_exit_patterns=frozenset([
            ExitPattern.FIXED_RR,
            ExitPattern.VOLATILITY_EXIT,
            ExitPattern.TRAILING_STOP,
            ExitPattern.SIGNAL_BASED,
        ]),
        allowed_indicators=frozenset([
            Indicator.RSI, Indicator.ROC, Indicator.MOM, Indicator.ADX,
            Indicator.SMA, Indicator.EMA, Indicator.DEMA, Indicator.TEMA,
            Indicator.CLOSE, Indicator.HIGH, Indicator.LOW,
            Indicator.ATR, Indicator.VOLUME, Indicator.OBV,
        ]),
        forbidden_indicators=frozenset([
            Indicator.BBANDS_WIDTH,  # Volatility squeeze - not momentum
        ]),
        allowed_market_types=frozenset([
            MarketType.TRENDING,
            MarketType.VOLATILE,
            MarketType.ANY,
        ]),
        min_risk_reward=1.5,
        max_risk_reward=5.0,
        min_atr_mult=1.5,
        max_atr_mult=4.0,
    ),

    StrategyType.BREAKOUT: StrategyProfile(
        strategy_type=StrategyType.BREAKOUT,
        allowed_entry_patterns=frozenset([
            EntryPattern.BREAK_LEVEL,
            EntryPattern.VOLATILITY_SQUEEZE,
        ]),
        allowed_exit_patterns=frozenset([
            ExitPattern.FIXED_RR,
            ExitPattern.VOLATILITY_EXIT,
            ExitPattern.TRAILING_STOP,
            ExitPattern.TIME_EXIT,
        ]),
        allowed_indicators=frozenset([
            Indicator.DONCHIAN_HIGH, Indicator.DONCHIAN_LOW,
            Indicator.HIGH, Indicator.LOW, Indicator.CLOSE,
            Indicator.ATR, Indicator.BBANDS_WIDTH, Indicator.STDDEV,
            Indicator.VOLUME, Indicator.ADX,
            Indicator.KELTNER_UPPER, Indicator.KELTNER_LOWER,
        ]),
        forbidden_indicators=frozenset([
            Indicator.RSI,  # Oscillator - opposite of breakout logic
            Indicator.STOCH_K, Indicator.STOCH_D,
        ]),
        allowed_market_types=frozenset([
            MarketType.RANGING,  # Breakout FROM range
            MarketType.QUIET,    # Low volatility precedes breakout
            MarketType.ANY,
        ]),
        min_risk_reward=2.0,
        max_risk_reward=5.0,
        min_atr_mult=1.5,
        max_atr_mult=4.0,
    ),

    StrategyType.VOLATILITY_EXPANSION: StrategyProfile(
        strategy_type=StrategyType.VOLATILITY_EXPANSION,
        allowed_entry_patterns=frozenset([
            EntryPattern.VOLATILITY_SQUEEZE,
            EntryPattern.THRESHOLD,
        ]),
        allowed_exit_patterns=frozenset([
            ExitPattern.VOLATILITY_EXIT,
            ExitPattern.TIME_EXIT,
            ExitPattern.FIXED_RR,
        ]),
        allowed_indicators=frozenset([
            Indicator.ATR, Indicator.STDDEV, Indicator.BBANDS_WIDTH,
            Indicator.KELTNER_UPPER, Indicator.KELTNER_LOWER,
            Indicator.BBANDS_UPPER, Indicator.BBANDS_LOWER,
            Indicator.CLOSE, Indicator.HIGH, Indicator.LOW,
            Indicator.ADX,
        ]),
        forbidden_indicators=frozenset([
            Indicator.RSI,  # Mean reversion indicator
            Indicator.CCI, Indicator.WILLIAMS_R,
        ]),
        allowed_market_types=frozenset([
            MarketType.QUIET,     # Entry during quiet
            MarketType.VOLATILE,  # Exit during volatile
            MarketType.ANY,
        ]),
        min_risk_reward=1.5,
        max_risk_reward=4.0,
        min_atr_mult=1.0,
        max_atr_mult=3.0,
    ),

    StrategyType.TIME_BASED: StrategyProfile(
        strategy_type=StrategyType.TIME_BASED,
        allowed_entry_patterns=frozenset([
            EntryPattern.TIME_TRIGGER,
            EntryPattern.THRESHOLD,
        ]),
        allowed_exit_patterns=frozenset([
            ExitPattern.TIME_EXIT,
            ExitPattern.FIXED_RR,
            ExitPattern.VOLATILITY_EXIT,
        ]),
        allowed_indicators=frozenset([
            Indicator.CLOSE, Indicator.HIGH, Indicator.LOW, Indicator.OPEN,
            Indicator.ATR, Indicator.SMA, Indicator.EMA,
            Indicator.RSI,  # For filtering
        ]),
        forbidden_indicators=frozenset([]),  # Time-based can use most
        allowed_market_types=frozenset([
            MarketType.ANY,
        ]),
        min_risk_reward=1.0,
        max_risk_reward=3.0,
        min_atr_mult=1.0,
        max_atr_mult=3.0,
    ),
}


def get_profile(strategy_type: StrategyType) -> StrategyProfile:
    """Get the permission profile for a strategy type"""
    return STRATEGY_PROFILES[strategy_type]


def is_indicator_allowed(strategy_type: StrategyType, indicator: Indicator) -> bool:
    """Check if an indicator is allowed for a strategy type"""
    profile = get_profile(strategy_type)
    if indicator in profile.forbidden_indicators:
        return False
    return indicator in profile.allowed_indicators


def is_market_type_allowed(strategy_type: StrategyType, market_type: MarketType) -> bool:
    """Check if a market type is allowed for a strategy type"""
    profile = get_profile(strategy_type)
    return market_type in profile.allowed_market_types


def get_allowed_indicators(strategy_type: StrategyType) -> Set[Indicator]:
    """Get all allowed indicators for a strategy type"""
    profile = get_profile(strategy_type)
    return set(profile.allowed_indicators)
