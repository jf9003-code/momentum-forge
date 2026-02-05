"""
D5 ROBUST MASTER - Parameter Definitions
Defines valid parameter ranges and distance calculations for clone detection.
"""

from typing import Dict, Tuple, Any, Optional
from .models import Indicator, Timeframe, StrategyType

# Default risk percentage
DEFAULT_RISK_PERCENT = 2.0


# ============================================
# INDICATOR PERIODS BY TIMEFRAME
# ============================================

LOOKBACK_PERIODS = {
    # Oscillators - shorter periods for lower timeframes
    Indicator.RSI: {
        Timeframe.M5: (5, 7, 10, 14),
        Timeframe.M15: (7, 10, 14, 21),
        Timeframe.H1: (7, 10, 14, 21, 28),
        Timeframe.H4: (10, 14, 21, 28),
        Timeframe.D1: (7, 14, 21, 28, 50),
    },
    Indicator.STOCH_K: {
        Timeframe.M5: (5, 7, 10, 14),
        Timeframe.M15: (7, 10, 14),
        Timeframe.H1: (7, 14, 21),
        Timeframe.H4: (14, 21, 28),
        Timeframe.D1: (14, 21, 28, 50),
    },
    Indicator.CCI: {
        Timeframe.M5: (10, 14, 20),
        Timeframe.M15: (14, 20, 50),
        Timeframe.H1: (14, 20, 50),
        Timeframe.H4: (20, 50, 100),
        Timeframe.D1: (14, 20, 50, 100),
    },

    # Moving Averages
    Indicator.SMA: {
        Timeframe.M5: (5, 10, 20, 50),
        Timeframe.M15: (10, 20, 50, 100),
        Timeframe.H1: (10, 20, 50, 100, 200),
        Timeframe.H4: (20, 50, 100, 200),
        Timeframe.D1: (10, 20, 50, 100, 200),
    },
    Indicator.EMA: {
        Timeframe.M5: (5, 10, 20, 50),
        Timeframe.M15: (10, 20, 50, 100),
        Timeframe.H1: (10, 20, 50, 100, 200),
        Timeframe.H4: (20, 50, 100, 200),
        Timeframe.D1: (8, 13, 21, 50, 100, 200),
    },

    # Volatility
    Indicator.ATR: {
        Timeframe.M5: (7, 10, 14),
        Timeframe.M15: (10, 14, 20),
        Timeframe.H1: (10, 14, 20),
        Timeframe.H4: (14, 20, 50),
        Timeframe.D1: (10, 14, 20, 50),
    },
    Indicator.STDDEV: {
        Timeframe.M5: (10, 20),
        Timeframe.M15: (10, 20, 50),
        Timeframe.H1: (10, 20, 50),
        Timeframe.H4: (20, 50),
        Timeframe.D1: (10, 20, 50),
    },

    # Breakout
    Indicator.DONCHIAN_HIGH: {
        Timeframe.M5: (10, 20, 50),
        Timeframe.M15: (20, 50, 100),
        Timeframe.H1: (20, 50, 100, 200),
        Timeframe.H4: (20, 50, 100),
        Timeframe.D1: (10, 20, 50, 100),
    },
    Indicator.DONCHIAN_LOW: {
        Timeframe.M5: (10, 20, 50),
        Timeframe.M15: (20, 50, 100),
        Timeframe.H1: (20, 50, 100, 200),
        Timeframe.H4: (20, 50, 100),
        Timeframe.D1: (10, 20, 50, 100),
    },

    # Momentum
    Indicator.ROC: {
        Timeframe.M5: (5, 10, 14),
        Timeframe.M15: (10, 14, 20),
        Timeframe.H1: (10, 14, 20, 50),
        Timeframe.H4: (14, 20, 50),
        Timeframe.D1: (10, 14, 20, 50),
    },
    Indicator.ADX: {
        Timeframe.M5: (7, 14),
        Timeframe.M15: (14, 20),
        Timeframe.H1: (14, 20, 28),
        Timeframe.H4: (14, 20, 28),
        Timeframe.D1: (14, 20, 28),
    },
}

# Default periods for indicators not in the lookup
DEFAULT_PERIODS = {
    Timeframe.M5: (5, 10, 14, 20),
    Timeframe.M15: (10, 14, 20, 50),
    Timeframe.H1: (10, 14, 20, 50),
    Timeframe.H4: (14, 20, 50, 100),
    Timeframe.D1: (14, 20, 50, 100, 200),
}


def get_lookback_periods(indicator: Indicator, timeframe: Timeframe) -> Tuple[int, ...]:
    """Get valid lookback periods for an indicator on a timeframe"""
    if indicator in LOOKBACK_PERIODS:
        if timeframe in LOOKBACK_PERIODS[indicator]:
            return LOOKBACK_PERIODS[indicator][timeframe]
    return DEFAULT_PERIODS.get(timeframe, (14, 20, 50))


# ============================================
# THRESHOLD VALUES
# ============================================

THRESHOLD_VALUES = {
    Indicator.RSI: {
        "oversold": (20, 25, 30),
        "overbought": (70, 75, 80),
        "momentum_long": (50, 55, 60),
        "momentum_short": (40, 45, 50),
    },
    Indicator.STOCH_K: {
        "oversold": (15, 20, 25),
        "overbought": (75, 80, 85),
    },
    Indicator.CCI: {
        "cci_oversold": (-200, -150, -100),
        "cci_overbought": (100, 150, 200),
    },
    Indicator.WILLIAMS_R: {
        "oversold": (-90, -85, -80),
        "overbought": (-20, -15, -10),
    },
}


def get_threshold_values(indicator: Indicator) -> Dict[str, Tuple]:
    """Get threshold values for an indicator"""
    return THRESHOLD_VALUES.get(indicator, {})


# ============================================
# ATR MULTIPLIERS BY STRATEGY TYPE
# ============================================

ATR_MULTIPLIERS = {
    "mean_reversion": (1.0, 1.5, 2.0),
    "momentum": (1.5, 2.0, 2.5, 3.0),
    "breakout": (1.5, 2.0, 2.5, 3.0),
    "volatility_expansion": (1.0, 1.5, 2.0),
    "time_based": (1.0, 1.5, 2.0),
}


def get_atr_multipliers(strategy_type: str) -> Tuple[float, ...]:
    """Get valid ATR multipliers for a strategy type"""
    return ATR_MULTIPLIERS.get(strategy_type, (1.5, 2.0, 2.5))


# ============================================
# RISK-REWARD RATIOS
# ============================================

def get_rr_ratios(min_rr: float = 1.0) -> Tuple[float, ...]:
    """Get valid risk-reward ratios starting from minimum"""
    all_rr = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    return tuple(rr for rr in all_rr if rr >= min_rr)


# ============================================
# TIME EXIT BARS
# ============================================

TIME_EXIT_BARS = {
    Timeframe.M5: (6, 12, 24, 48),      # 30m to 4h
    Timeframe.M15: (4, 8, 16, 32),      # 1h to 8h
    Timeframe.H1: (4, 8, 24, 48),       # 4h to 2 days
    Timeframe.H4: (3, 6, 12, 24),       # 12h to 4 days
    Timeframe.D1: (3, 5, 10, 20),       # 3 to 20 days
}


def get_time_exit_bars(timeframe: Timeframe) -> Tuple[int, ...]:
    """Get valid time exit bars for a timeframe"""
    return TIME_EXIT_BARS.get(timeframe, (5, 10, 20))


# ============================================
# PARAMETER DISTANCE CALCULATION - IMPROVED
# ============================================

def parameter_distance(params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
    """
    Calculate normalized distance between two parameter sets.
    Returns value in [0, 1] where 0 = identical, 1 = completely different.

    IMPROVED: Better handling of numeric parameters with relative differences.
    """
    if not params1 and not params2:
        return 0.0
    if not params1 or not params2:
        return 1.0

    # Get all keys
    all_keys = set(params1.keys()) | set(params2.keys())
    if not all_keys:
        return 0.0

    total_distance = 0.0
    total_weight = 0.0

    # Define weights for different parameter types
    param_weights = {
        "period": 1.5,
        "threshold": 1.0,
        "atr_sl_mult": 2.0,
        "atr_tp_mult": 2.0,
        "risk_reward": 1.5,
        "time_exit_bars": 1.0,
    }

    for key in all_keys:
        v1 = params1.get(key)
        v2 = params2.get(key)

        # Get weight for this parameter type
        weight = 1.0
        for param_type, w in param_weights.items():
            if param_type in key.lower():
                weight = w
                break

        total_weight += weight

        # Calculate distance for this parameter
        if v1 is None or v2 is None:
            # One is missing
            total_distance += weight
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            # Numeric comparison - use relative difference
            if v1 == v2:
                dist = 0.0
            elif v1 == 0 and v2 == 0:
                dist = 0.0
            else:
                max_val = max(abs(v1), abs(v2))
                min_val = min(abs(v1), abs(v2))
                if max_val > 0:
                    # Relative difference: how different are they proportionally?
                    dist = abs(v1 - v2) / max_val
                    # Clamp to [0, 1]
                    dist = min(1.0, dist)
                else:
                    dist = 0.0
            total_distance += weight * dist
        elif v1 != v2:
            # Non-numeric, different values
            total_distance += weight

    return total_distance / total_weight if total_weight > 0 else 0.0


def is_parameter_clone(params1: Dict, params2: Dict, threshold: float = 0.15) -> bool:
    """Check if two parameter sets are too similar (clones)"""
    return parameter_distance(params1, params2) < threshold
