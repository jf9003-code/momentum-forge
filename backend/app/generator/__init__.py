"""
D5 ROBUST MASTER - Institutional Strategy Generator
Main Package

This generator follows institutional principles:
1. Top-down composition, not brute force
2. Structural validation before any backtest
3. Diversity enforcement with CORRECTED algorithm
4. Coverage monitoring
"""

from .models import (
    StrategySpec,
    StrategyType,
    Timeframe,
    MarketType,
    EntryLogic,
    ExitLogic,
    RiskConfig,
    EntryPattern,
    ExitPattern,
    RiskModel,
    Indicator,
    RelationalOperator,
    Condition,
    IndicatorConfig,
)

from .generator import (
    InstitutionalGenerator,
    GeneratorConfig,
    GenerationResult,
    generate_strategies,
    generate_for_type,
)

from .validators import (
    validate_strategy,
    calculate_structural_distance,
    is_diverse_from_set,
    ValidationResult,
)

from .enumerator import (
    ControlledEnumerator,
    EnumerationConfig,
)

__all__ = [
    # Models
    "StrategySpec",
    "StrategyType",
    "Timeframe",
    "MarketType",
    "EntryLogic",
    "ExitLogic",
    "RiskConfig",
    "EntryPattern",
    "ExitPattern",
    "RiskModel",
    "Indicator",
    "RelationalOperator",
    "Condition",
    "IndicatorConfig",
    # Generator
    "InstitutionalGenerator",
    "GeneratorConfig",
    "GenerationResult",
    "generate_strategies",
    "generate_for_type",
    # Validators
    "validate_strategy",
    "calculate_structural_distance",
    "is_diverse_from_set",
    "ValidationResult",
    # Enumerator
    "ControlledEnumerator",
    "EnumerationConfig",
]
