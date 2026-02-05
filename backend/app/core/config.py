"""
D5 ROBUST MASTER - Configuration Module
Centralized configuration management
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration"""
    path: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'robustmark.db'
    ))


@dataclass
class ValidationConfig:
    """Validation pipeline configuration"""
    # Phase thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown_limit: float = -30.0
    min_trades: int = 30
    min_win_rate: float = 35.0
    min_profit_factor: float = 1.1

    # Monte Carlo settings
    monte_carlo_iterations: int = 1000
    monte_carlo_confidence: float = 0.95

    # Walk-Forward settings
    wfa_windows: int = 10
    wfa_min_pass_rate: float = 0.6

    # Bootstrap Reality Check
    brc_iterations: int = 1000
    brc_p_value_threshold: float = 0.05


@dataclass
class GeneratorConfig:
    """Strategy generator configuration"""
    # Clone filter - CORRECTED VALUES
    min_structural_distance: float = 0.08  # Reduced from 0.15 to allow more diversity within templates

    # Output limits
    max_strategies_per_template: int = 50
    max_total_strategies: int = 200

    # Diversity weights for distance calculation
    template_weight: float = 0.5  # Reduced from 1.0
    strategy_type_weight: float = 1.0  # Reduced from 2.0
    entry_pattern_weight: float = 0.4  # Reduced from 0.8
    exit_pattern_weight: float = 0.3  # Reduced from 0.6
    indicator_weight: float = 0.5  # Reduced from 0.7
    parameter_weight: float = 2.5  # Increased from 2.0 - parameters matter more!


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class AppConfig:
    """Main application configuration"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")

    def __post_init__(self):
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance"""
    return config
