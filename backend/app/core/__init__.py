"""
D5 ROBUST MASTER - Core Module
Database, Config, and Validation Pipeline
"""

from .database import (
    get_connection,
    init_database,
    save_validation,
    get_all_validations,
    get_validation,
    delete_validation,
    get_statistics,
    save_generated_strategies,
    get_generated_strategies,
    clear_generated_strategies,
    save_top5_strategies,
    get_top5_strategies,
    clear_top5_strategies,
    delete_top5_strategy,
)

__all__ = [
    "get_connection",
    "init_database",
    "save_validation",
    "get_all_validations",
    "get_validation",
    "delete_validation",
    "get_statistics",
    "save_generated_strategies",
    "get_generated_strategies",
    "clear_generated_strategies",
    "save_top5_strategies",
    "get_top5_strategies",
    "clear_top5_strategies",
    "delete_top5_strategy",
]
