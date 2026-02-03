"""
Factory para criacao de estrategias
"""
from typing import Dict, Any, Optional, Type, List
from enum import Enum

from src.strategies.base import BaseStrategy, StrategyMetadata
from src.strategies.momentum import (
    MomentumStrategy,
    RiskAdjustedMomentumStrategy,
    DualMomentumStrategy
)
from config.settings import MomentumStrategyConfig


class StrategyType(Enum):
    """Tipos de estrategias disponiveis"""
    MOMENTUM = "momentum"
    MOMENTUM_RISK_ADJUSTED = "momentum_risk_adjusted"
    DUAL_MOMENTUM = "dual_momentum"


class StrategyRegistry:
    """Registro de estrategias disponiveis"""

    _strategies: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """Registra uma nova estrategia"""
        cls._strategies[name] = strategy_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseStrategy]]:
        """Obtem classe de estrategia pelo nome"""
        return cls._strategies.get(name)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """Lista estrategias disponiveis"""
        return list(cls._strategies.keys())

    @classmethod
    def get_metadata(cls, name: str) -> Optional[StrategyMetadata]:
        """Obtem metadados de uma estrategia"""
        strategy_class = cls.get(name)
        if strategy_class:
            instance = strategy_class()
            return instance.metadata
        return None


# Registra estrategias padrao
StrategyRegistry.register(StrategyType.MOMENTUM.value, MomentumStrategy)
StrategyRegistry.register(StrategyType.MOMENTUM_RISK_ADJUSTED.value, RiskAdjustedMomentumStrategy)
StrategyRegistry.register(StrategyType.DUAL_MOMENTUM.value, DualMomentumStrategy)


class StrategyFactory:
    """Factory para criacao de estrategias"""

    @staticmethod
    def create(
        strategy_type: StrategyType,
        parameters: Optional[Dict[str, Any]] = None,
        config: Optional[MomentumStrategyConfig] = None
    ) -> BaseStrategy:
        """
        Cria instancia de estrategia.

        Args:
            strategy_type: Tipo da estrategia
            parameters: Parametros customizados
            config: Configuracao da estrategia

        Returns:
            Instancia da estrategia
        """
        if strategy_type == StrategyType.MOMENTUM:
            return MomentumStrategy(parameters=parameters, config=config)

        elif strategy_type == StrategyType.MOMENTUM_RISK_ADJUSTED:
            return RiskAdjustedMomentumStrategy(parameters=parameters, config=config)

        elif strategy_type == StrategyType.DUAL_MOMENTUM:
            safe_asset = parameters.get('safe_asset', 'SHY') if parameters else 'SHY'
            threshold = parameters.get('absolute_threshold', 0.0) if parameters else 0.0
            return DualMomentumStrategy(
                parameters=parameters,
                config=config,
                safe_asset=safe_asset,
                absolute_threshold=threshold
            )

        else:
            raise ValueError(f"Estrategia desconhecida: {strategy_type}")

    @staticmethod
    def create_from_name(
        name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> BaseStrategy:
        """
        Cria estrategia a partir do nome registrado.

        Args:
            name: Nome da estrategia no registro
            parameters: Parametros customizados

        Returns:
            Instancia da estrategia
        """
        strategy_class = StrategyRegistry.get(name)

        if strategy_class is None:
            raise ValueError(f"Estrategia '{name}' nao encontrada no registro")

        return strategy_class(parameters=parameters)

    @staticmethod
    def list_available() -> List[Dict[str, Any]]:
        """Lista estrategias disponiveis com metadados"""
        strategies = []

        for name in StrategyRegistry.list_strategies():
            metadata = StrategyRegistry.get_metadata(name)
            if metadata:
                strategies.append({
                    'name': name,
                    'display_name': metadata.name,
                    'version': metadata.version,
                    'description': metadata.description,
                    'asset_classes': metadata.asset_classes,
                    'parameters': metadata.parameters
                })

        return strategies


def get_strategy_presets() -> Dict[str, Dict[str, Any]]:
    """
    Retorna presets de estrategias pre-configuradas.

    Returns:
        Dicionario com presets disponiveis
    """
    return {
        'conservative': {
            'strategy_type': StrategyType.MOMENTUM,
            'parameters': {
                'lookback_short': 63,
                'lookback_long': 126,
                'weight_short': 0.5,
                'min_return': 0.0,
                'top_n': 3
            },
            'description': 'Configuracao conservadora com mais diversificacao'
        },
        'aggressive': {
            'strategy_type': StrategyType.MOMENTUM,
            'parameters': {
                'lookback_short': 42,
                'lookback_long': 84,
                'weight_short': 0.7,
                'min_return': -0.10,
                'top_n': 2
            },
            'description': 'Configuracao agressiva com concentracao maior'
        },
        'balanced': {
            'strategy_type': StrategyType.MOMENTUM_RISK_ADJUSTED,
            'parameters': {
                'lookback_short': 63,
                'lookback_long': 126,
                'weight_short': 0.6,
                'min_return': -0.05,
                'top_n': 2
            },
            'description': 'Configuracao balanceada com ajuste de volatilidade'
        },
        'dual_momentum': {
            'strategy_type': StrategyType.DUAL_MOMENTUM,
            'parameters': {
                'lookback_short': 252,
                'safe_asset': 'SHY',
                'absolute_threshold': 0.0
            },
            'description': 'Dual momentum com protecao em renda fixa'
        },
        'trend_following': {
            'strategy_type': StrategyType.MOMENTUM,
            'parameters': {
                'lookback_short': 126,
                'lookback_long': 252,
                'lookback_sma': 200,
                'weight_short': 0.4,
                'min_return': -0.05,
                'top_n': 4
            },
            'description': 'Trend following de longo prazo'
        }
    }


def create_from_preset(preset_name: str) -> BaseStrategy:
    """
    Cria estrategia a partir de um preset.

    Args:
        preset_name: Nome do preset

    Returns:
        Instancia da estrategia configurada
    """
    presets = get_strategy_presets()

    if preset_name not in presets:
        raise ValueError(f"Preset '{preset_name}' nao encontrado. "
                        f"Disponiveis: {list(presets.keys())}")

    preset = presets[preset_name]

    return StrategyFactory.create(
        strategy_type=preset['strategy_type'],
        parameters=preset['parameters']
    )
