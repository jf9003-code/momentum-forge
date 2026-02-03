"""
Algoritmos de Execucao
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from enum import Enum

from src.core.portfolio import Order, PositionSide


class AlgoType(Enum):
    """Tipos de algoritmos de execucao"""
    MARKET = "market"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"  # Percentage of Volume
    IS = "is"    # Implementation Shortfall
    ICEBERG = "iceberg"


@dataclass
class AlgoSlice:
    """Fatia de execucao do algoritmo"""
    slice_id: int
    quantity: float
    target_time: datetime
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    executed_qty: float = 0.0
    executed_price: float = 0.0
    status: str = "pending"


@dataclass
class AlgoExecution:
    """Resultado de execucao do algoritmo"""
    order_id: str
    algo_type: AlgoType
    slices: List[AlgoSlice]
    start_time: datetime
    end_time: Optional[datetime] = None
    total_quantity: float = 0.0
    avg_price: float = 0.0
    participation_rate: float = 0.0
    benchmark_price: float = 0.0
    slippage_bps: float = 0.0


class ExecutionAlgorithm(ABC):
    """Classe base para algoritmos de execucao"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_schedule(
        self,
        quantity: float,
        side: PositionSide,
        start_time: datetime,
        end_time: datetime,
        params: Dict[str, Any]
    ) -> List[AlgoSlice]:
        """Gera schedule de execucao"""
        pass

    @abstractmethod
    def should_execute_slice(
        self,
        slice: AlgoSlice,
        current_price: float,
        current_volume: float
    ) -> bool:
        """Determina se deve executar a fatia"""
        pass


class TWAPAlgorithm(ExecutionAlgorithm):
    """
    Time-Weighted Average Price.

    Divide a ordem em fatias iguais ao longo do tempo.
    """

    def __init__(self):
        super().__init__("TWAP")

    def generate_schedule(
        self,
        quantity: float,
        side: PositionSide,
        start_time: datetime,
        end_time: datetime,
        params: Dict[str, Any]
    ) -> List[AlgoSlice]:
        """Gera schedule TWAP"""
        num_slices = params.get('num_slices', 10)
        min_slice_size = params.get('min_slice_size', 100)

        # Calcula tamanho de cada fatia
        slice_qty = max(quantity / num_slices, min_slice_size)
        actual_slices = int(np.ceil(quantity / slice_qty))

        # Distribui no tempo
        duration = (end_time - start_time).total_seconds()
        interval = duration / actual_slices

        slices = []
        remaining = quantity

        for i in range(actual_slices):
            qty = min(slice_qty, remaining)
            target_time = start_time + timedelta(seconds=i * interval)

            slices.append(AlgoSlice(
                slice_id=i,
                quantity=qty,
                target_time=target_time
            ))

            remaining -= qty

        return slices

    def should_execute_slice(
        self,
        slice: AlgoSlice,
        current_price: float,
        current_volume: float
    ) -> bool:
        """Executa se chegou a hora"""
        return datetime.now() >= slice.target_time


class VWAPAlgorithm(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price.

    Distribui a ordem proporcional ao perfil de volume historico.
    """

    def __init__(self, volume_profile: Optional[Dict[int, float]] = None):
        super().__init__("VWAP")
        # Perfil de volume por hora (0-23) - default: distribuicao U-shape
        self.volume_profile = volume_profile or {
            9: 0.15, 10: 0.12, 11: 0.08, 12: 0.06, 13: 0.06,
            14: 0.08, 15: 0.12, 16: 0.18, 17: 0.15
        }

    def generate_schedule(
        self,
        quantity: float,
        side: PositionSide,
        start_time: datetime,
        end_time: datetime,
        params: Dict[str, Any]
    ) -> List[AlgoSlice]:
        """Gera schedule VWAP baseado no perfil de volume"""
        min_slice_size = params.get('min_slice_size', 100)

        # Horas de trading
        start_hour = max(9, start_time.hour)
        end_hour = min(17, end_time.hour)

        # Calcula volume proporcional por hora
        hours = list(range(start_hour, end_hour + 1))
        volumes = [self.volume_profile.get(h, 0.1) for h in hours]
        total_vol = sum(volumes)

        if total_vol == 0:
            total_vol = 1

        slices = []
        remaining = quantity

        for i, hour in enumerate(hours):
            # Proporcao do volume
            proportion = volumes[i] / total_vol
            qty = max(quantity * proportion, min_slice_size)
            qty = min(qty, remaining)

            if qty <= 0:
                continue

            target_time = start_time.replace(hour=hour, minute=0, second=0)
            if target_time < start_time:
                target_time = start_time

            slices.append(AlgoSlice(
                slice_id=i,
                quantity=qty,
                target_time=target_time
            ))

            remaining -= qty

        return slices

    def should_execute_slice(
        self,
        slice: AlgoSlice,
        current_price: float,
        current_volume: float
    ) -> bool:
        """Executa baseado no tempo e volume"""
        if datetime.now() < slice.target_time:
            return False

        return True


class POVAlgorithm(ExecutionAlgorithm):
    """
    Percentage of Volume.

    Executa como percentual do volume de mercado.
    """

    def __init__(self, target_pov: float = 0.10):
        super().__init__("POV")
        self.target_pov = target_pov  # 10% do volume

    def generate_schedule(
        self,
        quantity: float,
        side: PositionSide,
        start_time: datetime,
        end_time: datetime,
        params: Dict[str, Any]
    ) -> List[AlgoSlice]:
        """POV nao usa schedule fixo - responde ao mercado"""
        target_pov = params.get('target_pov', self.target_pov)

        # Cria uma unica fatia "virtual"
        return [AlgoSlice(
            slice_id=0,
            quantity=quantity,
            target_time=start_time,
            status="active"
        )]

    def should_execute_slice(
        self,
        slice: AlgoSlice,
        current_price: float,
        current_volume: float
    ) -> bool:
        """Executa proporcional ao volume"""
        return True

    def calculate_slice_quantity(
        self,
        remaining_qty: float,
        period_volume: float
    ) -> float:
        """Calcula quantidade para o periodo"""
        target_qty = period_volume * self.target_pov
        return min(target_qty, remaining_qty)


class IcebergAlgorithm(ExecutionAlgorithm):
    """
    Iceberg / Reserve Order.

    Mostra apenas parte da ordem no book.
    """

    def __init__(self, display_size: float = 100):
        super().__init__("Iceberg")
        self.display_size = display_size

    def generate_schedule(
        self,
        quantity: float,
        side: PositionSide,
        start_time: datetime,
        end_time: datetime,
        params: Dict[str, Any]
    ) -> List[AlgoSlice]:
        """Gera fatias do tamanho de display"""
        display = params.get('display_size', self.display_size)
        randomize = params.get('randomize', True)

        slices = []
        remaining = quantity
        i = 0

        while remaining > 0:
            if randomize:
                # Varia o tamanho +/- 20%
                qty = display * (0.8 + 0.4 * np.random.random())
            else:
                qty = display

            qty = min(qty, remaining)

            slices.append(AlgoSlice(
                slice_id=i,
                quantity=qty,
                target_time=start_time
            ))

            remaining -= qty
            i += 1

        return slices

    def should_execute_slice(
        self,
        slice: AlgoSlice,
        current_price: float,
        current_volume: float
    ) -> bool:
        """Executa proxima fatia quando atual for preenchida"""
        return slice.status == "pending"


class AlgorithmFactory:
    """Factory para algoritmos de execucao"""

    _algorithms = {
        AlgoType.TWAP: TWAPAlgorithm,
        AlgoType.VWAP: VWAPAlgorithm,
        AlgoType.POV: POVAlgorithm,
        AlgoType.ICEBERG: IcebergAlgorithm
    }

    @classmethod
    def create(
        cls,
        algo_type: AlgoType,
        **kwargs
    ) -> ExecutionAlgorithm:
        """Cria instancia de algoritmo"""
        algo_class = cls._algorithms.get(algo_type)

        if algo_class is None:
            raise ValueError(f"Algoritmo desconhecido: {algo_type}")

        return algo_class(**kwargs)

    @classmethod
    def list_algorithms(cls) -> List[str]:
        """Lista algoritmos disponiveis"""
        return [a.value for a in cls._algorithms.keys()]


class SmartOrderRouter:
    """
    Roteador inteligente de ordens.

    Seleciona melhor venue/algoritmo para execucao.
    """

    def __init__(self):
        self.venues: Dict[str, Dict] = {}
        self.default_algo = AlgoType.TWAP

    def add_venue(
        self,
        name: str,
        fee_bps: float,
        latency_ms: float,
        liquidity_score: float
    ) -> None:
        """Adiciona venue"""
        self.venues[name] = {
            'fee_bps': fee_bps,
            'latency_ms': latency_ms,
            'liquidity_score': liquidity_score
        }

    def select_venue(
        self,
        symbol: str,
        quantity: float,
        urgency: float = 0.5
    ) -> str:
        """
        Seleciona melhor venue.

        Args:
            symbol: Simbolo do ativo
            quantity: Quantidade
            urgency: 0=paciente, 1=urgente

        Returns:
            Nome do venue selecionado
        """
        if not self.venues:
            return "default"

        # Score = liquidez - custo + (latencia se urgente)
        scores = {}
        for name, info in self.venues.items():
            score = (
                info['liquidity_score'] -
                info['fee_bps'] / 10 -
                (info['latency_ms'] / 100 * urgency)
            )
            scores[name] = score

        return max(scores, key=scores.get)

    def select_algorithm(
        self,
        quantity: float,
        avg_daily_volume: float,
        urgency: float = 0.5,
        horizon_minutes: int = 60
    ) -> AlgoType:
        """
        Seleciona algoritmo baseado nas caracteristicas da ordem.

        Args:
            quantity: Quantidade da ordem
            avg_daily_volume: Volume medio diario
            urgency: Urgencia (0-1)
            horizon_minutes: Horizonte de execucao

        Returns:
            Tipo de algoritmo recomendado
        """
        participation = quantity / avg_daily_volume if avg_daily_volume > 0 else 0

        # Ordens pequenas (<1% ADV): Market direto
        if participation < 0.01:
            return AlgoType.MARKET

        # Ordens grandes (>5% ADV) ou baixa urgencia: VWAP
        if participation > 0.05 or urgency < 0.3:
            return AlgoType.VWAP

        # Urgencia alta: POV para nao impactar
        if urgency > 0.7:
            return AlgoType.POV

        # Default: TWAP
        return AlgoType.TWAP
