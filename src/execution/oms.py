"""
Order Management System (OMS)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from enum import Enum
from collections import deque
import logging
import uuid

from src.core.portfolio import Order, Trade, OrderType, OrderStatus, PositionSide


logger = logging.getLogger(__name__)


class OrderPriority(Enum):
    """Prioridade de ordens"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class OrderState(Enum):
    """Estados do ciclo de vida da ordem"""
    CREATED = "created"
    VALIDATED = "validated"
    RISK_APPROVED = "risk_approved"
    ROUTED = "routed"
    ACKNOWLEDGED = "acknowledged"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderEvent:
    """Evento de ordem"""
    order_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedOrder(Order):
    """Ordem com campos adicionais para OMS"""
    priority: OrderPriority = OrderPriority.NORMAL
    state: OrderState = OrderState.CREATED
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    algo: Optional[str] = None
    algo_params: Dict[str, Any] = field(default_factory=dict)
    events: List[OrderEvent] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    expiry: Optional[datetime] = None

    def add_event(self, event_type: str, data: Dict = None) -> None:
        """Adiciona evento ao historico"""
        self.events.append(OrderEvent(
            order_id=self.order_id,
            event_type=event_type,
            timestamp=datetime.now(),
            data=data or {}
        ))

    def transition_to(self, new_state: OrderState) -> None:
        """Transicao de estado"""
        old_state = self.state
        self.state = new_state
        self.add_event('state_change', {
            'from': old_state.value,
            'to': new_state.value
        })


class OrderValidator:
    """Validador de ordens"""

    def __init__(self):
        self.rules: List[Callable[[EnhancedOrder], tuple]] = []

    def add_rule(self, rule: Callable[[EnhancedOrder], tuple]) -> None:
        """Adiciona regra de validacao"""
        self.rules.append(rule)

    def validate(self, order: EnhancedOrder) -> tuple:
        """
        Valida ordem contra todas as regras.

        Returns:
            (is_valid, list of errors)
        """
        errors = []

        # Validacoes basicas
        if order.quantity <= 0:
            errors.append("Quantidade deve ser positiva")

        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            errors.append("Ordem limite requer preco limite")

        if order.order_type == OrderType.STOP and order.stop_price is None:
            errors.append("Ordem stop requer preco stop")

        # Regras customizadas
        for rule in self.rules:
            is_valid, error = rule(order)
            if not is_valid:
                errors.append(error)

        return len(errors) == 0, errors


class OrderManagementSystem:
    """
    Sistema de gerenciamento de ordens.

    Gerencia todo o ciclo de vida das ordens:
    criacao, validacao, roteamento, execucao.
    """

    def __init__(
        self,
        validator: Optional[OrderValidator] = None,
        max_order_history: int = 10000
    ):
        self.validator = validator or OrderValidator()
        self.max_order_history = max_order_history

        # Ordens por estado
        self.active_orders: Dict[str, EnhancedOrder] = {}
        self.completed_orders: deque = deque(maxlen=max_order_history)

        # Callbacks
        self._on_order_created: List[Callable] = []
        self._on_order_filled: List[Callable] = []
        self._on_order_cancelled: List[Callable] = []
        self._on_order_rejected: List[Callable] = []

        # Contadores
        self._order_counter = 0

    def generate_order_id(self) -> str:
        """Gera ID unico para ordem"""
        self._order_counter += 1
        return f"OMS-{datetime.now().strftime('%Y%m%d')}-{self._order_counter:06d}"

    def create_order(
        self,
        symbol: str,
        quantity: float,
        side: PositionSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        priority: OrderPriority = OrderPriority.NORMAL,
        algo: Optional[str] = None,
        algo_params: Optional[Dict] = None,
        tags: Optional[Dict] = None
    ) -> EnhancedOrder:
        """Cria nova ordem"""
        order = EnhancedOrder(
            order_id=self.generate_order_id(),
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            priority=priority,
            algo=algo,
            algo_params=algo_params or {},
            tags=tags or {}
        )

        order.add_event('created', {
            'symbol': symbol,
            'quantity': quantity,
            'side': side.value,
            'type': order_type.value
        })

        logger.info(f"Ordem criada: {order.order_id} - {symbol} {side.value} {quantity}")

        # Callbacks
        for callback in self._on_order_created:
            callback(order)

        return order

    def submit_order(self, order: EnhancedOrder) -> tuple:
        """
        Submete ordem para processamento.

        Returns:
            (success, message)
        """
        # Valida
        is_valid, errors = self.validator.validate(order)

        if not is_valid:
            order.transition_to(OrderState.REJECTED)
            order.status = OrderStatus.REJECTED
            self.completed_orders.append(order)

            for callback in self._on_order_rejected:
                callback(order, errors)

            return False, f"Ordem rejeitada: {', '.join(errors)}"

        order.transition_to(OrderState.VALIDATED)

        # TODO: Verificacao de risco
        order.transition_to(OrderState.RISK_APPROVED)

        # Adiciona a ordens ativas
        self.active_orders[order.order_id] = order
        order.status = OrderStatus.SUBMITTED
        order.transition_to(OrderState.ROUTED)

        logger.info(f"Ordem submetida: {order.order_id}")

        return True, "Ordem submetida com sucesso"

    def cancel_order(self, order_id: str) -> tuple:
        """
        Cancela uma ordem.

        Returns:
            (success, message)
        """
        if order_id not in self.active_orders:
            return False, "Ordem nao encontrada"

        order = self.active_orders[order_id]

        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False, "Ordem ja finalizada"

        order.status = OrderStatus.CANCELLED
        order.transition_to(OrderState.CANCELLED)

        del self.active_orders[order_id]
        self.completed_orders.append(order)

        for callback in self._on_order_cancelled:
            callback(order)

        logger.info(f"Ordem cancelada: {order_id}")

        return True, "Ordem cancelada"

    def fill_order(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: Optional[float] = None,
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> Optional[Trade]:
        """
        Executa fill de uma ordem.

        Args:
            order_id: ID da ordem
            fill_price: Preco de execucao
            fill_quantity: Quantidade executada (None = full fill)
            commission: Comissao
            slippage: Slippage

        Returns:
            Trade se sucesso
        """
        if order_id not in self.active_orders:
            logger.error(f"Ordem nao encontrada: {order_id}")
            return None

        order = self.active_orders[order_id]
        fill_qty = fill_quantity or (order.quantity - order.filled_quantity)

        # Atualiza ordem
        order.filled_quantity += fill_qty
        order.avg_fill_price = (
            (order.avg_fill_price * (order.filled_quantity - fill_qty) +
             fill_price * fill_qty) / order.filled_quantity
            if order.filled_quantity > 0 else fill_price
        )
        order.commission += commission
        order.slippage += slippage
        order.filled_at = datetime.now()

        # Cria trade
        trade = Trade(
            trade_id=f"TRD-{uuid.uuid4().hex[:8]}",
            order_id=order_id,
            symbol=order.symbol,
            quantity=fill_qty,
            price=fill_price,
            side=order.side,
            commission=commission,
            slippage=slippage,
            timestamp=datetime.now()
        )

        # Atualiza estado
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.transition_to(OrderState.FILLED)
            del self.active_orders[order_id]
            self.completed_orders.append(order)

            for callback in self._on_order_filled:
                callback(order, trade)

            logger.info(f"Ordem executada: {order_id} @ {fill_price}")
        else:
            order.status = OrderStatus.PARTIAL
            order.transition_to(OrderState.PARTIAL_FILL)

        order.add_event('fill', {
            'price': fill_price,
            'quantity': fill_qty,
            'commission': commission
        })

        return trade

    def get_order(self, order_id: str) -> Optional[EnhancedOrder]:
        """Obtem ordem por ID"""
        if order_id in self.active_orders:
            return self.active_orders[order_id]

        for order in self.completed_orders:
            if order.order_id == order_id:
                return order

        return None

    def get_active_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[EnhancedOrder]:
        """Retorna ordens ativas"""
        orders = list(self.active_orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def get_orders_by_state(self, state: OrderState) -> List[EnhancedOrder]:
        """Retorna ordens por estado"""
        return [o for o in self.active_orders.values() if o.state == state]

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancela todas as ordens"""
        to_cancel = list(self.active_orders.keys())

        if symbol:
            to_cancel = [
                oid for oid, o in self.active_orders.items()
                if o.symbol == symbol
            ]

        cancelled = 0
        for order_id in to_cancel:
            success, _ = self.cancel_order(order_id)
            if success:
                cancelled += 1

        return cancelled

    def on_order_created(self, callback: Callable) -> None:
        """Registra callback para ordem criada"""
        self._on_order_created.append(callback)

    def on_order_filled(self, callback: Callable) -> None:
        """Registra callback para ordem executada"""
        self._on_order_filled.append(callback)

    def on_order_cancelled(self, callback: Callable) -> None:
        """Registra callback para ordem cancelada"""
        self._on_order_cancelled.append(callback)

    def on_order_rejected(self, callback: Callable) -> None:
        """Registra callback para ordem rejeitada"""
        self._on_order_rejected.append(callback)

    def get_statistics(self) -> Dict:
        """Retorna estatisticas do OMS"""
        all_orders = list(self.active_orders.values()) + list(self.completed_orders)

        filled = [o for o in all_orders if o.status == OrderStatus.FILLED]
        cancelled = [o for o in all_orders if o.status == OrderStatus.CANCELLED]
        rejected = [o for o in all_orders if o.status == OrderStatus.REJECTED]

        total_commission = sum(o.commission for o in filled)
        total_slippage = sum(o.slippage for o in filled)

        return {
            'total_orders': len(all_orders),
            'active_orders': len(self.active_orders),
            'filled_orders': len(filled),
            'cancelled_orders': len(cancelled),
            'rejected_orders': len(rejected),
            'fill_rate': len(filled) / len(all_orders) if all_orders else 0,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'avg_fill_time_ms': 0  # TODO: implementar
        }
