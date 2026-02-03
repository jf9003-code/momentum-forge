"""
Gestao de Portfolio Institucional
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Position:
    """Representa uma posicao no portfolio"""
    symbol: str
    quantity: float
    avg_price: float
    side: PositionSide
    entry_date: datetime
    last_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    sector: Optional[str] = None

    @property
    def market_value(self) -> float:
        """Valor de mercado da posicao"""
        sign = 1 if self.side == PositionSide.LONG else -1
        return sign * self.quantity * self.last_price

    @property
    def cost_basis(self) -> float:
        """Custo base da posicao"""
        return self.quantity * self.avg_price

    @property
    def weight(self) -> float:
        """Peso no portfolio (precisa ser calculado externamente)"""
        return 0.0

    def update_price(self, price: float) -> None:
        """Atualiza preco e PnL nao realizado"""
        self.last_price = price
        sign = 1 if self.side == PositionSide.LONG else -1
        self.unrealized_pnl = sign * self.quantity * (price - self.avg_price)

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'side': self.side.value,
            'entry_date': self.entry_date,
            'last_price': self.last_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }


@dataclass
class Order:
    """Representa uma ordem"""
    order_id: str
    symbol: str
    quantity: float
    side: PositionSide
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None

    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]

    @property
    def fill_ratio(self) -> float:
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity


@dataclass
class Trade:
    """Representa uma execucao de trade"""
    trade_id: str
    order_id: str
    symbol: str
    quantity: float
    price: float
    side: PositionSide
    commission: float
    slippage: float
    timestamp: datetime

    @property
    def total_cost(self) -> float:
        """Custo total incluindo comissao e slippage"""
        return self.quantity * self.price + self.commission + self.slippage


class Portfolio:
    """Gerenciador de Portfolio Institucional"""

    def __init__(self, initial_capital: float, currency: str = "USD"):
        self.initial_capital = initial_capital
        self.currency = currency
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self.transaction_log: List[dict] = []
        self._order_counter = 0
        self._trade_counter = 0

    @property
    def total_equity(self) -> float:
        """Valor total do portfolio (cash + posicoes)"""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def total_exposure(self) -> float:
        """Exposicao total (long + short absoluto)"""
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        """Exposicao liquida (long - short)"""
        return sum(p.market_value for p in self.positions.values())

    @property
    def leverage(self) -> float:
        """Alavancagem atual"""
        if self.total_equity == 0:
            return 0.0
        return self.total_exposure / self.total_equity

    @property
    def long_exposure(self) -> float:
        """Exposicao long"""
        return sum(p.market_value for p in self.positions.values()
                   if p.side == PositionSide.LONG)

    @property
    def short_exposure(self) -> float:
        """Exposicao short (valor absoluto)"""
        return abs(sum(p.market_value for p in self.positions.values()
                       if p.side == PositionSide.SHORT))

    def get_position_weights(self) -> Dict[str, float]:
        """Retorna pesos de cada posicao"""
        total = self.total_equity
        if total == 0:
            return {}
        return {symbol: pos.market_value / total
                for symbol, pos in self.positions.items()}

    def get_sector_exposure(self) -> Dict[str, float]:
        """Retorna exposicao por setor"""
        sector_exposure = defaultdict(float)
        for pos in self.positions.values():
            sector = pos.sector or "Unknown"
            sector_exposure[sector] += pos.market_value
        return dict(sector_exposure)

    def update_prices(self, prices: Dict[str, float], timestamp: datetime) -> None:
        """Atualiza precos de todas as posicoes"""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])

        self.equity_history.append((timestamp, self.total_equity))

    def create_order(
        self,
        symbol: str,
        quantity: float,
        side: PositionSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """Cria uma nova ordem"""
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:08d}"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price
        )

        self.orders[order_id] = order
        return order

    def execute_order(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: Optional[float] = None,
        commission: float = 0.0,
        slippage: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> Trade:
        """Executa uma ordem (total ou parcial)"""
        if timestamp is None:
            timestamp = datetime.now()

        fill_qty = fill_quantity or order.quantity

        # Atualiza ordem
        order.filled_quantity += fill_qty
        order.avg_fill_price = (
            (order.avg_fill_price * (order.filled_quantity - fill_qty) +
             fill_price * fill_qty) / order.filled_quantity
            if order.filled_quantity > 0 else fill_price
        )
        order.commission += commission
        order.slippage += slippage

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = timestamp
        else:
            order.status = OrderStatus.PARTIAL

        # Cria trade
        self._trade_counter += 1
        trade = Trade(
            trade_id=f"TRD-{self._trade_counter:08d}",
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=fill_qty,
            price=fill_price,
            side=order.side,
            commission=commission,
            slippage=slippage,
            timestamp=timestamp
        )
        self.trades.append(trade)

        # Atualiza posicao
        self._update_position(trade)

        # Log
        self.transaction_log.append({
            'timestamp': timestamp,
            'type': 'trade',
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': fill_qty,
            'price': fill_price,
            'commission': commission,
            'slippage': slippage
        })

        return trade

    def _update_position(self, trade: Trade) -> None:
        """Atualiza posicao com base no trade"""
        symbol = trade.symbol

        # Custo total da transacao
        total_cost = trade.quantity * trade.price + trade.commission + trade.slippage

        if symbol not in self.positions:
            # Nova posicao
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=trade.quantity,
                avg_price=trade.price,
                side=trade.side,
                entry_date=trade.timestamp,
                last_price=trade.price
            )
            if trade.side == PositionSide.LONG:
                self.cash -= total_cost
            else:
                self.cash += trade.quantity * trade.price - trade.commission - trade.slippage
        else:
            position = self.positions[symbol]

            if position.side == trade.side:
                # Aumentando posicao
                new_quantity = position.quantity + trade.quantity
                position.avg_price = (
                    (position.avg_price * position.quantity + trade.price * trade.quantity) /
                    new_quantity
                )
                position.quantity = new_quantity

                if trade.side == PositionSide.LONG:
                    self.cash -= total_cost
                else:
                    self.cash += trade.quantity * trade.price - trade.commission - trade.slippage
            else:
                # Reduzindo ou fechando posicao
                if trade.quantity >= position.quantity:
                    # Fechando posicao
                    realized_pnl = (trade.price - position.avg_price) * position.quantity
                    if position.side == PositionSide.SHORT:
                        realized_pnl = -realized_pnl

                    position.realized_pnl += realized_pnl

                    if trade.side == PositionSide.LONG:
                        self.cash -= total_cost
                    else:
                        self.cash += position.quantity * trade.price - trade.commission - trade.slippage

                    remaining = trade.quantity - position.quantity

                    if remaining > 0:
                        # Invertendo posicao
                        position.quantity = remaining
                        position.side = trade.side
                        position.avg_price = trade.price
                        position.entry_date = trade.timestamp
                    else:
                        # Posicao fechada
                        del self.positions[symbol]
                else:
                    # Reduzindo posicao
                    realized_pnl = (trade.price - position.avg_price) * trade.quantity
                    if position.side == PositionSide.SHORT:
                        realized_pnl = -realized_pnl

                    position.realized_pnl += realized_pnl
                    position.quantity -= trade.quantity

                    if trade.side == PositionSide.LONG:
                        self.cash -= total_cost
                    else:
                        self.cash += trade.quantity * trade.price - trade.commission - trade.slippage

    def close_position(
        self,
        symbol: str,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> Optional[Trade]:
        """Fecha uma posicao existente"""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        close_side = PositionSide.SHORT if position.side == PositionSide.LONG else PositionSide.LONG

        order = self.create_order(
            symbol=symbol,
            quantity=position.quantity,
            side=close_side
        )

        return self.execute_order(
            order=order,
            fill_price=price,
            commission=commission,
            slippage=slippage,
            timestamp=timestamp
        )

    def close_all_positions(
        self,
        prices: Dict[str, float],
        commission_per_trade: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> List[Trade]:
        """Fecha todas as posicoes"""
        trades = []
        symbols = list(self.positions.keys())

        for symbol in symbols:
            if symbol in prices:
                trade = self.close_position(
                    symbol=symbol,
                    price=prices[symbol],
                    commission=commission_per_trade,
                    timestamp=timestamp
                )
                if trade:
                    trades.append(trade)

        return trades

    def get_equity_series(self) -> pd.Series:
        """Retorna serie temporal de equity"""
        if not self.equity_history:
            return pd.Series(dtype=float)

        dates, values = zip(*self.equity_history)
        return pd.Series(values, index=pd.DatetimeIndex(dates), name='equity')

    def get_returns_series(self) -> pd.Series:
        """Retorna serie de retornos"""
        equity = self.get_equity_series()
        if len(equity) < 2:
            return pd.Series(dtype=float)
        return equity.pct_change().dropna()

    def get_trades_df(self) -> pd.DataFrame:
        """Retorna DataFrame com historico de trades"""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'trade_id': t.trade_id,
                'order_id': t.order_id,
                'timestamp': t.timestamp,
                'symbol': t.symbol,
                'side': t.side.value,
                'quantity': t.quantity,
                'price': t.price,
                'commission': t.commission,
                'slippage': t.slippage,
                'total_cost': t.total_cost
            }
            for t in self.trades
        ])

    def get_positions_df(self) -> pd.DataFrame:
        """Retorna DataFrame com posicoes atuais"""
        if not self.positions:
            return pd.DataFrame()

        return pd.DataFrame([p.to_dict() for p in self.positions.values()])

    def snapshot(self) -> dict:
        """Retorna snapshot atual do portfolio"""
        return {
            'timestamp': datetime.now(),
            'cash': self.cash,
            'total_equity': self.total_equity,
            'total_exposure': self.total_exposure,
            'net_exposure': self.net_exposure,
            'leverage': self.leverage,
            'long_exposure': self.long_exposure,
            'short_exposure': self.short_exposure,
            'num_positions': len(self.positions),
            'num_trades': len(self.trades),
            'positions': self.get_position_weights()
        }

    def reset(self) -> None:
        """Reseta portfolio para estado inicial"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.equity_history.clear()
        self.transaction_log.clear()
        self._order_counter = 0
        self._trade_counter = 0
