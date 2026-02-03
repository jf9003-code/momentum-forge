"""
Testes unitarios para o modulo de Portfolio
"""
import pytest
from datetime import datetime
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.portfolio import (
    Portfolio,
    Position,
    Order,
    Trade,
    PositionSide,
    OrderType,
    OrderStatus
)


class TestPosition:
    """Testes para a classe Position"""

    def test_position_creation(self):
        """Testa criacao de posicao"""
        pos = Position(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            side=PositionSide.LONG,
            entry_date=datetime.now(),
            last_price=155.0
        )

        assert pos.symbol == "AAPL"
        assert pos.quantity == 100
        assert pos.avg_price == 150.0
        assert pos.side == PositionSide.LONG

    def test_market_value_long(self):
        """Testa valor de mercado para posicao long"""
        pos = Position(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            side=PositionSide.LONG,
            entry_date=datetime.now(),
            last_price=160.0
        )

        assert pos.market_value == 16000.0

    def test_market_value_short(self):
        """Testa valor de mercado para posicao short"""
        pos = Position(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            side=PositionSide.SHORT,
            entry_date=datetime.now(),
            last_price=140.0
        )

        assert pos.market_value == -14000.0

    def test_update_price(self):
        """Testa atualizacao de preco"""
        pos = Position(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            side=PositionSide.LONG,
            entry_date=datetime.now()
        )

        pos.update_price(160.0)

        assert pos.last_price == 160.0
        assert pos.unrealized_pnl == 1000.0  # (160-150) * 100


class TestPortfolio:
    """Testes para a classe Portfolio"""

    def test_portfolio_creation(self):
        """Testa criacao de portfolio"""
        port = Portfolio(initial_capital=100000.0)

        assert port.initial_capital == 100000.0
        assert port.cash == 100000.0
        assert port.total_equity == 100000.0
        assert len(port.positions) == 0

    def test_create_order(self):
        """Testa criacao de ordem"""
        port = Portfolio(initial_capital=100000.0)

        order = port.create_order(
            symbol="AAPL",
            quantity=100,
            side=PositionSide.LONG,
            order_type=OrderType.MARKET
        )

        assert order.symbol == "AAPL"
        assert order.quantity == 100
        assert order.side == PositionSide.LONG
        assert order.status == OrderStatus.PENDING

    def test_execute_buy_order(self):
        """Testa execucao de ordem de compra"""
        port = Portfolio(initial_capital=100000.0)

        order = port.create_order(
            symbol="AAPL",
            quantity=100,
            side=PositionSide.LONG
        )

        trade = port.execute_order(
            order=order,
            fill_price=150.0,
            commission=10.0,
            timestamp=datetime.now()
        )

        assert trade is not None
        assert "AAPL" in port.positions
        assert port.positions["AAPL"].quantity == 100
        assert port.cash < 100000.0

    def test_close_position(self):
        """Testa fechamento de posicao"""
        port = Portfolio(initial_capital=100000.0)

        # Abre posicao
        order = port.create_order("AAPL", 100, PositionSide.LONG)
        port.execute_order(order, 150.0, timestamp=datetime.now())

        # Fecha posicao
        trade = port.close_position(
            symbol="AAPL",
            price=160.0,
            timestamp=datetime.now()
        )

        assert trade is not None
        assert "AAPL" not in port.positions

    def test_total_equity_calculation(self):
        """Testa calculo de equity total"""
        port = Portfolio(initial_capital=100000.0)

        # Compra 100 acoes a 100
        order = port.create_order("AAPL", 100, PositionSide.LONG)
        port.execute_order(order, 100.0, timestamp=datetime.now())

        # Atualiza preco para 110
        port.update_prices({"AAPL": 110.0}, datetime.now())

        # Cash reduzido + valor da posicao
        assert port.total_equity > port.initial_capital

    def test_leverage_calculation(self):
        """Testa calculo de alavancagem"""
        port = Portfolio(initial_capital=100000.0)

        order = port.create_order("AAPL", 500, PositionSide.LONG)
        port.execute_order(order, 100.0, timestamp=datetime.now())
        port.update_prices({"AAPL": 100.0}, datetime.now())

        assert port.leverage == pytest.approx(0.5, rel=0.1)

    def test_get_equity_series(self):
        """Testa obtencao de serie de equity"""
        port = Portfolio(initial_capital=100000.0)

        # Simula alguns dias
        for i in range(5):
            port.update_prices(
                {"AAPL": 100.0 + i},
                datetime(2024, 1, 1 + i)
            )

        series = port.get_equity_series()
        assert len(series) == 5

    def test_reset(self):
        """Testa reset do portfolio"""
        port = Portfolio(initial_capital=100000.0)

        order = port.create_order("AAPL", 100, PositionSide.LONG)
        port.execute_order(order, 150.0, timestamp=datetime.now())

        port.reset()

        assert port.cash == port.initial_capital
        assert len(port.positions) == 0
        assert len(port.trades) == 0


class TestOrder:
    """Testes para a classe Order"""

    def test_order_fill_ratio(self):
        """Testa calculo de fill ratio"""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            side=PositionSide.LONG,
            order_type=OrderType.MARKET,
            filled_quantity=50
        )

        assert order.fill_ratio == 0.5

    def test_order_is_complete(self):
        """Testa verificacao de ordem completa"""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            side=PositionSide.LONG,
            order_type=OrderType.MARKET
        )

        assert not order.is_complete

        order.status = OrderStatus.FILLED
        assert order.is_complete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
