"""
Coletor de Metricas para Monitoramento
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time
from enum import Enum


class MetricType(Enum):
    """Tipos de metricas"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """Ponto de metrica"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricDefinition:
    """Definicao de uma metrica"""
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    labels: List[str] = field(default_factory=list)


class Counter:
    """Metrica do tipo contador"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0) -> None:
        """Incrementa contador"""
        with self._lock:
            self._value += value

    def get(self) -> float:
        """Obtem valor atual"""
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reseta contador"""
        with self._lock:
            self._value = 0.0


class Gauge:
    """Metrica do tipo gauge (valor instantaneo)"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Define valor"""
        with self._lock:
            self._value = value

    def inc(self, value: float = 1.0) -> None:
        """Incrementa valor"""
        with self._lock:
            self._value += value

    def dec(self, value: float = 1.0) -> None:
        """Decrementa valor"""
        with self._lock:
            self._value -= value

    def get(self) -> float:
        """Obtem valor atual"""
        with self._lock:
            return self._value


class Histogram:
    """Metrica do tipo histograma"""

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: List[float] = None
    ):
        self.name = name
        self.description = description
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        self._values: List[float] = []
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """Observa um valor"""
        with self._lock:
            self._values.append(value)

    def get_buckets(self) -> Dict[float, int]:
        """Obtem contagem por bucket"""
        with self._lock:
            result = {}
            for bucket in self.buckets:
                result[bucket] = sum(1 for v in self._values if v <= bucket)
            result[float('inf')] = len(self._values)
            return result

    def get_stats(self) -> Dict[str, float]:
        """Obtem estatisticas"""
        with self._lock:
            if not self._values:
                return {'count': 0, 'sum': 0, 'avg': 0, 'min': 0, 'max': 0}

            import numpy as np
            values = np.array(self._values)
            return {
                'count': len(values),
                'sum': float(values.sum()),
                'avg': float(values.mean()),
                'min': float(values.min()),
                'max': float(values.max()),
                'p50': float(np.percentile(values, 50)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99))
            }

    def reset(self) -> None:
        """Reseta histograma"""
        with self._lock:
            self._values.clear()


class MetricsRegistry:
    """Registro central de metricas"""

    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str = "") -> Counter:
        """Obtem ou cria contador"""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description)
            return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Obtem ou cria gauge"""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description)
            return self._gauges[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: List[float] = None
    ) -> Histogram:
        """Obtem ou cria histograma"""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, buckets)
            return self._histograms[name]

    def get_all_metrics(self) -> Dict[str, Any]:
        """Obtem todas as metricas"""
        with self._lock:
            result = {
                'counters': {name: c.get() for name, c in self._counters.items()},
                'gauges': {name: g.get() for name, g in self._gauges.items()},
                'histograms': {name: h.get_stats() for name, h in self._histograms.items()}
            }
            return result

    def reset_all(self) -> None:
        """Reseta todas as metricas"""
        with self._lock:
            for c in self._counters.values():
                c.reset()
            for h in self._histograms.values():
                h.reset()


# Registro global
_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Obtem registro global"""
    return _registry


class TradingMetrics:
    """Metricas especificas para trading"""

    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or get_registry()

        # Contadores
        self.trades_total = self.registry.counter(
            "trades_total",
            "Total de trades executados"
        )
        self.orders_total = self.registry.counter(
            "orders_total",
            "Total de ordens criadas"
        )
        self.orders_rejected = self.registry.counter(
            "orders_rejected",
            "Ordens rejeitadas"
        )
        self.risk_alerts = self.registry.counter(
            "risk_alerts_total",
            "Total de alertas de risco"
        )

        # Gauges
        self.portfolio_value = self.registry.gauge(
            "portfolio_value",
            "Valor atual do portfolio"
        )
        self.cash_balance = self.registry.gauge(
            "cash_balance",
            "Saldo em caixa"
        )
        self.active_positions = self.registry.gauge(
            "active_positions",
            "Numero de posicoes ativas"
        )
        self.leverage = self.registry.gauge(
            "leverage",
            "Alavancagem atual"
        )
        self.drawdown = self.registry.gauge(
            "drawdown",
            "Drawdown atual"
        )

        # Histogramas
        self.order_latency = self.registry.histogram(
            "order_latency_ms",
            "Latencia de execucao de ordens",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
        )
        self.trade_pnl = self.registry.histogram(
            "trade_pnl",
            "PnL por trade",
            buckets=[-1000, -500, -100, -50, 0, 50, 100, 500, 1000, 5000]
        )
        self.slippage = self.registry.histogram(
            "slippage_bps",
            "Slippage em basis points",
            buckets=[0, 1, 2, 5, 10, 20, 50, 100]
        )

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: float = 0,
        slippage_bps: float = 0,
        latency_ms: float = 0
    ) -> None:
        """Registra metricas de um trade"""
        self.trades_total.inc()
        self.trade_pnl.observe(pnl)
        self.slippage.observe(slippage_bps)
        if latency_ms > 0:
            self.order_latency.observe(latency_ms)

    def update_portfolio(
        self,
        value: float,
        cash: float,
        positions: int,
        leverage: float,
        drawdown: float
    ) -> None:
        """Atualiza metricas do portfolio"""
        self.portfolio_value.set(value)
        self.cash_balance.set(cash)
        self.active_positions.set(positions)
        self.leverage.set(leverage)
        self.drawdown.set(drawdown)

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo das metricas"""
        return {
            'trades': {
                'total': self.trades_total.get(),
                'pnl_stats': self.trade_pnl.get_stats()
            },
            'orders': {
                'total': self.orders_total.get(),
                'rejected': self.orders_rejected.get(),
                'latency_stats': self.order_latency.get_stats()
            },
            'portfolio': {
                'value': self.portfolio_value.get(),
                'cash': self.cash_balance.get(),
                'positions': self.active_positions.get(),
                'leverage': self.leverage.get(),
                'drawdown': self.drawdown.get()
            },
            'risk': {
                'alerts': self.risk_alerts.get()
            },
            'execution': {
                'slippage_stats': self.slippage.get_stats()
            }
        }


class HealthCheck:
    """Verificacao de saude do sistema"""

    def __init__(self):
        self.checks: Dict[str, Callable[[], bool]] = {}
        self.last_check: Dict[str, Dict] = {}

    def register(self, name: str, check_fn: Callable[[], bool]) -> None:
        """Registra verificacao"""
        self.checks[name] = check_fn

    def check(self, name: str) -> Dict:
        """Executa verificacao especifica"""
        if name not in self.checks:
            return {'status': 'unknown', 'error': 'Check not found'}

        try:
            start = time.time()
            result = self.checks[name]()
            duration = (time.time() - start) * 1000

            status = {
                'status': 'healthy' if result else 'unhealthy',
                'duration_ms': duration,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            status = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

        self.last_check[name] = status
        return status

    def check_all(self) -> Dict:
        """Executa todas as verificacoes"""
        results = {}
        all_healthy = True

        for name in self.checks:
            results[name] = self.check(name)
            if results[name]['status'] != 'healthy':
                all_healthy = False

        return {
            'overall': 'healthy' if all_healthy else 'unhealthy',
            'checks': results,
            'timestamp': datetime.now().isoformat()
        }
