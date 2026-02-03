"""
Sistema de Logging e Monitoramento
"""
import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from enum import Enum
import traceback


class LogLevel(Enum):
    """Niveis de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Entrada de log estruturada"""
    timestamp: str
    level: str
    module: str
    message: str
    data: Optional[Dict[str, Any]] = None
    exception: Optional[str] = None
    trace_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class StructuredLogger:
    """Logger estruturado para producao"""

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[str] = None,
        json_format: bool = False
    ):
        self.name = name
        self.json_format = json_format
        self._trace_id: Optional[str] = None

        # Configura logger Python padrao
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))

        # Remove handlers existentes
        self.logger.handlers.clear()

        # Formatter
        if json_format:
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        # File handler
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def set_trace_id(self, trace_id: str) -> None:
        """Define trace ID para correlacao"""
        self._trace_id = trace_id

    def _format_message(
        self,
        level: LogLevel,
        message: str,
        data: Optional[Dict] = None,
        exception: Optional[Exception] = None
    ) -> str:
        """Formata mensagem de log"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            module=self.name,
            message=message,
            data=data,
            exception=str(exception) if exception else None,
            trace_id=self._trace_id
        )

        if self.json_format:
            return entry.to_json()
        else:
            msg = message
            if data:
                msg += f" | data={data}"
            if exception:
                msg += f" | error={exception}"
            return msg

    def debug(self, message: str, data: Optional[Dict] = None) -> None:
        """Log nivel DEBUG"""
        self.logger.debug(self._format_message(LogLevel.DEBUG, message, data))

    def info(self, message: str, data: Optional[Dict] = None) -> None:
        """Log nivel INFO"""
        self.logger.info(self._format_message(LogLevel.INFO, message, data))

    def warning(self, message: str, data: Optional[Dict] = None) -> None:
        """Log nivel WARNING"""
        self.logger.warning(self._format_message(LogLevel.WARNING, message, data))

    def error(
        self,
        message: str,
        data: Optional[Dict] = None,
        exception: Optional[Exception] = None
    ) -> None:
        """Log nivel ERROR"""
        self.logger.error(
            self._format_message(LogLevel.ERROR, message, data, exception)
        )
        if exception:
            self.logger.error(traceback.format_exc())

    def critical(
        self,
        message: str,
        data: Optional[Dict] = None,
        exception: Optional[Exception] = None
    ) -> None:
        """Log nivel CRITICAL"""
        self.logger.critical(
            self._format_message(LogLevel.CRITICAL, message, data, exception)
        )
        if exception:
            self.logger.critical(traceback.format_exc())

    def trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_id: str
    ) -> None:
        """Log especifico para trades"""
        self.info("Trade executado", {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'order_id': order_id,
            'value': quantity * price
        })

    def risk_alert(
        self,
        alert_type: str,
        message: str,
        current_value: float,
        limit: float
    ) -> None:
        """Log especifico para alertas de risco"""
        self.warning(f"Risk Alert: {alert_type}", {
            'alert_type': alert_type,
            'message': message,
            'current_value': current_value,
            'limit': limit,
            'breach_pct': (current_value - limit) / limit * 100 if limit else 0
        })

    def performance(
        self,
        metric: str,
        value: float,
        benchmark: Optional[float] = None
    ) -> None:
        """Log especifico para metricas de performance"""
        data = {'metric': metric, 'value': value}
        if benchmark is not None:
            data['benchmark'] = benchmark
            data['vs_benchmark'] = value - benchmark
        self.info(f"Performance: {metric}={value:.4f}", data)


def setup_logging(
    name: str = "momentum_forge",
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False
) -> StructuredLogger:
    """
    Configura logging para a aplicacao.

    Args:
        name: Nome do logger
        level: Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Arquivo de log (opcional)
        json_format: Se True, usa formato JSON

    Returns:
        StructuredLogger configurado
    """
    log_level = LogLevel[level.upper()]

    return StructuredLogger(
        name=name,
        level=log_level,
        log_file=log_file,
        json_format=json_format
    )


# Logger global
_default_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """Obtem logger global"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger


def set_logger(logger: StructuredLogger) -> None:
    """Define logger global"""
    global _default_logger
    _default_logger = logger
