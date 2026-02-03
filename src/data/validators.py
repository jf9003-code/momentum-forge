"""
Validadores de dados
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import logging

from src.data.providers.base import DataQuality


logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severidade de problemas de validacao"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Problema encontrado na validacao"""
    severity: ValidationSeverity
    field: str
    message: str
    row_indices: Optional[List[int]] = None
    suggested_fix: Optional[str] = None


@dataclass
class ValidationResult:
    """Resultado da validacao"""
    is_valid: bool
    issues: List[ValidationIssue]
    statistics: Dict[str, float]
    quality_score: float  # 0-100

    def __str__(self) -> str:
        status = "VALIDO" if self.is_valid else "INVALIDO"
        return f"Validacao: {status} | Score: {self.quality_score:.1f}/100 | Issues: {len(self.issues)}"


class DataValidator:
    """Validador de dados de mercado"""

    def __init__(
        self,
        max_missing_pct: float = 0.05,
        max_gap_days: int = 5,
        min_data_points: int = 252,
        outlier_std_threshold: float = 5.0
    ):
        self.max_missing_pct = max_missing_pct
        self.max_gap_days = max_gap_days
        self.min_data_points = min_data_points
        self.outlier_std_threshold = outlier_std_threshold

    def validate(self, data: pd.DataFrame, symbol: str = "unknown") -> ValidationResult:
        """
        Valida DataFrame de dados OHLCV.

        Args:
            data: DataFrame com dados OHLCV
            symbol: Nome do simbolo (para mensagens)

        Returns:
            ValidationResult com detalhes
        """
        issues = []
        stats = {}

        if data.empty:
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    field="data",
                    message=f"DataFrame vazio para {symbol}"
                )],
                statistics={},
                quality_score=0.0
            )

        # Validacoes basicas
        issues.extend(self._validate_structure(data))
        issues.extend(self._validate_missing_values(data))
        issues.extend(self._validate_date_gaps(data))
        issues.extend(self._validate_price_consistency(data))
        issues.extend(self._validate_volume(data))
        issues.extend(self._validate_outliers(data))

        # Calcula estatisticas
        stats = self._calculate_statistics(data)

        # Calcula score de qualidade
        quality_score = self._calculate_quality_score(data, issues)

        # Determina se e valido
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        is_valid = len(critical_issues) == 0 and len(error_issues) <= 2

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            statistics=stats,
            quality_score=quality_score
        )

    def _validate_structure(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Valida estrutura basica do DataFrame"""
        issues = []
        required_cols = ['Close']
        optional_cols = ['Open', 'High', 'Low', 'Volume', 'Adj Close']

        for col in required_cols:
            if col not in data.columns:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    field=col,
                    message=f"Coluna obrigatoria '{col}' nao encontrada"
                ))

        if len(data) < self.min_data_points:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="rows",
                message=f"Poucos dados: {len(data)} (minimo recomendado: {self.min_data_points})"
            ))

        # Valida index
        if not isinstance(data.index, pd.DatetimeIndex):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field="index",
                message="Index deve ser DatetimeIndex",
                suggested_fix="df.index = pd.to_datetime(df.index)"
            ))

        return issues

    def _validate_missing_values(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Valida valores faltantes"""
        issues = []

        for col in data.columns:
            missing = data[col].isna().sum()
            missing_pct = missing / len(data)

            if missing_pct > self.max_missing_pct:
                severity = ValidationSeverity.ERROR if missing_pct > 0.10 else ValidationSeverity.WARNING
                issues.append(ValidationIssue(
                    severity=severity,
                    field=col,
                    message=f"{missing} valores faltantes ({missing_pct:.1%})",
                    row_indices=data[data[col].isna()].index.tolist()[:10]
                ))

        return issues

    def _validate_date_gaps(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Valida gaps de datas"""
        issues = []

        if not isinstance(data.index, pd.DatetimeIndex):
            return issues

        # Calcula diferenca entre datas consecutivas
        date_diffs = data.index.to_series().diff()

        # Considera apenas gaps maiores que fins de semana (3 dias)
        large_gaps = date_diffs[date_diffs > timedelta(days=self.max_gap_days)]

        if len(large_gaps) > 0:
            for gap_date, gap_size in large_gaps.items():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="dates",
                    message=f"Gap de {gap_size.days} dias antes de {gap_date.strftime('%Y-%m-%d')}"
                ))

        return issues

    def _validate_price_consistency(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Valida consistencia de precos OHLC"""
        issues = []

        # Precos devem ser positivos
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                negative = (data[col] <= 0).sum()
                if negative > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field=col,
                        message=f"{negative} valores negativos ou zero",
                        row_indices=data[data[col] <= 0].index.tolist()[:10]
                    ))

        # High >= Low
        if 'High' in data.columns and 'Low' in data.columns:
            invalid = (data['High'] < data['Low']).sum()
            if invalid > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field="High/Low",
                    message=f"{invalid} linhas onde High < Low",
                    row_indices=data[data['High'] < data['Low']].index.tolist()[:10]
                ))

        # Close deve estar entre High e Low
        if all(col in data.columns for col in ['High', 'Low', 'Close']):
            invalid = ((data['Close'] > data['High']) | (data['Close'] < data['Low'])).sum()
            if invalid > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="Close",
                    message=f"{invalid} linhas onde Close fora do range High/Low"
                ))

        return issues

    def _validate_volume(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Valida dados de volume"""
        issues = []

        if 'Volume' not in data.columns:
            return issues

        # Volume negativo
        negative = (data['Volume'] < 0).sum()
        if negative > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field="Volume",
                message=f"{negative} valores negativos"
            ))

        # Volume zero (pode ser valido em alguns casos)
        zero = (data['Volume'] == 0).sum()
        if zero > len(data) * 0.10:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="Volume",
                message=f"{zero} linhas com volume zero ({zero/len(data):.1%})"
            ))

        return issues

    def _validate_outliers(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Detecta outliers nos retornos"""
        issues = []

        if 'Close' not in data.columns:
            return issues

        returns = data['Close'].pct_change().dropna()

        if len(returns) < 30:
            return issues

        mean = returns.mean()
        std = returns.std()

        if std == 0:
            return issues

        z_scores = (returns - mean) / std
        outliers = z_scores.abs() > self.outlier_std_threshold

        if outliers.sum() > 0:
            severity = ValidationSeverity.ERROR if outliers.sum() > 5 else ValidationSeverity.WARNING
            outlier_dates = returns[outliers].index.tolist()[:10]
            issues.append(ValidationIssue(
                severity=severity,
                field="returns",
                message=f"{outliers.sum()} outliers detectados (Z-score > {self.outlier_std_threshold})",
                row_indices=outlier_dates,
                suggested_fix="Verificar splits/dividendos ou usar Adj Close"
            ))

        return issues

    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calcula estatisticas descritivas"""
        stats = {
            'total_rows': len(data),
            'start_date': str(data.index.min()),
            'end_date': str(data.index.max())
        }

        if 'Close' in data.columns:
            returns = data['Close'].pct_change().dropna()
            stats['mean_return'] = returns.mean()
            stats['std_return'] = returns.std()
            stats['min_return'] = returns.min()
            stats['max_return'] = returns.max()
            stats['skewness'] = returns.skew()
            stats['kurtosis'] = returns.kurtosis()

        if 'Volume' in data.columns:
            stats['avg_volume'] = data['Volume'].mean()
            stats['total_volume'] = data['Volume'].sum()

        return stats

    def _calculate_quality_score(
        self,
        data: pd.DataFrame,
        issues: List[ValidationIssue]
    ) -> float:
        """Calcula score de qualidade 0-100"""
        score = 100.0

        # Penalidades por tipo de issue
        penalties = {
            ValidationSeverity.INFO: 0,
            ValidationSeverity.WARNING: 5,
            ValidationSeverity.ERROR: 15,
            ValidationSeverity.CRITICAL: 40
        }

        for issue in issues:
            score -= penalties[issue.severity]

        # Penalidade por dados faltantes
        missing_pct = data.isna().sum().sum() / (len(data) * len(data.columns))
        score -= missing_pct * 50

        # Penalidade por poucos dados
        if len(data) < self.min_data_points:
            score -= (1 - len(data) / self.min_data_points) * 20

        return max(0.0, min(100.0, score))


class DataCleaner:
    """Limpa e corrige dados de mercado"""

    def __init__(
        self,
        fill_method: str = 'ffill',
        max_fill_periods: int = 5,
        remove_outliers: bool = True,
        outlier_std: float = 5.0
    ):
        self.fill_method = fill_method
        self.max_fill_periods = max_fill_periods
        self.remove_outliers = remove_outliers
        self.outlier_std = outlier_std

    def clean(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Limpa dados.

        Returns:
            Tuple de (DataFrame limpo, Dict com contagem de correcoes)
        """
        df = data.copy()
        changes = {
            'filled_missing': 0,
            'removed_duplicates': 0,
            'fixed_negative_volume': 0,
            'capped_outliers': 0,
            'removed_rows': 0
        }

        # Remove duplicatas
        original_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        changes['removed_duplicates'] = original_len - len(df)

        # Ordena por data
        df = df.sort_index()

        # Preenche valores faltantes
        if self.fill_method == 'ffill':
            filled = df.isna().sum().sum()
            df = df.ffill(limit=self.max_fill_periods)
            changes['filled_missing'] = filled - df.isna().sum().sum()
        elif self.fill_method == 'interpolate':
            filled = df.isna().sum().sum()
            df = df.interpolate(method='time', limit=self.max_fill_periods)
            changes['filled_missing'] = filled - df.isna().sum().sum()

        # Corrige volume negativo
        if 'Volume' in df.columns:
            negative = (df['Volume'] < 0).sum()
            df['Volume'] = df['Volume'].clip(lower=0)
            changes['fixed_negative_volume'] = negative

        # Limita outliers
        if self.remove_outliers and 'Close' in df.columns:
            returns = df['Close'].pct_change()
            mean = returns.mean()
            std = returns.std()

            if std > 0:
                upper = mean + self.outlier_std * std
                lower = mean - self.outlier_std * std
                outliers = (returns > upper) | (returns < lower)
                changes['capped_outliers'] = outliers.sum()

                # Em vez de remover, cap os valores
                if changes['capped_outliers'] > 0:
                    logger.warning(f"Encontrados {changes['capped_outliers']} outliers")

        # Remove linhas ainda com NaN em colunas criticas
        original_len = len(df)
        df = df.dropna(subset=['Close'])
        changes['removed_rows'] = original_len - len(df)

        return df, changes


def validate_backtest_data(
    data: pd.DataFrame,
    min_history: int = 252,
    required_symbols: Optional[List[str]] = None
) -> ValidationResult:
    """
    Valida dados para uso em backtest.

    Args:
        data: DataFrame com dados (pode ser MultiIndex)
        min_history: Minimo de dias de historico
        required_symbols: Lista de simbolos obrigatorios

    Returns:
        ValidationResult
    """
    validator = DataValidator(min_data_points=min_history)
    issues = []
    stats = {}

    if data.empty:
        return ValidationResult(
            is_valid=False,
            issues=[ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                field="data",
                message="DataFrame vazio"
            )],
            statistics={},
            quality_score=0.0
        )

    # Extrai simbolos
    if isinstance(data.columns, pd.MultiIndex):
        symbols = data.columns.get_level_values(0).unique().tolist()
    else:
        symbols = list(set([c.split('_')[0] for c in data.columns if '_' in c]))

    stats['total_symbols'] = len(symbols)
    stats['total_rows'] = len(data)

    # Verifica simbolos obrigatorios
    if required_symbols:
        missing = set(required_symbols) - set(symbols)
        if missing:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field="symbols",
                message=f"Simbolos faltantes: {missing}"
            ))

    # Valida cada simbolo
    valid_symbols = 0
    for symbol in symbols:
        if isinstance(data.columns, pd.MultiIndex):
            symbol_data = data[symbol]
        else:
            cols = [c for c in data.columns if c.startswith(f"{symbol}_")]
            if not cols:
                continue
            symbol_data = data[cols].rename(columns=lambda x: x.replace(f"{symbol}_", ""))

        result = validator.validate(symbol_data, symbol)
        if result.is_valid:
            valid_symbols += 1
        else:
            for issue in result.issues:
                if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                    issue.field = f"{symbol}.{issue.field}"
                    issues.append(issue)

    stats['valid_symbols'] = valid_symbols
    stats['invalid_symbols'] = len(symbols) - valid_symbols

    # Calcula score geral
    if len(symbols) > 0:
        quality_score = (valid_symbols / len(symbols)) * 100
    else:
        quality_score = 0

    is_valid = (
        len([i for i in issues if i.severity == ValidationSeverity.CRITICAL]) == 0 and
        valid_symbols >= len(symbols) * 0.8
    )

    return ValidationResult(
        is_valid=is_valid,
        issues=issues,
        statistics=stats,
        quality_score=quality_score
    )
