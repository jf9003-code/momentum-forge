"""
Clone Filter and Strategy Validators - CORRECTED VERSION
Fixes the over-rejection issue by adjusting weight distribution
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from backend.app.core.config import get_config
from backend.app.generator.models import StrategySpec
from backend.app.generator.parameters import parameter_distance

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of strategy validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    score: float = 0.0


class CloneFilter:
    """
    CORRECTED Clone Filter Implementation

    Original Issue: Same-template strategies could only achieve ~0.38 max distance
    because template/type weights (42% combined) were always 0 within same template.

    Fix: Reduced min_distance threshold and rebalanced weights to prioritize
    parameter differences for same-template comparisons.
    """

    def __init__(self, min_distance: float = None):
        config = get_config()
        self.min_distance = min_distance or config.generator.min_structural_distance
        self.config = config.generator

        # CORRECTED weight distribution
        # Original had too much weight on template/type (42% combined)
        # New distribution focuses more on parameters and indicators
        self.weights = {
            'strategy_type': 0.10,      # Reduced from 0.17
            'template': 0.08,           # Reduced from 0.25
            'indicators': 0.20,         # Same
            'entry_conditions': 0.12,   # Increased from 0.10
            'exit_conditions': 0.10,    # Increased from 0.08
            'parameters': 0.25,         # Increased from 0.12
            'risk_config': 0.15,        # Increased from 0.08
        }

        self.existing_strategies: List[Dict] = []

    def add_existing(self, strategies: List[Dict]):
        """Add existing strategies for comparison"""
        self.existing_strategies.extend(strategies)

    def clear(self):
        """Clear existing strategies"""
        self.existing_strategies = []

    def calculate_structural_distance(
        self,
        strategy1: Dict,
        strategy2: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """
        CORRECTED structural distance calculation

        Returns total distance and component breakdown for debugging
        """
        components = {}

        # Strategy type distance
        type1 = strategy1.get('strategy_type', '')
        type2 = strategy2.get('strategy_type', '')
        components['strategy_type'] = 0.0 if type1 == type2 else 1.0

        # Template distance
        template1 = strategy1.get('template_id', strategy1.get('template', ''))
        template2 = strategy2.get('template_id', strategy2.get('template', ''))
        components['template'] = 0.0 if template1 == template2 else 1.0

        # Indicator distance (Jaccard-based)
        indicators1 = self._extract_indicators(strategy1)
        indicators2 = self._extract_indicators(strategy2)
        components['indicators'] = self._jaccard_distance(indicators1, indicators2)

        # Entry conditions distance
        entry1 = self._extract_conditions(strategy1, 'entry')
        entry2 = self._extract_conditions(strategy2, 'entry')
        components['entry_conditions'] = self._condition_distance(entry1, entry2)

        # Exit conditions distance
        exit1 = self._extract_conditions(strategy1, 'exit')
        exit2 = self._extract_conditions(strategy2, 'exit')
        components['exit_conditions'] = self._condition_distance(exit1, exit2)

        # Parameter distance - ENHANCED
        params1 = self._extract_all_parameters(strategy1)
        params2 = self._extract_all_parameters(strategy2)
        components['parameters'] = parameter_distance(params1, params2)

        # Risk config distance
        risk1 = strategy1.get('risk_config', {})
        risk2 = strategy2.get('risk_config', {})
        components['risk_config'] = self._risk_distance(risk1, risk2)

        # Calculate weighted total
        total_distance = sum(
            components[k] * self.weights[k]
            for k in self.weights.keys()
        )

        # ENHANCEMENT: Boost distance for same-template strategies with different params
        if components['template'] == 0 and components['parameters'] > 0.3:
            # Same template but significantly different parameters
            # Apply a boost to prevent over-rejection
            boost = components['parameters'] * 0.15
            total_distance = min(1.0, total_distance + boost)

        return total_distance, components

    def _extract_indicators(self, strategy: Dict) -> set:
        """Extract indicator names from strategy"""
        indicators = set()

        # From indicators list
        for ind in strategy.get('indicators', []):
            if isinstance(ind, dict):
                indicators.add(ind.get('name', ''))
            else:
                indicators.add(str(ind))

        # From entry logic
        entry = strategy.get('entry_logic', {})
        if isinstance(entry, dict):
            for cond in entry.get('conditions', []):
                if isinstance(cond, dict):
                    indicators.add(cond.get('indicator', ''))

        # From exit logic
        exit_logic = strategy.get('exit_logic', {})
        if isinstance(exit_logic, dict):
            for cond in exit_logic.get('conditions', []):
                if isinstance(cond, dict):
                    indicators.add(cond.get('indicator', ''))

        indicators.discard('')
        return indicators

    def _extract_conditions(self, strategy: Dict, logic_type: str) -> List[str]:
        """Extract condition signatures"""
        conditions = []

        key = f'{logic_type}_logic'
        logic = strategy.get(key, {})

        if isinstance(logic, dict):
            for cond in logic.get('conditions', []):
                if isinstance(cond, dict):
                    sig = f"{cond.get('indicator', '')}_{cond.get('operator', '')}_{cond.get('compare_to', '')}"
                    conditions.append(sig)

        return conditions

    def _extract_all_parameters(self, strategy: Dict) -> Dict:
        """Extract all parameters from strategy for distance calculation"""
        params = {}

        # Direct parameters
        for key in ['parameters', 'indicator_params']:
            if key in strategy and isinstance(strategy[key], dict):
                params.update(strategy[key])

        # From indicators
        for ind in strategy.get('indicators', []):
            if isinstance(ind, dict):
                ind_params = ind.get('parameters', {})
                if isinstance(ind_params, dict):
                    prefix = ind.get('name', 'ind')
                    for k, v in ind_params.items():
                        params[f"{prefix}_{k}"] = v

        # From risk config
        risk = strategy.get('risk_config', {})
        if isinstance(risk, dict):
            for k in ['stop_loss_atr', 'take_profit_atr', 'atr_sl_mult', 'atr_tp_mult',
                      'risk_per_trade', 'max_position_size']:
                if k in risk:
                    params[f"risk_{k}"] = risk[k]

        return params

    def _jaccard_distance(self, set1: set, set2: set) -> float:
        """Calculate Jaccard distance between two sets"""
        if not set1 and not set2:
            return 0.0
        union = set1 | set2
        if not union:
            return 0.0
        intersection = set1 & set2
        return 1.0 - (len(intersection) / len(union))

    def _condition_distance(self, conds1: List[str], conds2: List[str]) -> float:
        """Calculate distance between condition lists"""
        if not conds1 and not conds2:
            return 0.0
        set1 = set(conds1)
        set2 = set(conds2)
        return self._jaccard_distance(set1, set2)

    def _risk_distance(self, risk1: Dict, risk2: Dict) -> float:
        """Calculate distance between risk configurations"""
        if not risk1 and not risk2:
            return 0.0

        distances = []

        # Position sizing method
        psm1 = risk1.get('position_sizing_method', '')
        psm2 = risk2.get('position_sizing_method', '')
        distances.append(0.0 if psm1 == psm2 else 1.0)

        # Numeric parameters
        numeric_keys = ['stop_loss_atr', 'take_profit_atr', 'risk_per_trade',
                       'max_position_size', 'atr_sl_mult', 'atr_tp_mult']

        for key in numeric_keys:
            v1 = risk1.get(key)
            v2 = risk2.get(key)
            if v1 is not None and v2 is not None:
                try:
                    v1, v2 = float(v1), float(v2)
                    if max(abs(v1), abs(v2)) > 0:
                        diff = abs(v1 - v2) / max(abs(v1), abs(v2))
                        distances.append(min(1.0, diff))
                except (ValueError, TypeError):
                    pass

        return sum(distances) / len(distances) if distances else 0.0

    def is_clone(self, new_strategy: Dict, debug: bool = False) -> Tuple[bool, Optional[str], float]:
        """
        Check if strategy is a clone of existing ones

        Returns: (is_clone, closest_match_id, min_distance)
        """
        min_distance = float('inf')
        closest_match = None

        for existing in self.existing_strategies:
            distance, components = self.calculate_structural_distance(new_strategy, existing)

            if debug:
                logger.debug(f"Distance to {existing.get('id', 'unknown')}: {distance:.4f}")
                logger.debug(f"  Components: {components}")

            if distance < min_distance:
                min_distance = distance
                closest_match = existing.get('id', existing.get('strategy_id'))

        is_clone = min_distance < self.min_distance

        if debug and is_clone:
            logger.info(f"Strategy rejected as clone. Distance: {min_distance:.4f} < {self.min_distance}")

        return is_clone, closest_match, min_distance

    def filter_batch(
        self,
        strategies: List[Dict],
        debug: bool = False
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter a batch of strategies, removing clones

        Returns: (accepted, rejected)
        """
        accepted = []
        rejected = []

        # Create working copy of existing
        working_set = list(self.existing_strategies)

        for strategy in strategies:
            # Temporarily add accepted strategies to working set
            temp_filter = CloneFilter(self.min_distance)
            temp_filter.existing_strategies = working_set + accepted

            is_clone, match_id, distance = temp_filter.is_clone(strategy, debug)

            if is_clone:
                strategy['_rejection_reason'] = f'Clone of {match_id} (distance: {distance:.4f})'
                rejected.append(strategy)
            else:
                strategy['_clone_distance'] = distance
                accepted.append(strategy)

        return accepted, rejected


class StrategyValidator:
    """Validates strategy specifications for completeness and correctness"""

    REQUIRED_FIELDS = [
        'strategy_type',
        'entry_logic',
        'exit_logic',
    ]

    VALID_STRATEGY_TYPES = [
        'mean_reversion', 'momentum', 'breakout',
        'volatility_expansion', 'time_based'
    ]

    VALID_TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1']

    def validate(self, strategy: Dict) -> ValidationResult:
        """Validate a strategy specification"""
        errors = []
        warnings = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in strategy or strategy[field] is None:
                errors.append(f"Missing required field: {field}")

        # Validate strategy type
        stype = strategy.get('strategy_type')
        if stype and stype not in self.VALID_STRATEGY_TYPES:
            errors.append(f"Invalid strategy type: {stype}")

        # Validate timeframe if present
        tf = strategy.get('timeframe')
        if tf and tf not in self.VALID_TIMEFRAMES:
            warnings.append(f"Non-standard timeframe: {tf}")

        # Validate entry logic
        entry = strategy.get('entry_logic', {})
        if isinstance(entry, dict):
            entry_errors = self._validate_logic(entry, 'entry')
            errors.extend(entry_errors)
        else:
            errors.append("entry_logic must be a dictionary")

        # Validate exit logic
        exit_logic = strategy.get('exit_logic', {})
        if isinstance(exit_logic, dict):
            exit_errors = self._validate_logic(exit_logic, 'exit')
            errors.extend(exit_errors)
        else:
            errors.append("exit_logic must be a dictionary")

        # Validate risk config if present
        risk = strategy.get('risk_config')
        if risk:
            risk_errors, risk_warnings = self._validate_risk(risk)
            errors.extend(risk_errors)
            warnings.extend(risk_warnings)

        # Validate indicators if present
        indicators = strategy.get('indicators', [])
        for i, ind in enumerate(indicators):
            ind_errors = self._validate_indicator(ind, i)
            errors.extend(ind_errors)

        # Calculate validation score
        score = 1.0 - (len(errors) * 0.2 + len(warnings) * 0.05)
        score = max(0.0, score)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            score=score
        )

    def _validate_logic(self, logic: Dict, logic_type: str) -> List[str]:
        """Validate entry/exit logic"""
        errors = []

        conditions = logic.get('conditions', [])
        if not conditions:
            errors.append(f"{logic_type}_logic has no conditions")

        for i, cond in enumerate(conditions):
            if not isinstance(cond, dict):
                errors.append(f"{logic_type} condition {i} must be a dictionary")
                continue

            if 'indicator' not in cond and 'type' not in cond:
                errors.append(f"{logic_type} condition {i} missing indicator or type")

        return errors

    def _validate_risk(self, risk: Dict) -> Tuple[List[str], List[str]]:
        """Validate risk configuration"""
        errors = []
        warnings = []

        # Check position sizing
        psm = risk.get('position_sizing_method')
        valid_psm = ['fixed_percentage', 'fixed_fractional', 'kelly',
                     'volatility_based', 'fixed_units']
        if psm and psm not in valid_psm:
            errors.append(f"Invalid position sizing method: {psm}")

        # Check risk per trade
        rpt = risk.get('risk_per_trade')
        if rpt is not None:
            try:
                rpt = float(rpt)
                if rpt > 0.05:
                    warnings.append(f"High risk per trade: {rpt*100:.1f}%")
                elif rpt <= 0:
                    errors.append("risk_per_trade must be positive")
            except (ValueError, TypeError):
                errors.append("risk_per_trade must be numeric")

        # Check stop loss / take profit
        sl = risk.get('stop_loss_atr') or risk.get('atr_sl_mult')
        tp = risk.get('take_profit_atr') or risk.get('atr_tp_mult')

        if sl is not None and tp is not None:
            try:
                sl, tp = float(sl), float(tp)
                if tp / sl < 1.0:
                    warnings.append(f"Risk/reward ratio below 1:1 ({tp/sl:.2f})")
            except (ValueError, TypeError, ZeroDivisionError):
                pass

        return errors, warnings

    def _validate_indicator(self, indicator: Dict, index: int) -> List[str]:
        """Validate indicator specification"""
        errors = []

        if not isinstance(indicator, dict):
            errors.append(f"Indicator {index} must be a dictionary")
            return errors

        if 'name' not in indicator:
            errors.append(f"Indicator {index} missing name")

        params = indicator.get('parameters', {})
        if params and not isinstance(params, dict):
            errors.append(f"Indicator {index} parameters must be a dictionary")

        return errors


def validate_strategy_batch(strategies: List[Dict]) -> Dict:
    """
    Validate a batch of strategies

    Returns summary with valid/invalid counts and details
    """
    validator = StrategyValidator()

    results = {
        'total': len(strategies),
        'valid': 0,
        'invalid': 0,
        'warnings': 0,
        'details': []
    }

    for i, strategy in enumerate(strategies):
        result = validator.validate(strategy)

        detail = {
            'index': i,
            'strategy_id': strategy.get('id', f'strategy_{i}'),
            'is_valid': result.is_valid,
            'errors': result.errors,
            'warnings': result.warnings,
            'score': result.score
        }

        results['details'].append(detail)

        if result.is_valid:
            results['valid'] += 1
        else:
            results['invalid'] += 1

        if result.warnings:
            results['warnings'] += 1

    return results
