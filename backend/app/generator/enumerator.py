"""
Controlled Enumerator - Slot Filler Pattern
Top-down composition for institutional strategy generation
"""
from typing import Dict, List, Optional, Any, Iterator, Tuple
from dataclasses import dataclass, field
import random
import itertools
import logging
from datetime import datetime

from backend.app.generator.templates import (
    StrategyTemplate, IndicatorSlot, ConditionSlot,
    get_template, get_templates_for_type, get_all_templates
)
from backend.app.generator.permissions import get_profile_for_type
from backend.app.core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class SlotChoice:
    """Represents a choice for filling a slot"""
    slot_name: str
    chosen_value: Any
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilledTemplate:
    """A template with all slots filled"""
    template_id: str
    strategy_type: str
    indicators: List[Dict]
    entry_logic: Dict
    exit_logic: Dict
    parameters: Dict
    risk_config: Dict
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'template_id': self.template_id,
            'strategy_type': self.strategy_type,
            'indicators': self.indicators,
            'entry_logic': self.entry_logic,
            'exit_logic': self.exit_logic,
            'parameters': self.parameters,
            'risk_config': self.risk_config,
            'metadata': self.metadata
        }


class SlotFiller:
    """
    Fills template slots with valid choices
    Uses controlled enumeration to generate diverse strategies
    """

    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.config = get_config().generator

    def fill_indicator_slot(
        self,
        slot: IndicatorSlot,
        profile: Optional[Dict] = None,
        randomize: bool = True
    ) -> Dict:
        """Fill an indicator slot with a valid choice"""
        allowed = slot.allowed_indicators.copy()

        # Filter by profile if provided
        if profile:
            forbidden = profile.get('forbidden_indicators', [])
            allowed = [ind for ind in allowed if ind not in forbidden]

            profile_allowed = profile.get('allowed_indicators', [])
            if profile_allowed:
                allowed = [ind for ind in allowed if ind in profile_allowed]

        if not allowed:
            allowed = [slot.default_indicator] if slot.default_indicator else slot.allowed_indicators[:1]

        # Choose indicator
        if randomize:
            chosen = self.rng.choice(allowed)
        else:
            chosen = allowed[0]

        # Generate parameters
        params = self._generate_parameters(slot.parameter_ranges, randomize)

        return {
            'name': chosen,
            'slot': slot.name,
            'category': slot.category,
            'parameters': params
        }

    def fill_condition_slot(
        self,
        slot: ConditionSlot,
        indicator: Dict,
        randomize: bool = True
    ) -> Dict:
        """Fill a condition slot with a valid choice"""
        operators = slot.operators
        compare_options = slot.compare_to_options

        if randomize:
            operator = self.rng.choice(operators) if operators else ">"
            compare_to = self.rng.choice(compare_options) if compare_options else "value"
        else:
            operator = operators[0] if operators else ">"
            compare_to = compare_options[0] if compare_options else "value"

        return {
            'indicator': indicator['name'],
            'indicator_slot': slot.indicator_slot,
            'operator': operator,
            'compare_to': compare_to,
            'condition_name': slot.name
        }

    def _generate_parameters(
        self,
        ranges: Dict[str, tuple],
        randomize: bool = True
    ) -> Dict:
        """Generate parameter values within ranges"""
        params = {}

        for param_name, range_val in ranges.items():
            if isinstance(range_val, (list, tuple)):
                if len(range_val) == 2:
                    min_val, max_val = range_val
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        if randomize:
                            params[param_name] = self.rng.randint(min_val, max_val)
                        else:
                            params[param_name] = (min_val + max_val) // 2
                    elif isinstance(min_val, float) or isinstance(max_val, float):
                        if randomize:
                            params[param_name] = round(
                                self.rng.uniform(float(min_val), float(max_val)),
                                2
                            )
                        else:
                            params[param_name] = round((float(min_val) + float(max_val)) / 2, 2)
                    elif isinstance(min_val, str):
                        # List of string choices
                        if randomize:
                            params[param_name] = self.rng.choice(range_val)
                        else:
                            params[param_name] = range_val[0]
                else:
                    # Multiple choices
                    if randomize:
                        params[param_name] = self.rng.choice(range_val)
                    else:
                        params[param_name] = range_val[0]
            else:
                params[param_name] = range_val

        return params


class ControlledEnumerator:
    """
    Generates strategies through controlled enumeration of template slots

    Key principles:
    1. Top-down composition (not brute force)
    2. Controlled variation within templates
    3. Profile-based filtering
    4. Clone prevention through diversity tracking
    """

    def __init__(self, seed: int = None):
        self.seed = seed or int(datetime.now().timestamp())
        self.rng = random.Random(self.seed)
        self.filler = SlotFiller(self.seed)
        self.config = get_config().generator

        # Track generated combinations for diversity
        self.generated_signatures: set = set()

    def enumerate_for_type(
        self,
        strategy_type: str,
        count: int = 10,
        timeframe: str = None,
        market_type: str = None
    ) -> List[FilledTemplate]:
        """Generate strategies for a specific type"""
        templates = get_templates_for_type(strategy_type)
        if not templates:
            logger.warning(f"No templates found for type: {strategy_type}")
            return []

        profile = get_profile_for_type(strategy_type)
        results = []
        attempts = 0
        max_attempts = count * 5  # Allow for failed attempts

        while len(results) < count and attempts < max_attempts:
            attempts += 1

            # Select template
            template = self.rng.choice(templates)

            # Filter by timeframe/market if specified
            if timeframe and template.allowed_timeframes:
                if timeframe not in template.allowed_timeframes:
                    continue

            if market_type and template.allowed_markets:
                if market_type not in template.allowed_markets:
                    continue

            # Fill template
            filled = self._fill_template(template, profile)
            if filled:
                # Check for uniqueness
                signature = self._generate_signature(filled)
                if signature not in self.generated_signatures:
                    self.generated_signatures.add(signature)
                    results.append(filled)

        return results

    def enumerate_mixed(
        self,
        type_distribution: Dict[str, int],
        timeframe: str = None,
        market_type: str = None
    ) -> List[FilledTemplate]:
        """
        Generate strategies with specified distribution across types

        Args:
            type_distribution: Dict mapping strategy_type to count
                e.g., {"momentum": 5, "mean_reversion": 3, "breakout": 2}
        """
        results = []

        for strategy_type, count in type_distribution.items():
            type_results = self.enumerate_for_type(
                strategy_type=strategy_type,
                count=count,
                timeframe=timeframe,
                market_type=market_type
            )
            results.extend(type_results)

        # Shuffle to mix types
        self.rng.shuffle(results)
        return results

    def enumerate_all_types(
        self,
        count_per_type: int = 2,
        timeframe: str = None,
        market_type: str = None
    ) -> List[FilledTemplate]:
        """Generate strategies across all types equally"""
        all_types = ["mean_reversion", "momentum", "breakout",
                     "volatility_expansion", "time_based"]

        distribution = {t: count_per_type for t in all_types}
        return self.enumerate_mixed(distribution, timeframe, market_type)

    def _fill_template(
        self,
        template: StrategyTemplate,
        profile: Optional[Dict] = None
    ) -> Optional[FilledTemplate]:
        """Fill all slots in a template"""
        try:
            # Fill indicator slots
            indicators = {}
            indicator_list = []

            for slot in template.indicator_slots:
                if not slot.required and self.rng.random() < 0.3:
                    continue  # Skip optional slots 30% of time

                filled_ind = self.filler.fill_indicator_slot(
                    slot, profile, randomize=True
                )
                indicators[slot.name] = filled_ind
                indicator_list.append(filled_ind)

            # Fill entry conditions
            entry_conditions = []
            for slot in template.entry_slots:
                if slot.indicator_slot in indicators:
                    ind = indicators[slot.indicator_slot]
                    condition = self.filler.fill_condition_slot(
                        slot, ind, randomize=True
                    )
                    entry_conditions.append(condition)

            if not entry_conditions:
                return None  # Need at least one entry condition

            # Fill exit conditions
            exit_conditions = []
            for slot in template.exit_slots:
                if slot.indicator_slot in indicators:
                    ind = indicators[slot.indicator_slot]
                    condition = self.filler.fill_condition_slot(
                        slot, ind, randomize=True
                    )
                    exit_conditions.append(condition)

            if not exit_conditions:
                # Add default exit if none filled
                exit_conditions.append({
                    'type': 'time_exit',
                    'bars': self.rng.randint(10, 30)
                })

            # Generate risk config
            risk_config = self._generate_risk_config(template, profile)

            # Collect all parameters
            all_params = {}
            for ind in indicator_list:
                prefix = ind['slot']
                for k, v in ind.get('parameters', {}).items():
                    all_params[f"{prefix}_{k}"] = v

            return FilledTemplate(
                template_id=template.template_id,
                strategy_type=template.strategy_type.value,
                indicators=indicator_list,
                entry_logic={
                    'logic_type': template.entry_logic_type,
                    'conditions': entry_conditions
                },
                exit_logic={
                    'logic_type': template.exit_logic_type,
                    'conditions': exit_conditions
                },
                parameters=all_params,
                risk_config=risk_config,
                metadata={
                    'template_name': template.name,
                    'generated_at': datetime.now().isoformat(),
                    'seed': self.seed
                }
            )

        except Exception as e:
            logger.error(f"Error filling template {template.template_id}: {e}")
            return None

    def _generate_risk_config(
        self,
        template: StrategyTemplate,
        profile: Optional[Dict] = None
    ) -> Dict:
        """Generate risk configuration based on template and profile"""
        constraints = template.risk_constraints

        # Position sizing method
        allowed_psm = constraints.get('position_sizing', ['fixed_fractional'])
        psm = self.rng.choice(allowed_psm)

        # ATR multipliers for stops
        max_stop_atr = constraints.get('max_stop_atr', 2.0)
        stop_atr = round(self.rng.uniform(1.0, max_stop_atr), 2)

        # Risk reward
        min_rr = constraints.get('min_risk_reward', 1.5)
        take_profit_atr = round(stop_atr * self.rng.uniform(min_rr, min_rr + 1.0), 2)

        # Risk per trade
        if profile:
            max_risk = profile.get('risk_constraints', {}).get('max_risk_per_trade', 0.02)
        else:
            max_risk = 0.02

        risk_per_trade = round(self.rng.uniform(0.005, max_risk), 4)

        config = {
            'position_sizing_method': psm,
            'atr_sl_mult': stop_atr,
            'atr_tp_mult': take_profit_atr,
            'risk_per_trade': risk_per_trade,
            'max_position_size': round(self.rng.uniform(0.05, 0.15), 2),
            'risk_reward_ratio': round(take_profit_atr / stop_atr, 2)
        }

        # Add time exit if template supports it
        time_exit = constraints.get('time_exit_bars')
        if time_exit:
            config['time_exit_bars'] = self.rng.randint(10, time_exit)

        return config

    def _generate_signature(self, filled: FilledTemplate) -> str:
        """Generate a unique signature for clone detection"""
        components = [
            filled.template_id,
            filled.strategy_type,
            '-'.join(sorted(ind['name'] for ind in filled.indicators)),
            filled.entry_logic.get('logic_type', ''),
            '-'.join(sorted(
                c.get('operator', '') + c.get('compare_to', '')
                for c in filled.entry_logic.get('conditions', [])
            )),
            str(filled.risk_config.get('atr_sl_mult', '')),
            str(filled.risk_config.get('atr_tp_mult', '')),
        ]
        return '|'.join(components)

    def reset(self, new_seed: int = None):
        """Reset enumerator state"""
        self.seed = new_seed or int(datetime.now().timestamp())
        self.rng = random.Random(self.seed)
        self.filler = SlotFiller(self.seed)
        self.generated_signatures.clear()


class BatchEnumerator:
    """
    Handles batch generation with progress tracking and cancellation
    """

    def __init__(self):
        self.enumerator = ControlledEnumerator()
        self.is_running = False
        self.should_stop = False
        self.progress = 0
        self.total = 0

    def generate_batch(
        self,
        total_count: int,
        type_weights: Dict[str, float] = None,
        timeframe: str = None,
        market_type: str = None,
        callback: callable = None
    ) -> Iterator[FilledTemplate]:
        """
        Generate a batch of strategies with progress tracking

        Args:
            total_count: Total number of strategies to generate
            type_weights: Weights for each strategy type (normalized internally)
            timeframe: Optional timeframe filter
            market_type: Optional market filter
            callback: Optional callback(progress, total) for progress updates

        Yields:
            FilledTemplate instances
        """
        self.is_running = True
        self.should_stop = False
        self.progress = 0
        self.total = total_count

        # Default weights
        if type_weights is None:
            type_weights = {
                'mean_reversion': 0.25,
                'momentum': 0.25,
                'breakout': 0.20,
                'volatility_expansion': 0.15,
                'time_based': 0.15
            }

        # Calculate counts from weights
        total_weight = sum(type_weights.values())
        distribution = {}
        remaining = total_count

        for i, (stype, weight) in enumerate(type_weights.items()):
            if i == len(type_weights) - 1:
                # Last type gets remainder
                distribution[stype] = remaining
            else:
                count = int(total_count * (weight / total_weight))
                distribution[stype] = count
                remaining -= count

        # Generate per type
        for strategy_type, count in distribution.items():
            if self.should_stop:
                break

            strategies = self.enumerator.enumerate_for_type(
                strategy_type=strategy_type,
                count=count,
                timeframe=timeframe,
                market_type=market_type
            )

            for strategy in strategies:
                if self.should_stop:
                    break

                self.progress += 1
                if callback:
                    callback(self.progress, self.total)

                yield strategy

        self.is_running = False

    def stop(self):
        """Request stop of batch generation"""
        self.should_stop = True

    def get_progress(self) -> Tuple[int, int]:
        """Get current progress"""
        return self.progress, self.total
