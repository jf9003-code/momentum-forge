"""
Institutional Strategy Generator - Main Class
Orchestrates the generation pipeline with clone filtering and validation
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging
import asyncio

from backend.app.generator.enumerator import ControlledEnumerator, BatchEnumerator, FilledTemplate
from backend.app.generator.validators import CloneFilter, StrategyValidator, ValidationResult
from backend.app.generator.templates import get_template, get_all_templates
from backend.app.generator.permissions import get_profile_for_type
from backend.app.core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a generation batch"""
    batch_id: str
    total_generated: int
    total_accepted: int
    total_rejected: int
    accepted_strategies: List[Dict]
    rejected_strategies: List[Dict]
    validation_summary: Dict
    clone_filter_stats: Dict
    duration_ms: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'batch_id': self.batch_id,
            'total_generated': self.total_generated,
            'total_accepted': self.total_accepted,
            'total_rejected': self.total_rejected,
            'accepted_strategies': self.accepted_strategies,
            'rejected_strategies': self.rejected_strategies,
            'validation_summary': self.validation_summary,
            'clone_filter_stats': self.clone_filter_stats,
            'duration_ms': self.duration_ms,
            'metadata': self.metadata
        }


class InstitutionalGenerator:
    """
    Main generator class for institutional trading strategies

    Features:
    - Top-down template-based generation
    - Corrected clone filter with adjusted thresholds
    - Batch generation with progress tracking
    - Async support with START/STOP control
    - Full validation pipeline
    """

    def __init__(self, seed: int = None):
        self.config = get_config().generator
        self.seed = seed or int(datetime.now().timestamp())

        self.enumerator = ControlledEnumerator(self.seed)
        self.clone_filter = CloneFilter()
        self.validator = StrategyValidator()

        # Batch control
        self.batch_enumerator = BatchEnumerator()
        self._is_running = False
        self._should_stop = False

        # Statistics
        self.stats = {
            'total_generated': 0,
            'total_accepted': 0,
            'total_rejected_clone': 0,
            'total_rejected_validation': 0
        }

    def generate(
        self,
        count: int,
        strategy_types: List[str] = None,
        timeframe: str = None,
        market_type: str = None,
        existing_strategies: List[Dict] = None
    ) -> GenerationResult:
        """
        Generate a batch of strategies synchronously

        Args:
            count: Number of strategies to generate
            strategy_types: List of types to include (None = all)
            timeframe: Optional timeframe filter
            market_type: Optional market filter
            existing_strategies: Existing strategies for clone comparison
        """
        start_time = datetime.now()
        batch_id = f"batch_{start_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Initialize clone filter with existing strategies
        self.clone_filter.clear()
        if existing_strategies:
            self.clone_filter.add_existing(existing_strategies)

        # Determine type distribution
        if strategy_types is None:
            strategy_types = ['mean_reversion', 'momentum', 'breakout',
                            'volatility_expansion', 'time_based']

        count_per_type = max(1, count // len(strategy_types))
        distribution = {t: count_per_type for t in strategy_types}

        # Adjust for remainder
        remainder = count - sum(distribution.values())
        for i, t in enumerate(strategy_types):
            if i < remainder:
                distribution[t] += 1

        # Generate strategies
        raw_strategies = self.enumerator.enumerate_mixed(
            type_distribution=distribution,
            timeframe=timeframe,
            market_type=market_type
        )

        # Convert to dicts and add IDs
        strategies = []
        for i, filled in enumerate(raw_strategies):
            strategy = filled.to_dict()
            strategy['id'] = f"{batch_id}_strat_{i:04d}"
            strategy['batch_id'] = batch_id
            strategies.append(strategy)

        # Validate and filter
        accepted, rejected, validation_summary, clone_stats = self._process_strategies(strategies)

        # Update stats
        self.stats['total_generated'] += len(strategies)
        self.stats['total_accepted'] += len(accepted)

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        return GenerationResult(
            batch_id=batch_id,
            total_generated=len(strategies),
            total_accepted=len(accepted),
            total_rejected=len(rejected),
            accepted_strategies=accepted,
            rejected_strategies=rejected,
            validation_summary=validation_summary,
            clone_filter_stats=clone_stats,
            duration_ms=duration_ms,
            metadata={
                'strategy_types': strategy_types,
                'timeframe': timeframe,
                'market_type': market_type,
                'seed': self.seed
            }
        )

    async def generate_async(
        self,
        count: int,
        strategy_types: List[str] = None,
        timeframe: str = None,
        market_type: str = None,
        existing_strategies: List[Dict] = None,
        progress_callback: callable = None
    ) -> GenerationResult:
        """
        Generate strategies asynchronously with progress tracking

        Supports cancellation via stop() method
        """
        start_time = datetime.now()
        batch_id = f"batch_{start_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        self._is_running = True
        self._should_stop = False

        # Initialize clone filter
        self.clone_filter.clear()
        if existing_strategies:
            self.clone_filter.add_existing(existing_strategies)

        # Determine distribution
        if strategy_types is None:
            strategy_types = ['mean_reversion', 'momentum', 'breakout',
                            'volatility_expansion', 'time_based']

        count_per_type = max(1, count // len(strategy_types))
        type_weights = {t: count_per_type / count for t in strategy_types}

        strategies = []

        def on_progress(current, total):
            if progress_callback:
                progress_callback(current, total, batch_id)

        # Generate with progress
        for filled in self.batch_enumerator.generate_batch(
            total_count=count,
            type_weights=type_weights,
            timeframe=timeframe,
            market_type=market_type,
            callback=on_progress
        ):
            if self._should_stop:
                break

            strategy = filled.to_dict()
            strategy['id'] = f"{batch_id}_strat_{len(strategies):04d}"
            strategy['batch_id'] = batch_id
            strategies.append(strategy)

            # Yield control periodically
            if len(strategies) % 10 == 0:
                await asyncio.sleep(0)

        self._is_running = False

        # Process generated strategies
        accepted, rejected, validation_summary, clone_stats = self._process_strategies(strategies)

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        return GenerationResult(
            batch_id=batch_id,
            total_generated=len(strategies),
            total_accepted=len(accepted),
            total_rejected=len(rejected),
            accepted_strategies=accepted,
            rejected_strategies=rejected,
            validation_summary=validation_summary,
            clone_filter_stats=clone_stats,
            duration_ms=duration_ms,
            metadata={
                'strategy_types': strategy_types,
                'timeframe': timeframe,
                'market_type': market_type,
                'seed': self.seed,
                'was_cancelled': self._should_stop
            }
        )

    def _process_strategies(
        self,
        strategies: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], Dict, Dict]:
        """Process strategies through validation and clone filter"""
        accepted = []
        rejected = []

        validation_results = {
            'total': len(strategies),
            'valid': 0,
            'invalid': 0,
            'errors': []
        }

        clone_stats = {
            'checked': len(strategies),
            'clones_detected': 0,
            'unique': 0,
            'avg_distance': 0.0
        }

        distances = []

        for strategy in strategies:
            # Validate first
            val_result = self.validator.validate(strategy)

            if not val_result.is_valid:
                validation_results['invalid'] += 1
                strategy['_rejection_reason'] = f"Validation failed: {val_result.errors}"
                rejected.append(strategy)
                self.stats['total_rejected_validation'] += 1
                continue

            validation_results['valid'] += 1

            # Clone filter
            is_clone, match_id, distance = self.clone_filter.is_clone(strategy)
            distances.append(distance)

            if is_clone:
                clone_stats['clones_detected'] += 1
                strategy['_rejection_reason'] = f"Clone of {match_id} (distance: {distance:.4f})"
                rejected.append(strategy)
                self.stats['total_rejected_clone'] += 1
            else:
                clone_stats['unique'] += 1
                strategy['_clone_distance'] = distance
                accepted.append(strategy)

                # Add to clone filter for subsequent checks
                self.clone_filter.add_existing([strategy])

        # Calculate average distance
        if distances:
            clone_stats['avg_distance'] = round(sum(distances) / len(distances), 4)

        return accepted, rejected, validation_results, clone_stats

    def stop(self):
        """Request stop of async generation"""
        self._should_stop = True
        self.batch_enumerator.stop()

    def is_running(self) -> bool:
        """Check if generation is in progress"""
        return self._is_running

    def get_progress(self) -> Tuple[int, int]:
        """Get current generation progress"""
        return self.batch_enumerator.get_progress()

    def reset(self, new_seed: int = None):
        """Reset generator state"""
        self.seed = new_seed or int(datetime.now().timestamp())
        self.enumerator.reset(self.seed)
        self.clone_filter.clear()
        self.stats = {
            'total_generated': 0,
            'total_accepted': 0,
            'total_rejected_clone': 0,
            'total_rejected_validation': 0
        }

    def get_stats(self) -> Dict:
        """Get generation statistics"""
        return self.stats.copy()

    def get_available_templates(self) -> List[Dict]:
        """Get all available templates"""
        templates = get_all_templates()
        return [t.to_dict() for t in templates]


class GeneratorService:
    """
    Service layer for strategy generation

    Provides high-level API for generation workflows
    """

    _instance: Optional['GeneratorService'] = None
    _generator: Optional[InstitutionalGenerator] = None

    @classmethod
    def get_instance(cls) -> 'GeneratorService':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._generator = InstitutionalGenerator()
        self._current_batch_id: Optional[str] = None

    def generate_batch(
        self,
        count: int,
        strategy_types: List[str] = None,
        timeframe: str = None,
        market_type: str = None,
        existing_strategies: List[Dict] = None
    ) -> GenerationResult:
        """Generate a batch of strategies"""
        result = self._generator.generate(
            count=count,
            strategy_types=strategy_types,
            timeframe=timeframe,
            market_type=market_type,
            existing_strategies=existing_strategies
        )
        self._current_batch_id = result.batch_id
        return result

    async def generate_batch_async(
        self,
        count: int,
        strategy_types: List[str] = None,
        timeframe: str = None,
        market_type: str = None,
        existing_strategies: List[Dict] = None,
        progress_callback: callable = None
    ) -> GenerationResult:
        """Generate batch asynchronously"""
        result = await self._generator.generate_async(
            count=count,
            strategy_types=strategy_types,
            timeframe=timeframe,
            market_type=market_type,
            existing_strategies=existing_strategies,
            progress_callback=progress_callback
        )
        self._current_batch_id = result.batch_id
        return result

    def stop_generation(self):
        """Stop current generation"""
        self._generator.stop()

    def is_generating(self) -> bool:
        """Check if generation is in progress"""
        return self._generator.is_running()

    def get_progress(self) -> Dict:
        """Get generation progress"""
        current, total = self._generator.get_progress()
        return {
            'current': current,
            'total': total,
            'percent': round(current / total * 100, 1) if total > 0 else 0,
            'batch_id': self._current_batch_id
        }

    def get_stats(self) -> Dict:
        """Get generator statistics"""
        return self._generator.get_stats()

    def get_templates(self) -> List[Dict]:
        """Get available templates"""
        return self._generator.get_available_templates()

    def reset(self, seed: int = None):
        """Reset generator"""
        self._generator.reset(seed)
        self._current_batch_id = None
