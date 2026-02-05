"""
Validation API Routes
Handles 7-phase validation pipeline
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid

from backend.app.core.database import (
    save_validation_result,
    get_validation_results,
    get_strategy_by_id
)

router = APIRouter(prefix="/validation", tags=["validation"])


class ValidationPhase(str, Enum):
    """7-Phase validation pipeline"""
    PHASE_1_STRUCTURAL = "structural"
    PHASE_2_LOGICAL = "logical"
    PHASE_3_BACKTEST = "backtest"
    PHASE_4_MONTE_CARLO = "monte_carlo"
    PHASE_5_WALK_FORWARD = "walk_forward"
    PHASE_6_OUT_OF_SAMPLE = "out_of_sample"
    PHASE_7_ROBUSTNESS = "robustness"


class PhaseResult(BaseModel):
    """Result of a single validation phase"""
    phase: ValidationPhase
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    metrics: Dict[str, Any] = {}
    details: Optional[str] = None
    duration_ms: float = 0


class ValidationRequest(BaseModel):
    """Request model for validation"""
    strategy_id: str
    phases: Optional[List[ValidationPhase]] = Field(
        default=None,
        description="Phases to run (None = all)"
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase-specific configuration"
    )


class ValidationResponse(BaseModel):
    """Response model for validation"""
    validation_id: str
    strategy_id: str
    overall_passed: bool
    overall_score: float
    phases_completed: int
    phase_results: List[PhaseResult]
    duration_ms: float


class BatchValidationRequest(BaseModel):
    """Request for batch validation"""
    strategy_ids: List[str]
    phases: Optional[List[ValidationPhase]] = None
    stop_on_failure: bool = Field(
        default=True,
        description="Stop validating strategy on first phase failure"
    )


# Mock validation functions (to be replaced with real implementations)
def _validate_structural(strategy: Dict, config: Dict) -> PhaseResult:
    """Phase 1: Structural validation"""
    errors = []

    # Check required fields
    if not strategy.get('entry_logic'):
        errors.append("Missing entry logic")
    if not strategy.get('exit_logic'):
        errors.append("Missing exit logic")
    if not strategy.get('indicators'):
        errors.append("No indicators defined")

    passed = len(errors) == 0
    score = 1.0 - (len(errors) * 0.25)

    return PhaseResult(
        phase=ValidationPhase.PHASE_1_STRUCTURAL,
        passed=passed,
        score=max(0.0, score),
        metrics={"error_count": len(errors)},
        details="; ".join(errors) if errors else "Structural validation passed"
    )


def _validate_logical(strategy: Dict, config: Dict) -> PhaseResult:
    """Phase 2: Logical validation"""
    warnings = []

    # Check for logical consistency
    entry = strategy.get('entry_logic', {})
    exit_logic = strategy.get('exit_logic', {})

    entry_conditions = entry.get('conditions', [])
    exit_conditions = exit_logic.get('conditions', [])

    if not entry_conditions:
        warnings.append("No entry conditions")
    if not exit_conditions:
        warnings.append("No exit conditions (will use stop/target only)")

    # Check risk config
    risk = strategy.get('risk_config', {})
    if not risk.get('atr_sl_mult') and not risk.get('stop_loss_atr'):
        warnings.append("No stop loss defined")

    passed = len([w for w in warnings if "No entry" in w or "No exit" in w]) == 0
    score = 1.0 - (len(warnings) * 0.15)

    return PhaseResult(
        phase=ValidationPhase.PHASE_2_LOGICAL,
        passed=passed,
        score=max(0.0, score),
        metrics={"warning_count": len(warnings)},
        details="; ".join(warnings) if warnings else "Logical validation passed"
    )


def _validate_backtest(strategy: Dict, config: Dict) -> PhaseResult:
    """Phase 3: Backtest validation (mock)"""
    # This would run actual backtest
    # For now, return mock result

    import random
    random.seed(hash(str(strategy.get('id', ''))))

    win_rate = random.uniform(0.35, 0.65)
    profit_factor = random.uniform(0.8, 2.5)
    sharpe = random.uniform(-0.5, 2.0)
    max_dd = random.uniform(0.05, 0.35)

    passed = profit_factor > 1.0 and win_rate > 0.4 and sharpe > 0
    score = min(1.0, (profit_factor - 1.0) * 0.5 + win_rate * 0.3 + max(0, sharpe) * 0.2)

    return PhaseResult(
        phase=ValidationPhase.PHASE_3_BACKTEST,
        passed=passed,
        score=max(0.0, min(1.0, score)),
        metrics={
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4)
        },
        details=f"PF: {profit_factor:.2f}, WR: {win_rate:.1%}, Sharpe: {sharpe:.2f}"
    )


def _validate_monte_carlo(strategy: Dict, config: Dict) -> PhaseResult:
    """Phase 4: Monte Carlo validation (mock)"""
    import random
    random.seed(hash(str(strategy.get('id', '')) + "mc"))

    confidence_95 = random.uniform(0.7, 1.3)
    ruin_prob = random.uniform(0.01, 0.15)

    passed = confidence_95 > 0.9 and ruin_prob < 0.10
    score = min(1.0, confidence_95 * 0.7 + (1 - ruin_prob) * 0.3)

    return PhaseResult(
        phase=ValidationPhase.PHASE_4_MONTE_CARLO,
        passed=passed,
        score=max(0.0, score),
        metrics={
            "confidence_95_pf": round(confidence_95, 4),
            "ruin_probability": round(ruin_prob, 4),
            "simulations": 1000
        },
        details=f"95% Confidence PF: {confidence_95:.2f}, Ruin: {ruin_prob:.1%}"
    )


def _validate_walk_forward(strategy: Dict, config: Dict) -> PhaseResult:
    """Phase 5: Walk Forward validation (mock)"""
    import random
    random.seed(hash(str(strategy.get('id', '')) + "wf"))

    efficiency = random.uniform(0.3, 0.9)
    stability = random.uniform(0.4, 1.0)

    passed = efficiency > 0.5 and stability > 0.6
    score = efficiency * 0.6 + stability * 0.4

    return PhaseResult(
        phase=ValidationPhase.PHASE_5_WALK_FORWARD,
        passed=passed,
        score=max(0.0, min(1.0, score)),
        metrics={
            "walk_forward_efficiency": round(efficiency, 4),
            "parameter_stability": round(stability, 4),
            "windows": 5
        },
        details=f"WF Efficiency: {efficiency:.1%}, Stability: {stability:.1%}"
    )


def _validate_out_of_sample(strategy: Dict, config: Dict) -> PhaseResult:
    """Phase 6: Out of Sample validation (mock)"""
    import random
    random.seed(hash(str(strategy.get('id', '')) + "oos"))

    oos_ratio = random.uniform(0.5, 1.2)
    degradation = random.uniform(0.0, 0.4)

    passed = oos_ratio > 0.7 and degradation < 0.3
    score = oos_ratio * 0.7 + (1 - degradation) * 0.3

    return PhaseResult(
        phase=ValidationPhase.PHASE_6_OUT_OF_SAMPLE,
        passed=passed,
        score=max(0.0, min(1.0, score)),
        metrics={
            "oos_is_ratio": round(oos_ratio, 4),
            "performance_degradation": round(degradation, 4)
        },
        details=f"OOS/IS: {oos_ratio:.2f}, Degradation: {degradation:.1%}"
    )


def _validate_robustness(strategy: Dict, config: Dict) -> PhaseResult:
    """Phase 7: Robustness validation (mock)"""
    import random
    random.seed(hash(str(strategy.get('id', '')) + "rob"))

    param_sensitivity = random.uniform(0.3, 0.9)
    market_diversity = random.uniform(0.4, 1.0)
    time_stability = random.uniform(0.5, 1.0)

    passed = param_sensitivity > 0.5 and market_diversity > 0.5 and time_stability > 0.6
    score = param_sensitivity * 0.4 + market_diversity * 0.3 + time_stability * 0.3

    return PhaseResult(
        phase=ValidationPhase.PHASE_7_ROBUSTNESS,
        passed=passed,
        score=max(0.0, min(1.0, score)),
        metrics={
            "parameter_sensitivity": round(param_sensitivity, 4),
            "market_diversity_score": round(market_diversity, 4),
            "time_stability_score": round(time_stability, 4)
        },
        details=f"Param: {param_sensitivity:.1%}, Market: {market_diversity:.1%}, Time: {time_stability:.1%}"
    )


PHASE_VALIDATORS = {
    ValidationPhase.PHASE_1_STRUCTURAL: _validate_structural,
    ValidationPhase.PHASE_2_LOGICAL: _validate_logical,
    ValidationPhase.PHASE_3_BACKTEST: _validate_backtest,
    ValidationPhase.PHASE_4_MONTE_CARLO: _validate_monte_carlo,
    ValidationPhase.PHASE_5_WALK_FORWARD: _validate_walk_forward,
    ValidationPhase.PHASE_6_OUT_OF_SAMPLE: _validate_out_of_sample,
    ValidationPhase.PHASE_7_ROBUSTNESS: _validate_robustness,
}


@router.post("/validate", response_model=ValidationResponse)
async def validate_strategy(request: ValidationRequest):
    """
    Validate a strategy through the 7-phase pipeline
    """
    # Get strategy
    strategy = get_strategy_by_id(request.strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy not found: {request.strategy_id}"
        )

    start_time = datetime.now()
    validation_id = f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Determine phases to run
    phases = request.phases or list(ValidationPhase)
    config = request.config or {}

    phase_results = []
    overall_passed = True

    for phase in phases:
        validator = PHASE_VALIDATORS.get(phase)
        if not validator:
            continue

        phase_start = datetime.now()
        result = validator(strategy, config)
        result.duration_ms = (datetime.now() - phase_start).total_seconds() * 1000
        phase_results.append(result)

        if not result.passed:
            overall_passed = False

    # Calculate overall score
    if phase_results:
        overall_score = sum(r.score for r in phase_results) / len(phase_results)
    else:
        overall_score = 0.0

    duration_ms = (datetime.now() - start_time).total_seconds() * 1000

    # Save result
    validation_data = {
        'validation_id': validation_id,
        'strategy_id': request.strategy_id,
        'overall_passed': overall_passed,
        'overall_score': overall_score,
        'phases_completed': len(phase_results),
        'phase_results': [r.dict() for r in phase_results],
        'duration_ms': duration_ms,
        'created_at': datetime.now().isoformat()
    }
    save_validation_result(validation_data)

    return ValidationResponse(
        validation_id=validation_id,
        strategy_id=request.strategy_id,
        overall_passed=overall_passed,
        overall_score=round(overall_score, 4),
        phases_completed=len(phase_results),
        phase_results=phase_results,
        duration_ms=duration_ms
    )


@router.post("/batch")
async def batch_validate(request: BatchValidationRequest):
    """
    Validate multiple strategies
    """
    results = []

    for strategy_id in request.strategy_ids:
        try:
            req = ValidationRequest(
                strategy_id=strategy_id,
                phases=request.phases
            )
            result = await validate_strategy(req)
            results.append({
                "strategy_id": strategy_id,
                "passed": result.overall_passed,
                "score": result.overall_score,
                "validation_id": result.validation_id
            })

            if request.stop_on_failure and not result.overall_passed:
                break

        except HTTPException as e:
            results.append({
                "strategy_id": strategy_id,
                "error": e.detail
            })

    passed_count = sum(1 for r in results if r.get('passed', False))

    return {
        "total": len(request.strategy_ids),
        "validated": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "results": results
    }


@router.get("/results/{strategy_id}")
async def get_validation_history(strategy_id: str):
    """Get validation history for a strategy"""
    results = get_validation_results(strategy_id)

    return {
        "strategy_id": strategy_id,
        "count": len(results),
        "validations": results
    }


@router.get("/phases")
async def list_phases():
    """List all validation phases"""
    return {
        "phases": [
            {
                "id": phase.value,
                "name": phase.name,
                "order": i + 1
            }
            for i, phase in enumerate(ValidationPhase)
        ]
    }
