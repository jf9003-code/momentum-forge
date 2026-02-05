"""
Top 5 Strategies API Routes
Handles Top 5 management with proper accumulation (FIXED)
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from backend.app.core.database import (
    save_top5_strategies,
    get_top5_strategies,
    get_top5_history,
    get_strategy_by_id,
    clear_top5_strategies
)

router = APIRouter(prefix="/top5", tags=["top5"])


class Top5SaveRequest(BaseModel):
    """Request to save strategies to Top 5"""
    strategy_ids: List[str] = Field(
        ...,
        min_items=1,
        max_items=5,
        description="IDs of strategies to add to Top 5"
    )
    batch_name: Optional[str] = Field(
        default=None,
        description="Optional name for this batch"
    )


class Top5Strategy(BaseModel):
    """Model for a Top 5 strategy"""
    id: str
    strategy_id: str
    batch_id: str
    strategy_type: str
    template_id: Optional[str]
    overall_score: float
    validation_passed: bool
    created_at: str
    data: Dict[str, Any]


class Top5Response(BaseModel):
    """Response model for Top 5 operations"""
    batch_id: str
    strategies_added: int
    total_top5: int
    message: str


@router.post("/save", response_model=Top5Response)
async def save_to_top5(request: Top5SaveRequest):
    """
    Save strategies to Top 5

    FIXED: Now properly accumulates across batches instead of overwriting.
    Each save creates a new batch_id and strategies are stored with
    unique compound IDs (batch_id + original_id).
    """
    # Fetch full strategy data
    strategies = []
    not_found = []

    for sid in request.strategy_ids:
        strategy = get_strategy_by_id(sid)
        if strategy:
            strategies.append(strategy)
        else:
            not_found.append(sid)

    if not_found:
        raise HTTPException(
            status_code=404,
            detail=f"Strategies not found: {', '.join(not_found)}"
        )

    if not strategies:
        raise HTTPException(
            status_code=400,
            detail="No valid strategies to save"
        )

    # Create batch name
    batch_name = request.batch_name or f"top5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Save with unique batch_id (accumulation, not overwrite)
    batch_id = save_top5_strategies(strategies, batch_name)

    # Get current total
    all_top5 = get_top5_strategies()

    return Top5Response(
        batch_id=batch_id,
        strategies_added=len(strategies),
        total_top5=len(all_top5),
        message=f"Added {len(strategies)} strategies to Top 5 (batch: {batch_id})"
    )


@router.get("/list")
async def list_top5(
    limit: Optional[int] = None,
    batch_id: Optional[str] = None
):
    """
    List Top 5 strategies

    Returns all accumulated Top 5 strategies across all batches,
    or filtered by batch_id if specified.
    """
    strategies = get_top5_strategies(limit=limit, batch_id=batch_id)

    # Group by batch for summary
    batches = {}
    for s in strategies:
        bid = s.get('batch_id', 'unknown')
        if bid not in batches:
            batches[bid] = []
        batches[bid].append(s)

    return {
        "total": len(strategies),
        "batch_count": len(batches),
        "batches": {
            bid: len(strats) for bid, strats in batches.items()
        },
        "strategies": strategies
    }


@router.get("/history")
async def get_history():
    """
    Get Top 5 batch history

    Shows all batches that have been saved to Top 5
    """
    history = get_top5_history()

    return {
        "total_batches": len(history),
        "batches": history
    }


@router.get("/batch/{batch_id}")
async def get_batch(batch_id: str):
    """Get strategies from a specific Top 5 batch"""
    strategies = get_top5_strategies(batch_id=batch_id)

    if not strategies:
        raise HTTPException(
            status_code=404,
            detail=f"Batch not found: {batch_id}"
        )

    return {
        "batch_id": batch_id,
        "count": len(strategies),
        "strategies": strategies
    }


@router.delete("/clear")
async def clear_all_top5(confirm: bool = False):
    """
    Clear all Top 5 strategies

    Requires confirm=true to prevent accidental deletion
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to clear Top 5"
        )

    count = clear_top5_strategies()

    return {
        "status": "cleared",
        "strategies_removed": count,
        "message": "All Top 5 strategies have been removed"
    }


@router.post("/promote/{strategy_id}")
async def promote_to_validation(strategy_id: str):
    """
    Promote a Top 5 strategy to 7-Phase validation

    This creates a validation request for the strategy
    """
    # Find strategy in Top 5
    top5 = get_top5_strategies()
    strategy = None

    for s in top5:
        if s.get('id') == strategy_id or s.get('strategy_id') == strategy_id:
            strategy = s
            break

    if not strategy:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy not found in Top 5: {strategy_id}"
        )

    # Return validation request details
    return {
        "status": "ready_for_validation",
        "strategy_id": strategy.get('id') or strategy.get('strategy_id'),
        "strategy_type": strategy.get('strategy_type'),
        "validation_endpoint": "/validation/validate",
        "message": "Use POST /validation/validate with this strategy_id to start 7-Phase validation"
    }


@router.get("/stats")
async def get_top5_stats():
    """Get statistics about Top 5 strategies"""
    strategies = get_top5_strategies()
    history = get_top5_history()

    # Calculate stats
    if strategies:
        type_counts = {}
        scores = []

        for s in strategies:
            stype = s.get('strategy_type', 'unknown')
            type_counts[stype] = type_counts.get(stype, 0) + 1

            score = s.get('overall_score') or s.get('_clone_distance', 0)
            if score:
                scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0
    else:
        type_counts = {}
        avg_score = 0

    return {
        "total_strategies": len(strategies),
        "total_batches": len(history),
        "type_distribution": type_counts,
        "average_score": round(avg_score, 4),
        "oldest_batch": history[0]['batch_id'] if history else None,
        "newest_batch": history[-1]['batch_id'] if history else None
    }
