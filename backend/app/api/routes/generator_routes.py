"""
Generator API Routes
Handles strategy generation endpoints with batch control
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime

from backend.app.generator.generator import GeneratorService, GenerationResult
from backend.app.core.database import (
    save_generated_strategies,
    get_generated_strategies,
    get_top5_strategies
)

router = APIRouter(prefix="/generator", tags=["generator"])

# Global state for batch control
_generation_task: Optional[asyncio.Task] = None
_generation_result: Optional[GenerationResult] = None


class GenerateRequest(BaseModel):
    """Request model for strategy generation"""
    count: int = Field(default=10, ge=1, le=100, description="Number of strategies to generate")
    strategy_types: Optional[List[str]] = Field(
        default=None,
        description="Strategy types to include"
    )
    timeframe: Optional[str] = Field(default=None, description="Timeframe filter")
    market_type: Optional[str] = Field(default=None, description="Market type filter")
    use_existing: bool = Field(
        default=True,
        description="Use existing strategies for clone comparison"
    )


class GenerateResponse(BaseModel):
    """Response model for generation"""
    batch_id: str
    total_generated: int
    total_accepted: int
    total_rejected: int
    strategies: List[Dict[str, Any]]
    duration_ms: float
    clone_filter_stats: Dict[str, Any]


class ProgressResponse(BaseModel):
    """Response model for progress"""
    is_running: bool
    current: int
    total: int
    percent: float
    batch_id: Optional[str]


@router.post("/generate", response_model=GenerateResponse)
async def generate_strategies(request: GenerateRequest):
    """
    Generate a batch of strategies synchronously

    Returns immediately with generated strategies
    """
    service = GeneratorService.get_instance()

    if service.is_generating():
        raise HTTPException(
            status_code=409,
            detail="Generation already in progress. Use /stop to cancel or wait."
        )

    # Get existing strategies for clone comparison
    existing = []
    if request.use_existing:
        existing = get_generated_strategies(limit=500)
        # Also include top 5 strategies
        top5 = get_top5_strategies()
        existing.extend(top5)

    # Generate
    result = service.generate_batch(
        count=request.count,
        strategy_types=request.strategy_types,
        timeframe=request.timeframe,
        market_type=request.market_type,
        existing_strategies=existing
    )

    # Save to database
    if result.accepted_strategies:
        save_generated_strategies(result.accepted_strategies, result.batch_id)

    return GenerateResponse(
        batch_id=result.batch_id,
        total_generated=result.total_generated,
        total_accepted=result.total_accepted,
        total_rejected=result.total_rejected,
        strategies=result.accepted_strategies,
        duration_ms=result.duration_ms,
        clone_filter_stats=result.clone_filter_stats
    )


@router.post("/start")
async def start_generation(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """
    Start async batch generation

    Returns immediately, use /progress to check status
    """
    global _generation_task, _generation_result

    service = GeneratorService.get_instance()

    if service.is_generating():
        raise HTTPException(
            status_code=409,
            detail="Generation already in progress"
        )

    # Get existing for clone comparison
    existing = []
    if request.use_existing:
        existing = get_generated_strategies(limit=500)
        top5 = get_top5_strategies()
        existing.extend(top5)

    async def run_generation():
        global _generation_result
        result = await service.generate_batch_async(
            count=request.count,
            strategy_types=request.strategy_types,
            timeframe=request.timeframe,
            market_type=request.market_type,
            existing_strategies=existing
        )
        _generation_result = result

        # Save to database
        if result.accepted_strategies:
            save_generated_strategies(result.accepted_strategies, result.batch_id)

    _generation_task = asyncio.create_task(run_generation())

    return {
        "status": "started",
        "message": f"Generation of {request.count} strategies started",
        "use_progress_endpoint": "/generator/progress"
    }


@router.post("/stop")
async def stop_generation():
    """Stop ongoing generation"""
    service = GeneratorService.get_instance()

    if not service.is_generating():
        return {"status": "not_running", "message": "No generation in progress"}

    service.stop_generation()

    return {
        "status": "stopping",
        "message": "Generation stop requested"
    }


@router.get("/progress", response_model=ProgressResponse)
async def get_progress():
    """Get current generation progress"""
    service = GeneratorService.get_instance()
    progress = service.get_progress()

    return ProgressResponse(
        is_running=service.is_generating(),
        current=progress['current'],
        total=progress['total'],
        percent=progress['percent'],
        batch_id=progress['batch_id']
    )


@router.get("/result")
async def get_result():
    """Get result of last generation"""
    global _generation_result

    service = GeneratorService.get_instance()

    if service.is_generating():
        raise HTTPException(
            status_code=409,
            detail="Generation still in progress"
        )

    if _generation_result is None:
        raise HTTPException(
            status_code=404,
            detail="No generation result available"
        )

    return _generation_result.to_dict()


@router.get("/stats")
async def get_stats():
    """Get generator statistics"""
    service = GeneratorService.get_instance()
    return service.get_stats()


@router.get("/templates")
async def get_templates():
    """Get available strategy templates"""
    service = GeneratorService.get_instance()
    templates = service.get_templates()
    return {
        "count": len(templates),
        "templates": templates
    }


@router.post("/reset")
async def reset_generator(seed: Optional[int] = None):
    """Reset generator state"""
    service = GeneratorService.get_instance()

    if service.is_generating():
        raise HTTPException(
            status_code=409,
            detail="Cannot reset while generation in progress"
        )

    service.reset(seed)

    return {
        "status": "reset",
        "message": "Generator state reset",
        "seed": seed
    }


@router.get("/strategies")
async def list_generated_strategies(
    limit: int = 50,
    offset: int = 0,
    batch_id: Optional[str] = None
):
    """List generated strategies from database"""
    strategies = get_generated_strategies(
        limit=limit,
        offset=offset,
        batch_id=batch_id
    )

    return {
        "count": len(strategies),
        "strategies": strategies
    }
