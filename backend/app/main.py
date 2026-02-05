"""
D5 Robust Master - Backend Main Entry Point
Institutional Trading Strategy Validation Platform
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
from datetime import datetime

from backend.app.core.config import get_config
from backend.app.core.database import init_database
from backend.app.api.routes import generator_router, validation_router, top5_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting D5 Robust Master Backend...")

    # Initialize database
    init_database()
    logger.info("Database initialized")

    # Load config
    config = get_config()
    logger.info(f"Config loaded: {config.app.app_name}")

    yield

    # Shutdown
    logger.info("Shutting down D5 Robust Master Backend...")


# Create FastAPI app
app = FastAPI(
    title="D5 Robust Master",
    description="Institutional Trading Strategy Validation Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if get_config().app.debug else "An error occurred"
        }
    )


# Include routers
app.include_router(generator_router)
app.include_router(validation_router)
app.include_router(top5_router)


# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "D5 Robust Master",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "generator": "/generator",
            "validation": "/validation",
            "top5": "/top5",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# API info
@app.get("/info")
async def api_info():
    """Get API information"""
    config = get_config()
    return {
        "app_name": config.app.app_name,
        "version": config.app.version,
        "debug": config.app.debug,
        "features": {
            "generator": {
                "min_structural_distance": config.generator.min_structural_distance,
                "max_batch_size": config.generator.max_batch_size
            },
            "validation": {
                "phases": 7
            },
            "top5": {
                "accumulation": True,
                "batch_tracking": True
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
