"""API Routes Package"""
from backend.app.api.routes.generator_routes import router as generator_router
from backend.app.api.routes.validation_routes import router as validation_router
from backend.app.api.routes.top5_routes import router as top5_router

__all__ = ['generator_router', 'validation_router', 'top5_router']
