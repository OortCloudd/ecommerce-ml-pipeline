"""
FastAPI application for e-commerce recommendations
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .prediction_service import (
    PredictionService,
    RecommendationRequest,
    UserEvent,
    RecommendationResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="E-commerce Recommendation API",
    description="""
    Real-time recommendation API for e-commerce applications.
    Features:
    - Personalized product recommendations
    - Confidence intervals for predictions
    - Real-time event processing
    - Background feature updates
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with actual domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
CACHE_DIR = os.getenv('CACHE_DIR', 'cache')
FEATURE_UPDATE_INTERVAL = int(os.getenv('FEATURE_UPDATE_INTERVAL', '3600'))

prediction_service = PredictionService(
    model_dir=MODEL_DIR,
    cache_dir=CACHE_DIR,
    feature_update_interval=FEATURE_UPDATE_INTERVAL
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global exception handler caught: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint
    
    Returns:
        Dict containing service health status and metadata
    """
    try:
        return {
            "status": "healthy",
            "model_version": prediction_service.model_version,
            "last_feature_update": prediction_service.last_feature_update.isoformat(),
            "service_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service unhealthy"
        )

@app.post(
    "/recommendations",
    response_model=RecommendationResponse,
    response_model_exclude_none=True
)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks
) -> RecommendationResponse:
    """
    Get personalized recommendations for a user
    
    Args:
        request: Recommendation request containing user ID and preferences
        background_tasks: FastAPI background tasks
        
    Returns:
        RecommendationResponse containing recommended items with confidence intervals
        
    Raises:
        HTTPException: If recommendations cannot be generated
    """
    try:
        recommendations = prediction_service.get_recommendations(
            request,
            background_tasks
        )
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )

@app.post("/events")
async def record_event(
    event: UserEvent,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """
    Record a new user interaction event
    
    Args:
        event: User interaction event
        background_tasks: FastAPI background tasks
        
    Returns:
        Dict containing operation status
        
    Raises:
        HTTPException: If event cannot be processed
    """
    try:
        prediction_service.update_event(event, background_tasks)
        return {
            "status": "success",
            "message": "Event recorded successfully"
        }
    except Exception as e:
        logger.error(f"Error recording event: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record event: {str(e)}"
        )

@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get service metrics
    
    Returns:
        Dict containing service metrics and statistics
    """
    try:
        current_time = datetime.now()
        feature_age = (
            current_time - prediction_service.last_feature_update
        ).total_seconds()
        
        return {
            "model_version": prediction_service.model_version,
            "feature_freshness_seconds": feature_age,
            "cache_size": {
                "users": len(prediction_service.user_features_cache),
                "items": len(prediction_service.item_features_cache)
            },
            "service_uptime": (
                current_time - app.state.start_time
            ).total_seconds() if hasattr(app.state, 'start_time') else 0
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    app.state.start_time = datetime.now()
    logger.info("Service started successfully")

def custom_openapi():
    """Custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security schemes if needed
    # openapi_schema["components"]["securitySchemes"] = {...}
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
