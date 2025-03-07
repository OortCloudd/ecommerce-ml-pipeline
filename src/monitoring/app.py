"""
FastAPI application for model monitoring
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import logging
from datetime import datetime, timedelta
import pandas as pd

from .monitoring_service import MonitoringService, MonitoringConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="E-commerce Model Monitoring API",
    description="""
    Model monitoring API for e-commerce recommendations.
    Features:
    - Performance metrics tracking
    - Data drift detection
    - Retraining triggers
    - Metrics visualization
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

# Initialize monitoring service
monitoring_config = MonitoringConfig(
    performance_threshold=float(os.getenv('PERFORMANCE_THRESHOLD', '0.1')),
    drift_threshold=float(os.getenv('DRIFT_THRESHOLD', '0.05')),
    monitoring_window_days=int(os.getenv('MONITORING_WINDOW_DAYS', '7')),
    s3_bucket=os.getenv('S3_BUCKET', 'ecommerce-ml-pipeline-data'),
    s3_prefix=os.getenv('S3_PREFIX', 'monitoring')
)

monitoring_service = MonitoringService(
    config=monitoring_config,
    metrics_dir=os.getenv('METRICS_DIR', 'monitoring_metrics')
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
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Dict containing service health status
    """
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service unhealthy"
        )

@app.post("/metrics/collect")
async def collect_metrics(
    predictions_file: str,
    actuals_file: str,
    model_version: str,
    background_tasks: BackgroundTasks
):
    """
    Collect metrics from prediction and actual data
    
    Args:
        predictions_file: Path to predictions parquet file
        actuals_file: Path to actuals parquet file
        model_version: Model version
        background_tasks: FastAPI background tasks
        
    Returns:
        Dict containing collected metrics
    """
    try:
        # Load data
        predictions_df = pd.read_parquet(predictions_file)
        actuals_df = pd.read_parquet(actuals_file)
        
        # Collect metrics
        metrics = await monitoring_service.collect_metrics(
            predictions_df,
            actuals_df,
            model_version,
            background_tasks
        )
        
        return metrics
    except Exception as e:
        logger.error(f"Error collecting metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to collect metrics: {str(e)}"
        )

@app.get("/metrics/summary")
async def get_metrics_summary(days: int = None):
    """
    Get summary of recent metrics
    
    Args:
        days: Number of days to include in summary
        
    Returns:
        Dict containing metrics summary
    """
    try:
        summary = monitoring_service.get_metrics_summary(days)
        return summary
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics summary: {str(e)}"
        )

@app.get("/metrics/retraining")
async def check_retraining():
    """
    Check if model retraining is needed
    
    Returns:
        Dict containing retraining status
    """
    try:
        retraining_needed = monitoring_service.metrics_collector.check_retraining_needed(
            performance_threshold=monitoring_config.performance_threshold,
            drift_threshold=monitoring_config.drift_threshold
        )
        
        return {
            "retraining_needed": retraining_needed,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking retraining status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check retraining status: {str(e)}"
        )
