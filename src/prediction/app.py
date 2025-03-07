"""
FastAPI application for prediction service
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import boto3
import os
import json
from pathlib import Path
import logging
from prometheus_client import Counter, Histogram, generate_latest

from ..model.collaborative.als_trainer import ALSTrainer
from ..model.ranking.conformal_trainer import ConformalRankingTrainer
from ..processing.feature_engineering import AdvancedFeatureEngineer
from ..monitoring.metrics_collector import MetricsCollector

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Prometheus metrics
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time spent processing prediction request')
PREDICTION_REQUESTS = Counter('prediction_requests_total', 'Total prediction requests')
FEATURE_UPDATE_ERRORS = Counter('feature_update_errors_total', 'Feature update errors')

app = FastAPI(title="E-commerce Recommendation Service")

# Models
class RecommendationRequest(BaseModel):
    user_id: str
    n_items: int = 10
    include_metadata: bool = False
    min_confidence: float = 0.0

class EventRecord(BaseModel):
    user_id: str
    item_id: str
    event_type: str
    timestamp: str
    metadata: Optional[Dict] = None

class Recommendation(BaseModel):
    item_id: str
    score: float
    confidence_lower: float
    confidence_upper: float
    metadata: Optional[Dict] = None

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]
    model_version: str
    feature_timestamp: str

# Global variables
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")
FEATURE_PATH = os.getenv("FEATURE_PATH", "/app/features")
S3_BUCKET = os.getenv("S3_BUCKET", "ecommerce-ml-pipeline-data")
FEATURE_UPDATE_INTERVAL = int(os.getenv("FEATURE_UPDATE_INTERVAL", "3600"))  # 1 hour default

# Initialize clients and models
s3_client = boto3.client('s3', region_name=os.getenv("AWS_DEFAULT_REGION", "eu-west-1"))
als_model = ALSTrainer()
ranking_model = ConformalRankingTrainer()
feature_engineer = AdvancedFeatureEngineer()
metrics_collector = MetricsCollector(
    metrics_dir="/app/metrics",
    monitoring_window_days=7
)

# Cache for features
feature_cache = {
    'user_features': None,
    'item_features': None,
    'last_update': None
}

async def update_features():
    """Update feature cache from S3"""
    try:
        # Download latest features
        for file in ['user_features.parquet', 'item_features.parquet']:
            s3_client.download_file(
                S3_BUCKET,
                f'processed/{file}',
                f'{FEATURE_PATH}/{file}'
            )
        
        # Load features
        feature_cache['user_features'] = pd.read_parquet(f'{FEATURE_PATH}/user_features.parquet')
        feature_cache['item_features'] = pd.read_parquet(f'{FEATURE_PATH}/item_features.parquet')
        feature_cache['last_update'] = datetime.now()
        
        logger.info("Features updated successfully")
    except Exception as e:
        FEATURE_UPDATE_ERRORS.inc()
        logger.error(f"Error updating features: {e}")
        raise

async def ensure_features_fresh(background_tasks: BackgroundTasks):
    """Ensure features are up to date"""
    if (feature_cache['last_update'] is None or
        datetime.now() - feature_cache['last_update'] > timedelta(seconds=FEATURE_UPDATE_INTERVAL)):
        background_tasks.add_task(update_features)

def load_models():
    """Load models from disk"""
    try:
        als_model.load_model(f'{MODEL_PATH}/als_model.npz')
        ranking_model.load_model(f'{MODEL_PATH}/ranking_model.cbm')
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    load_models()
    await update_features()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model_version = "1.0.0"  # Should be loaded from model metadata
        last_update = feature_cache['last_update'].isoformat() if feature_cache['last_update'] else None
        
        return {
            "status": "healthy",
            "model_version": model_version,
            "last_feature_update": last_update,
            "feature_freshness_seconds": (datetime.now() - feature_cache['last_update']).seconds if feature_cache['last_update'] else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest, background_tasks: BackgroundTasks):
    """Get recommendations for a user"""
    PREDICTION_REQUESTS.inc()
    
    with PREDICTION_LATENCY.time():
        try:
            # Ensure features are fresh
            await ensure_features_fresh(background_tasks)
            
            # Get user features
            if str(request.user_id) not in feature_cache['user_features'].index:
                raise HTTPException(status_code=404, detail="User not found")
            
            user_features = feature_cache['user_features'].loc[str(request.user_id)]
            
            # Get ALS recommendations
            als_candidates = als_model.recommend(
                str(request.user_id),
                n=min(request.n_items * 2, 100)  # Get more candidates for ranking
            )
            
            # Prepare ranking features
            ranking_features = []
            for item_id, _ in als_candidates:
                if str(item_id) in feature_cache['item_features'].index:
                    item_features = feature_cache['item_features'].loc[str(item_id)]
                    ranking_features.append({
                        'user_id': request.user_id,
                        'item_id': item_id,
                        **user_features.to_dict(),
                        **item_features.to_dict()
                    })
            
            ranking_df = pd.DataFrame(ranking_features)
            
            # Get ranking predictions with confidence
            predictions, (lower_bounds, upper_bounds), pred_metrics = ranking_model.predict(
                ranking_df.values,
                return_confidence=True,
                return_metrics=True
            )
            
            # Filter by confidence if requested
            confidence_width = upper_bounds - lower_bounds
            confident_mask = confidence_width <= (1 - request.min_confidence)
            
            # Sort by predicted score and get top N
            sorted_indices = np.argsort(-predictions)
            confident_indices = sorted_indices[confident_mask[sorted_indices]]
            top_indices = confident_indices[:request.n_items]
            
            # Prepare recommendations
            recommendations = []
            for idx in top_indices:
                item_id = ranking_df.iloc[idx]['item_id']
                metadata = None
                if request.include_metadata:
                    item_features = feature_cache['item_features'].loc[str(item_id)]
                    metadata = {
                        'category': item_features['category'],
                        'price': item_features['price'],
                        'popularity': item_features['popularity']
                    }
                
                recommendations.append(Recommendation(
                    item_id=str(item_id),
                    score=float(predictions[idx]),
                    confidence_lower=float(lower_bounds[idx]),
                    confidence_upper=float(upper_bounds[idx]),
                    metadata=metadata
                ))
            
            return RecommendationResponse(
                recommendations=recommendations,
                model_version="1.0.0",
                feature_timestamp=feature_cache['last_update'].isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/events")
async def record_event(event: EventRecord, background_tasks: BackgroundTasks):
    """Record a user event"""
    try:
        # Store event in S3
        event_data = {
            **event.dict(),
            'timestamp': datetime.fromisoformat(event.timestamp).timestamp()
        }
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=f'events/{event.timestamp[:10]}/{event.user_id}_{datetime.now().timestamp()}.json',
            Body=json.dumps(event_data)
        )
        
        # Trigger feature update if needed
        await ensure_features_fresh(background_tasks)
        
        return {"status": "success", "event_id": f"{event.user_id}_{event.timestamp}"}
    except Exception as e:
        logger.error(f"Error recording event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()
