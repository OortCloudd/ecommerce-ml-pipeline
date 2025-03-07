"""
FastAPI application for prediction service
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import boto3
import os
import json
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest
import random
import asyncio

from ..model.collaborative.als_trainer import ALSTrainer
from ..model.ranking.conformal_trainer import ConformalRankingTrainer
from ..processing.feature_engineering import AdvancedFeatureEngineer

# Check if metrics should be skipped
SKIP_METRICS = os.getenv("SKIP_METRICS", "false").lower() == "true"

if not SKIP_METRICS:
    try:
        from ..monitoring.metrics_collector import MetricsCollector
    except ImportError:
        logging.warning("Could not import MetricsCollector, metrics collection will be disabled")
        SKIP_METRICS = True

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
if not SKIP_METRICS:
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

# Function to retry storing events in S3
async def retry_store_event(event_data):
    """Retry storing an event in S3 after a failure"""
    try:
        # Wait a bit before retrying
        await asyncio.sleep(2)
        
        # Retry the S3 upload
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=f'events/retry/{event_data["user_id"]}_{datetime.now().timestamp()}.json',
            Body=json.dumps(event_data)
        )
        logger.info(f"Successfully retried storing event in S3: user_id={event_data['user_id']}")
    except Exception as e:
        logger.error(f"Failed to retry storing event in S3: {e}")

async def update_features():
    """Update feature cache from S3"""
    try:
        dev_mode = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
        feature_path = os.getenv("FEATURE_PATH", "/app/features")
        
        # Check if feature files exist
        user_features_path = f'{feature_path}/user_features.parquet'
        item_features_path = f'{feature_path}/item_features.parquet'
        
        if not os.path.exists(user_features_path) or not os.path.exists(item_features_path):
            if dev_mode:
                logger.warning("Feature files not found in development mode")
                # Try to download from S3 if AWS credentials are valid
                try:
                    # Download latest features
                    for file in ['user_features.parquet', 'item_features.parquet']:
                        s3_client.download_file(
                            S3_BUCKET,
                            f'processed/{file}',
                            f'{feature_path}/{file}'
                        )
                    logger.info("Features downloaded from S3 successfully")
                except Exception as e:
                    logger.warning(f"Could not download features from S3 in development mode: {e}")
                    # In development mode, create dummy feature files if they don't exist
                    if not os.path.exists(user_features_path):
                        logger.warning("Creating dummy user features in development mode")
                        # Create dummy user features
                        user_df = pd.DataFrame({
                            'user_id': [str(i) for i in range(100)],
                            'activity': np.random.random(100),
                            'recency': np.random.random(100)
                        })
                        user_df.set_index('user_id', inplace=True)
                        os.makedirs(os.path.dirname(user_features_path), exist_ok=True)
                        user_df.to_parquet(user_features_path)
                    
                    if not os.path.exists(item_features_path):
                        logger.warning("Creating dummy item features in development mode")
                        # Create dummy item features
                        item_df = pd.DataFrame({
                            'item_id': [str(i) for i in range(100)],
                            'category': np.random.choice(['electronics', 'clothing', 'home'], 100),
                            'price': np.random.random(100) * 100,
                            'popularity': np.random.random(100)
                        })
                        item_df.set_index('item_id', inplace=True)
                        os.makedirs(os.path.dirname(item_features_path), exist_ok=True)
                        item_df.to_parquet(item_features_path)
            else:
                # In production mode, try to download from S3
                # Download latest features
                for file in ['user_features.parquet', 'item_features.parquet']:
                    s3_client.download_file(
                        S3_BUCKET,
                        f'processed/{file}',
                        f'{feature_path}/{file}'
                    )
        
        # Load features if files exist
        if os.path.exists(user_features_path):
            feature_cache['user_features'] = pd.read_parquet(user_features_path)
        
        if os.path.exists(item_features_path):
            feature_cache['item_features'] = pd.read_parquet(item_features_path)
        
        feature_cache['last_update'] = datetime.now()
        
        logger.info("Features updated successfully")
    except Exception as e:
        FEATURE_UPDATE_ERRORS.inc()
        if os.getenv("DEVELOPMENT_MODE", "false").lower() == "true":
            logger.warning(f"Error updating features in development mode: {e}")
        else:
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
        model_path = os.getenv("MODEL_PATH", "/app/models")
        als_path = f'{model_path}/als_model.npz'
        ranking_path = f'{model_path}/ranking_model.cbm'
        
        # Check if we're in development mode and if model files exist
        dev_mode = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
        prod_ready = os.getenv("PRODUCTION_READY", "false").lower() == "true"
        allow_pickle = os.getenv("ALLOW_PICKLE", "false").lower() == "true"
        
        if dev_mode and (not os.path.exists(als_path) or not os.path.exists(ranking_path)):
            logger.warning("Running in development mode with missing model files. Using dummy models.")
            # Don't attempt to load models in dev mode if files don't exist
            return
            
        # Load ALS model with allow_pickle if specified
        try:
            if allow_pickle:
                # Monkey patch the load_model method to use allow_pickle=True
                original_load = np.load
                np.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'allow_pickle': True})
                als_model.load_model(als_path)
                np.load = original_load
            else:
                als_model.load_model(als_path)
        except Exception as e:
            logger.error(f"Error loading ALS model: {e}")
            if not dev_mode:
                raise
        
        # Load ranking model
        try:
            ranking_model.load_model(ranking_path)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ranking model: {e}")
            if not dev_mode:
                raise
    except Exception as e:
        if os.getenv("DEVELOPMENT_MODE", "false").lower() == "true":
            logger.warning(f"Error loading models in development mode: {e}")
        else:
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
            # Check if we're in development mode
            dev_mode = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
            prod_ready = os.getenv("PRODUCTION_READY", "false").lower() == "true"
            
            # Ensure features are fresh
            await ensure_features_fresh(background_tasks)
            
            # In development mode, if features or models aren't available, return dummy recommendations
            # But if PRODUCTION_READY is true, make the dummy recommendations more realistic
            if dev_mode and (feature_cache['user_features'] is None or feature_cache['item_features'] is None):
                if prod_ready:
                    logger.info("Using production-quality dummy recommendations in development mode")
                    # Generate more realistic recommendations with proper structure
                    recommendations = []
                    for i in range(min(request.n_items, 10)):
                        item_id = str(1000 + i)
                        base_score = 0.95 - (i * 0.05)
                        confidence = 0.1 + (i * 0.01)
                        recommendations.append(Recommendation(
                            item_id=item_id,
                            score=base_score,
                            confidence_lower=max(0, base_score - confidence),
                            confidence_upper=min(1.0, base_score + confidence),
                            metadata={
                                "category": random.choice(["electronics", "clothing", "home", "sports"]),
                                "price": round(random.uniform(10.0, 100.0), 2),
                                "popularity": round(random.uniform(0.5, 0.99), 2),
                                "in_stock": random.choice([True, True, True, False])
                            }
                        ))
                    
                    return RecommendationResponse(
                        recommendations=recommendations,
                        model_version="prod-ready-1.0.0",
                        feature_timestamp=datetime.now().isoformat()
                    )
                else:
                    logger.warning("Using basic dummy recommendations in development mode due to missing features")
                    recommendations = [
                        Recommendation(
                            item_id=str(i),
                            score=0.9 - (i * 0.1),
                            confidence_lower=0.7 - (i * 0.1),
                            confidence_upper=0.95 - (i * 0.05),
                            metadata={"category": "dummy", "price": 10.0, "popularity": 0.8}
                        ) for i in range(min(request.n_items, 10))
                    ]
                    
                    return RecommendationResponse(
                        recommendations=recommendations,
                        model_version="dev-dummy-1.0.0",
                        feature_timestamp=datetime.now().isoformat()
                    )
            
            # Get user features
            if str(request.user_id) not in feature_cache['user_features'].index:
                if dev_mode:
                    # In dev mode, use dummy user features
                    logger.warning(f"User {request.user_id} not found, using dummy user in development mode")
                    user_features = feature_cache['user_features'].iloc[0] if len(feature_cache['user_features']) > 0 else None
                else:
                    raise HTTPException(status_code=404, detail="User not found")
            else:
                user_features = feature_cache['user_features'].loc[str(request.user_id)]
            
            # In development mode, if models aren't loaded, return dummy recommendations
            if dev_mode and (not hasattr(als_model, 'model') or als_model.model is None):
                logger.warning("Using dummy recommendations in development mode due to missing models")
                recommendations = [
                    Recommendation(
                        item_id=str(i),
                        score=0.9 - (i * 0.1),
                        confidence_lower=0.7 - (i * 0.1),
                        confidence_upper=0.95 - (i * 0.05),
                        metadata={"category": "dummy", "price": 10.0, "popularity": 0.8} if request.include_metadata else None
                    ) for i in range(min(request.n_items, 10))
                ]
                
                return RecommendationResponse(
                    recommendations=recommendations,
                    model_version="dev-dummy-1.0.0",
                    feature_timestamp=datetime.now().isoformat() if feature_cache['last_update'] is None else feature_cache['last_update'].isoformat()
                )
            
            # Get ALS recommendations
            try:
                als_candidates = als_model.recommend(
                    str(request.user_id),
                    n=min(request.n_items * 2, 100)  # Get more candidates for ranking
                )
            except Exception as e:
                if dev_mode:
                    logger.warning(f"Error getting ALS recommendations in development mode: {e}")
                    # Use random item IDs as candidates
                    item_ids = list(feature_cache['item_features'].index) if feature_cache['item_features'] is not None else [str(i) for i in range(100)]
                    import random
                    random.shuffle(item_ids)
                    als_candidates = [(item_id, 0.5) for item_id in item_ids[:min(request.n_items * 2, 100)]]
                else:
                    raise
            
            # Prepare ranking features
            ranking_features = []
            for item_id, _ in als_candidates:
                if feature_cache['item_features'] is not None and str(item_id) in feature_cache['item_features'].index:
                    item_features = feature_cache['item_features'].loc[str(item_id)]
                    ranking_features.append({
                        'user_id': request.user_id,
                        'item_id': item_id,
                        **user_features.to_dict(),
                        **item_features.to_dict()
                    })
            
            ranking_df = pd.DataFrame(ranking_features) if ranking_features else None
            
            # Get ranking predictions with confidence
            try:
                if ranking_df is not None and len(ranking_df) > 0 and hasattr(ranking_model, 'model') and ranking_model.model is not None:
                    predictions, (lower_bounds, upper_bounds), pred_metrics = ranking_model.predict(
                        ranking_df.values,
                        return_confidence=True,
                        return_metrics=True
                    )
                else:
                    if dev_mode:
                        # Generate dummy predictions in development mode
                        logger.warning("Using dummy ranking predictions in development mode")
                        n_items = len(ranking_features) if ranking_features else min(request.n_items, 10)
                        predictions = np.array([0.9 - (i * 0.05) for i in range(n_items)])
                        lower_bounds = np.array([0.7 - (i * 0.05) for i in range(n_items)])
                        upper_bounds = np.array([0.95 - (i * 0.02) for i in range(n_items)])
                        pred_metrics = {"dummy": True}
                    else:
                        raise ValueError("Ranking model not loaded or no valid ranking features")
            except Exception as e:
                if dev_mode:
                    # Generate dummy predictions in development mode
                    logger.warning(f"Error in ranking prediction in development mode: {e}")
                    n_items = len(ranking_features) if ranking_features else min(request.n_items, 10)
                    predictions = np.array([0.9 - (i * 0.05) for i in range(n_items)])
                    lower_bounds = np.array([0.7 - (i * 0.05) for i in range(n_items)])
                    upper_bounds = np.array([0.95 - (i * 0.02) for i in range(n_items)])
                    pred_metrics = {"dummy": True}
                else:
                    raise
            
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
                if ranking_df is not None:
                    item_id = ranking_df.iloc[idx]['item_id']
                else:
                    # In dev mode with no ranking_df
                    item_id = str(idx)
                    
                metadata = None
                if request.include_metadata:
                    if feature_cache['item_features'] is not None and str(item_id) in feature_cache['item_features'].index:
                        item_features = feature_cache['item_features'].loc[str(item_id)]
                        metadata = {
                            'category': item_features['category'],
                            'price': item_features['price'],
                            'popularity': item_features['popularity']
                        }
                    else:
                        # Dummy metadata in dev mode
                        metadata = {
                            'category': 'dummy',
                            'price': 10.0,
                            'popularity': 0.8
                        }
                
                recommendations.append(Recommendation(
                    item_id=str(item_id),
                    score=float(predictions[idx]),
                    confidence_lower=float(lower_bounds[idx]),
                    confidence_upper=float(upper_bounds[idx]),
                    metadata=metadata
                ))
            
            # Record metrics if enabled
            if not SKIP_METRICS:
                try:
                    metrics_collector.record_recommendation_metrics(
                        user_id=request.user_id,
                        recommendations=[r.item_id for r in recommendations],
                        scores=[r.score for r in recommendations],
                        confidence_bounds=[(r.confidence_lower, r.confidence_upper) for r in recommendations],
                        prediction_metrics=pred_metrics
                    )
                except Exception as e:
                    logger.warning(f"Failed to record metrics: {e}")
            
            return RecommendationResponse(
                recommendations=recommendations,
                model_version="1.0.0" if not dev_mode else "dev-1.0.0",
                feature_timestamp=feature_cache['last_update'].isoformat() if feature_cache['last_update'] else datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/events")
async def record_event(event: EventRecord, background_tasks: BackgroundTasks):
    """Record a user event"""
    try:
        dev_mode = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
        prod_ready = os.getenv("PRODUCTION_READY", "false").lower() == "true"
        
        # Store event in S3
        event_data = {
            **event.dict(),
            'timestamp': datetime.fromisoformat(event.timestamp).timestamp()
        }
        
        try:
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=f'events/{event.timestamp[:10]}/{event.user_id}_{datetime.now().timestamp()}.json',
                Body=json.dumps(event_data)
            )
            if prod_ready:
                logger.info(f"Successfully stored event in S3: user_id={event.user_id}, event_type={event.event_type}")
        except Exception as e:
            if dev_mode:
                if prod_ready:
                    # In production-ready mode, we want to be more informative about errors
                    logger.error(f"Failed to store event in S3 but continuing in development mode: {e}")
                    # Schedule a retry in the background
                    background_tasks.add_task(retry_store_event, event_data)
                else:
                    logger.warning(f"Failed to store event in S3 in development mode: {e}")
            else:
                raise
        
        # Record metrics if enabled
        if not SKIP_METRICS:
            try:
                metrics_collector.record_event_metrics(
                    user_id=event.user_id,
                    item_id=event.item_id,
                    event_type=event.event_type,
                    timestamp=event.timestamp
                )
            except Exception as e:
                logger.warning(f"Failed to record event metrics: {e}")
        
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
