"""
FastAPI application for prediction service
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import boto3
import os
import json
from pathlib import Path

app = FastAPI(title="E-commerce Recommendation Service")

# Models
class RecommendationRequest(BaseModel):
    user_id: str
    n_items: int = 10
    include_metadata: bool = False

class EventRecord(BaseModel):
    user_id: str
    item_id: str
    event_type: str
    timestamp: str

class Recommendation(BaseModel):
    item_id: str
    score: float
    confidence_lower: float
    confidence_upper: float
    metadata: Optional[Dict] = None

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]
    model_version: str

# Global variables
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")
FEATURE_PATH = os.getenv("FEATURE_PATH", "/app/features")
S3_BUCKET = os.getenv("S3_BUCKET", "ecommerce-ml-pipeline-data")

# Initialize S3 client
s3_client = boto3.client('s3', region_name=os.getenv("AWS_DEFAULT_REGION", "eu-west-1"))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model_version = "1.0.0"  # This should be loaded from model metadata
        last_update = datetime.now().isoformat()
        
        return {
            "status": "healthy",
            "model_version": model_version,
            "last_feature_update": last_update
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations for a user"""
    try:
        # This is a mock implementation
        # In production, this would use the actual model
        n_items = min(request.n_items, 50)  # Cap at 50 items
        
        # Generate mock recommendations
        recommendations = []
        for i in range(n_items):
            score = np.random.random()
            confidence_width = np.random.random() * 0.2
            recommendation = Recommendation(
                item_id=f"item_{1000 + i}",
                score=score,
                confidence_lower=max(0, score - confidence_width),
                confidence_upper=min(1, score + confidence_width),
                metadata={
                    "category": f"category_{i % 10}",
                    "price": round(np.random.uniform(10, 1000), 2)
                } if request.include_metadata else None
            )
            recommendations.append(recommendation)
        
        # Sort by score descending
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return RecommendationResponse(
            recommendations=recommendations,
            model_version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/events")
async def record_event(event: EventRecord):
    """Record a user event"""
    try:
        # In production, this would store the event
        # For now, we just validate and acknowledge
        return {"status": "success", "event_id": "mock_id"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    # This would be replaced with proper Prometheus metrics
    return "# HELP model_predictions_total Total number of predictions\n" + \
           "# TYPE model_predictions_total counter\n" + \
           "model_predictions_total 100\n"
