"""
Integration tests for prediction and monitoring services
"""
import pytest
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import boto3
from typing import Dict, List

# Service URLs
PREDICTION_URL = "http://localhost:8000"
MONITORING_URL = "http://localhost:8001"

# Test data paths (using RetailRocket dataset samples)
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")
S3_BUCKET = "ecommerce-ml-pipeline-data"
S3_PREFIX = "test"

@pytest.fixture
def s3_client():
    """Create S3 client"""
    return boto3.client(
        's3',
        region_name='eu-west-1'
    )

@pytest.fixture
def test_events():
    """Load test events data"""
    events_path = os.path.join(TEST_DATA_DIR, "events_sample.parquet")
    return pd.read_parquet(events_path)

@pytest.fixture
def test_items():
    """Load test items data"""
    items_path = os.path.join(TEST_DATA_DIR, "items_sample.parquet")
    return pd.read_parquet(items_path)

def test_prediction_service_health():
    """Test prediction service health endpoint"""
    response = requests.get(f"{PREDICTION_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data
    assert "last_feature_update" in data

def test_monitoring_service_health():
    """Test monitoring service health endpoint"""
    response = requests.get(f"{MONITORING_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_get_recommendations(test_events):
    """Test recommendation endpoint"""
    # Get a sample user
    user_id = test_events["visitorid"].iloc[0]
    
    # Request recommendations
    response = requests.post(
        f"{PREDICTION_URL}/recommendations",
        json={
            "user_id": str(user_id),
            "n_items": 10,
            "include_metadata": True
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 10
    
    # Check recommendation properties
    for rec in data["recommendations"]:
        assert "item_id" in rec
        assert "score" in rec
        assert "confidence_lower" in rec
        assert "confidence_upper" in rec
        
        # Validate confidence intervals
        assert rec["confidence_lower"] <= rec["score"] <= rec["confidence_upper"]

def test_record_event(test_events):
    """Test event recording endpoint"""
    # Create a sample event
    event = {
        "user_id": str(test_events["visitorid"].iloc[0]),
        "item_id": str(test_events["itemid"].iloc[0]),
        "event_type": "view",
        "timestamp": datetime.now().isoformat()
    }
    
    # Record event
    response = requests.post(
        f"{PREDICTION_URL}/events",
        json=event
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

def test_metrics_collection(test_events, test_items, s3_client):
    """Test metrics collection endpoint"""
    # Prepare test data
    predictions_df = pd.DataFrame({
        "user_id": test_events["visitorid"].head(100),
        "item_id": test_events["itemid"].head(100),
        "score": np.random.random(100),
        "confidence_lower": np.random.random(100) * 0.5,
        "confidence_upper": np.random.random(100) * 0.5 + 0.5
    })
    
    actuals_df = test_events.head(100)
    
    # Save test data to S3
    predictions_path = f"{S3_PREFIX}/test_predictions.parquet"
    actuals_path = f"{S3_PREFIX}/test_actuals.parquet"
    
    with pd.io.parquet.ParquetWriter(predictions_path, predictions_df.schema) as writer:
        writer.write_table(predictions_df)
    
    with pd.io.parquet.ParquetWriter(actuals_path, actuals_df.schema) as writer:
        writer.write_table(actuals_df)
    
    s3_client.upload_file(predictions_path, S3_BUCKET, predictions_path)
    s3_client.upload_file(actuals_path, S3_BUCKET, actuals_path)
    
    # Collect metrics
    response = requests.post(
        f"{MONITORING_URL}/metrics/collect",
        json={
            "predictions_file": f"s3://{S3_BUCKET}/{predictions_path}",
            "actuals_file": f"s3://{S3_BUCKET}/{actuals_path}",
            "model_version": "test-1.0.0"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate metrics structure
    assert "performance_metrics" in data
    assert "drift_metrics" in data
    assert "retraining_needed" in data
    
    # Clean up test data
    s3_client.delete_object(Bucket=S3_BUCKET, Key=predictions_path)
    s3_client.delete_object(Bucket=S3_BUCKET, Key=actuals_path)
    os.remove(predictions_path)
    os.remove(actuals_path)

def test_metrics_summary():
    """Test metrics summary endpoint"""
    response = requests.get(
        f"{MONITORING_URL}/metrics/summary",
        params={"days": 7}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate summary structure
    assert "time_range" in data
    assert "performance" in data
    assert "drift" in data
    assert "retraining_status" in data
    
    # Check performance metrics
    perf = data["performance"]
    assert "ndcg_mean" in perf
    assert "ndcg_std" in perf
    assert "ndcg_trend" in perf
    
    # Check drift metrics
    drift = data["drift"]
    assert "drift_frequency" in drift
    assert "last_drift_detected" in drift

def test_retraining_check():
    """Test retraining check endpoint"""
    response = requests.get(f"{MONITORING_URL}/metrics/retraining")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "retraining_needed" in data
    assert isinstance(data["retraining_needed"], bool)
    assert "timestamp" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
