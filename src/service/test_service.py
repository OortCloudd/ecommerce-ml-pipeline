"""
Test script for prediction service
"""
import requests
import json
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_health():
    """Test health check endpoint"""
    response = requests.get('http://localhost:8000/health')
    logger.info(f"Health check response: {response.json()}")
    assert response.status_code == 200

def test_recommendations():
    """Test recommendations endpoint"""
    request = {
        "visitor_id": 12345,
        "n_recommendations": 5,
        "include_confidence": True,
        "context": {
            "time_of_day": datetime.now().hour / 24.0,
            "day_of_week": datetime.now().weekday() / 7.0,
            "device_type": 1  # desktop
        }
    }
    
    response = requests.post(
        'http://localhost:8000/recommendations',
        json=request
    )
    
    logger.info(f"Recommendations response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200

def test_event_ingestion():
    """Test event ingestion endpoint"""
    event = {
        "visitor_id": 12345,
        "item_id": 67890,
        "event_type": "view",
        "timestamp": time.time(),
        "additional_features": {
            "session_id": "abc123",
            "referrer": "search"
        }
    }
    
    response = requests.post(
        'http://localhost:8000/events',
        json=event
    )
    
    logger.info(f"Event ingestion response: {response.json()}")
    assert response.status_code == 200

def test_metrics():
    """Test metrics endpoint"""
    response = requests.get('http://localhost:8000/metrics')
    logger.info(f"Metrics response: {response.json()}")
    assert response.status_code == 200

if __name__ == "__main__":
    # Wait for service to start
    time.sleep(5)
    
    try:
        test_health()
        test_recommendations()
        test_event_ingestion()
        test_metrics()
        logger.info("All tests passed!")
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        raise
