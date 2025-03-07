"""
FastAPI service for real-time recommendations
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import json
from pathlib import Path

from ..model.collaborative.als_trainer import ALSTrainer
from ..model.ranking.conformal_trainer import ConformalRankingTrainer
from ..processing.feature_engineering import AdvancedFeatureEngineer
from ..processing.data_processor import DataProcessor

logger = logging.getLogger(__name__)

class UserEvent(BaseModel):
    """User interaction event"""
    visitor_id: int = Field(..., description="Unique visitor ID")
    item_id: int = Field(..., description="Item ID that was interacted with")
    event_type: str = Field(..., description="Type of event (view, addtocart, transaction)")
    timestamp: Optional[float] = Field(None, description="Event timestamp in seconds since epoch")
    additional_features: Optional[Dict] = Field(None, description="Additional event features")

    class Config:
        json_schema_extra = {
            "example": {
                "visitor_id": 12345,
                "item_id": 67890,
                "event_type": "view",
                "timestamp": 1646646000.0,
                "additional_features": {
                    "session_id": "abc123",
                    "referrer": "search"
                }
            }
        }

class RecommendationRequest(BaseModel):
    """Request for recommendations"""
    visitor_id: int = Field(..., description="Visitor ID to get recommendations for")
    n_recommendations: int = Field(10, description="Number of recommendations to return")
    include_confidence: bool = Field(True, description="Whether to include confidence intervals")
    context: Optional[Dict] = Field(None, description="Additional context for recommendations")

    class Config:
        json_schema_extra = {
            "example": {
                "visitor_id": 12345,
                "n_recommendations": 5,
                "include_confidence": True,
                "context": {
                    "time_of_day": 0.5,
                    "day_of_week": 0.7,
                    "device_type": 1
                }
            }
        }

class ItemScore(BaseModel):
    """Item score with confidence interval"""
    item_id: int = Field(..., description="Item ID")
    score: float = Field(..., description="Recommendation score")
    confidence_lower: Optional[float] = Field(None, description="Lower bound of confidence interval")
    confidence_upper: Optional[float] = Field(None, description="Upper bound of confidence interval")
    category_id: Optional[int] = Field(None, description="Category ID of the item")
    price: Optional[float] = Field(None, description="Price of the item")

class RecommendationResponse(BaseModel):
    """Response containing recommendations"""
    items: List[ItemScore] = Field(..., description="List of recommended items with scores")
    model_version: str = Field(..., description="Version of the model used")
    prediction_time: float = Field(..., description="Time taken to generate recommendations")
    additional_info: Optional[Dict] = Field(None, description="Additional response information")

class PredictionService:
    def __init__(
        self,
        model_dir: str,
        cache_dir: str,
        feature_update_interval: int = 3600  # 1 hour
    ):
        """
        Initialize prediction service
        
        Args:
            model_dir: Directory containing trained models
            cache_dir: Directory for feature cache
            feature_update_interval: Seconds between feature updates
        """
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.feature_update_interval = feature_update_interval
        
        # Initialize models
        self.als_model = ALSTrainer()
        self.ranking_model = ConformalRankingTrainer()
        
        # Load latest models
        self._load_latest_models()
        
        # Initialize feature engineering
        self.feature_engineer = AdvancedFeatureEngineer()
        self.data_processor = DataProcessor()
        
        # Initialize feature cache
        self.user_features_cache = {}
        self.item_features_cache = {}
        self.last_feature_update = datetime.now()
        
        # Load initial features
        self._load_feature_cache()
        
    def _load_latest_models(self):
        """Load the latest versions of ALS and ranking models"""
        try:
            # Find latest model versions
            als_models = list(self.model_dir.glob('als_model_v*.pkl'))
            ranking_models = list(self.model_dir.glob('ranking_model_v*.pkl'))
            
            if not als_models or not ranking_models:
                raise ValueError("No models found in model directory")
            
            # Load latest versions
            latest_als = max(als_models, key=lambda x: x.stat().st_mtime)
            latest_ranking = max(ranking_models, key=lambda x: x.stat().st_mtime)
            
            self.als_model.load_model(str(latest_als))
            self.ranking_model.load_model(str(latest_ranking))
            
            self.model_version = f"als_{latest_als.stem}_ranking_{latest_ranking.stem}"
            logger.info(f"Loaded models version: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
            
    def _load_feature_cache(self):
        """Load feature cache from disk"""
        try:
            user_cache_file = self.cache_dir / 'user_features_cache.parquet'
            item_cache_file = self.cache_dir / 'item_features_cache.parquet'
            
            if user_cache_file.exists():
                self.user_features_cache = pd.read_parquet(user_cache_file)
            if item_cache_file.exists():
                self.item_features_cache = pd.read_parquet(item_cache_file)
                
            logger.info("Loaded feature cache successfully")
            
        except Exception as e:
            logger.warning(f"Could not load feature cache: {e}")
            
    def _save_feature_cache(self):
        """Save feature cache to disk"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            if self.user_features_cache:
                pd.DataFrame(self.user_features_cache).to_parquet(
                    self.cache_dir / 'user_features_cache.parquet'
                )
            if self.item_features_cache:
                pd.DataFrame(self.item_features_cache).to_parquet(
                    self.cache_dir / 'item_features_cache.parquet'
                )
                
            logger.info("Saved feature cache successfully")
            
        except Exception as e:
            logger.error(f"Error saving feature cache: {e}")
            
    def _update_features(self, background_tasks: BackgroundTasks):
        """Update feature cache if needed"""
        now = datetime.now()
        if (now - self.last_feature_update).total_seconds() > self.feature_update_interval:
            background_tasks.add_task(self._async_update_features)
            
    async def _async_update_features(self):
        """Asynchronously update features"""
        try:
            # Load recent events from database/cache
            recent_events = self._load_recent_events()
            
            # Update features
            if recent_events is not None:
                user_features = self.feature_engineer.engineer_features(
                    recent_events,
                    self.item_features_cache,
                    None  # category data
                )
                
                # Update cache
                self.user_features_cache.update(user_features['user_features'])
                self.item_features_cache.update(user_features['item_features'])
                
                # Save cache
                self._save_feature_cache()
                
            self.last_feature_update = datetime.now()
            logger.info("Updated features successfully")
            
        except Exception as e:
            logger.error(f"Error updating features: {e}")
            
    def _load_recent_events(self):
        """Load recent events for feature updates"""
        # TODO: Implement loading recent events from database/cache
        return None
        
    def get_recommendations(
        self,
        request: RecommendationRequest,
        background_tasks: BackgroundTasks
    ) -> RecommendationResponse:
        """
        Get recommendations for a user
        
        Args:
            request: Recommendation request
            background_tasks: FastAPI background tasks
            
        Returns:
            RecommendationResponse with recommendations
        """
        try:
            start_time = datetime.now()
            
            # Update features if needed
            self._update_features(background_tasks)
            
            # Get ALS recommendations
            als_scores = self.als_model.predict(
                request.visitor_id,
                n_items=request.n_recommendations * 2  # Get more for reranking
            )
            
            # Prepare features for ranking
            candidate_items = [score[0] for score in als_scores]
            features = self._prepare_ranking_features(
                request.visitor_id,
                candidate_items,
                request.context
            )
            
            # Get ranking scores with confidence
            if request.include_confidence:
                scores, (lower, upper) = self.ranking_model.predict(
                    features,
                    return_confidence=True
                )
            else:
                scores = self.ranking_model.predict(
                    features,
                    return_confidence=False
                )
                lower = upper = None
            
            # Sort items by score
            items = []
            for i, item_id in enumerate(candidate_items):
                item = ItemScore(
                    item_id=item_id,
                    score=float(scores[i])
                )
                
                if request.include_confidence:
                    item.confidence_lower = float(lower[i])
                    item.confidence_upper = float(upper[i])
                    
                if item_id in self.item_features_cache:
                    item_features = self.item_features_cache[item_id]
                    item.category_id = item_features.get('categoryid')
                    item.price = item_features.get('price')
                    
                items.append(item)
                
            # Sort by score and take top N
            items.sort(key=lambda x: x.score, reverse=True)
            items = items[:request.n_recommendations]
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            return RecommendationResponse(
                items=items,
                model_version=self.model_version,
                prediction_time=prediction_time,
                additional_info={
                    'feature_freshness': (datetime.now() - self.last_feature_update).total_seconds()
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    def _prepare_ranking_features(
        self,
        visitor_id: int,
        item_ids: List[int],
        context: Optional[Dict] = None
    ) -> np.ndarray:
        """Prepare features for ranking model"""
        features = []
        
        # Get user features
        user_features = self.user_features_cache.get(visitor_id, {})
        
        for item_id in item_ids:
            # Get item features
            item_features = self.item_features_cache.get(item_id, {})
            
            # Combine features
            feature_vector = []
            
            # Add user features
            for feature in ['total_events', 'total_purchases', 'activity_duration_days']:
                feature_vector.append(user_features.get(feature, 0))
                
            # Add item features
            for feature in ['total_interactions', 'total_purchases', 'popularity_score']:
                feature_vector.append(item_features.get(feature, 0))
                
            # Add context features if available
            if context:
                feature_vector.extend([
                    context.get('time_of_day', 0),
                    context.get('day_of_week', 0),
                    context.get('device_type', 0)
                ])
                
            features.append(feature_vector)
            
        return np.array(features)
        
    def update_event(
        self,
        event: UserEvent,
        background_tasks: BackgroundTasks
    ):
        """
        Update user event and trigger feature recalculation
        
        Args:
            event: New user event
            background_tasks: FastAPI background tasks
        """
        try:
            # Store event
            self._store_event(event)
            
            # Trigger feature update
            background_tasks.add_task(self._async_update_features)
            
        except Exception as e:
            logger.error(f"Error updating event: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    def _store_event(self, event: UserEvent):
        """Store user event"""
        # TODO: Implement event storage in database/cache
        pass
