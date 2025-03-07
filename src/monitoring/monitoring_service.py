"""
Model monitoring service for e-commerce recommendations
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import boto3
from pathlib import Path
import json

from .metrics_collector import MetricsCollector
from ..processing.data_processor import DataProcessor

logger = logging.getLogger(__name__)

class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    performance_threshold: float = Field(
        0.1,
        description="Maximum allowed performance degradation"
    )
    drift_threshold: float = Field(
        0.05,
        description="Maximum allowed drift p-value"
    )
    monitoring_window_days: int = Field(
        7,
        description="Days of data to use for monitoring"
    )
    s3_bucket: str = Field(
        ...,
        description="S3 bucket for storing metrics"
    )
    s3_prefix: str = Field(
        "monitoring",
        description="S3 prefix for metrics storage"
    )

class MonitoringService:
    def __init__(
        self,
        config: MonitoringConfig,
        metrics_dir: str = "monitoring_metrics"
    ):
        """
        Initialize monitoring service
        
        Args:
            config: Monitoring configuration
            metrics_dir: Directory to store metrics
        """
        self.config = config
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize AWS clients
        self.s3 = boto3.client('s3')
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            metrics_dir=str(self.metrics_dir),
            monitoring_window_days=config.monitoring_window_days
        )
        self.data_processor = DataProcessor()
        
    async def collect_metrics(
        self,
        predictions_df: pd.DataFrame,
        actuals_df: pd.DataFrame,
        model_version: str,
        background_tasks: BackgroundTasks
    ):
        """
        Collect and store metrics
        
        Args:
            predictions_df: Model predictions
            actuals_df: Actual interactions
            model_version: Model version
            background_tasks: FastAPI background tasks
        """
        try:
            # Collect performance metrics
            performance_metrics = self.metrics_collector.collect_performance_metrics(
                predictions_df,
                actuals_df,
                model_version
            )
            
            # Detect data drift
            current_features = self.data_processor.engineer_user_features(actuals_df)
            drift_metrics = self.metrics_collector.detect_data_drift(
                current_features,
                feature_columns=current_features.columns.tolist()
            )
            
            # Check if retraining needed
            retraining_needed = self.metrics_collector.check_retraining_needed(
                performance_threshold=self.config.performance_threshold,
                drift_threshold=self.config.drift_threshold
            )
            
            # Upload metrics to S3 in background
            background_tasks.add_task(
                self._upload_metrics_to_s3,
                performance_metrics,
                drift_metrics,
                retraining_needed
            )
            
            return {
                "performance_metrics": performance_metrics,
                "drift_metrics": drift_metrics,
                "retraining_needed": retraining_needed
            }
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to collect metrics: {str(e)}"
            )
            
    def _upload_metrics_to_s3(
        self,
        performance_metrics: Dict,
        drift_metrics: Dict,
        retraining_needed: bool
    ):
        """Upload metrics to S3"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics = {
                "timestamp": timestamp,
                "performance_metrics": performance_metrics,
                "drift_metrics": drift_metrics,
                "retraining_needed": retraining_needed
            }
            
            # Upload to S3
            key = f"{self.config.s3_prefix}/metrics_{timestamp}.json"
            self.s3.put_object(
                Bucket=self.config.s3_bucket,
                Key=key,
                Body=json.dumps(metrics)
            )
            
            logger.info(f"Uploaded metrics to S3: {key}")
            
        except Exception as e:
            logger.error(f"Error uploading metrics to S3: {e}")
            
    def get_metrics_summary(
        self,
        days: Optional[int] = None
    ) -> Dict:
        """
        Get summary of recent metrics
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Dict containing metrics summary
        """
        try:
            days = days or self.config.monitoring_window_days
            
            # Load recent metrics
            performance_metrics = self.metrics_collector._load_recent_metrics(
                'performance',
                days=days
            )
            drift_metrics = self.metrics_collector._load_recent_metrics(
                'drift',
                days=days
            )
            
            if not performance_metrics or not drift_metrics:
                return {
                    "error": "No metrics available for the specified time range"
                }
                
            # Calculate summary statistics
            ndcg_values = [m['ndcg'] for m in performance_metrics]
            drift_detected = [m['drift_detected'] for m in drift_metrics]
            
            summary = {
                "time_range": f"Last {days} days",
                "performance": {
                    "ndcg_mean": np.mean(ndcg_values),
                    "ndcg_std": np.std(ndcg_values),
                    "ndcg_trend": np.polyfit(
                        range(len(ndcg_values)),
                        ndcg_values,
                        1
                    )[0]
                },
                "drift": {
                    "drift_frequency": sum(drift_detected) / len(drift_detected),
                    "last_drift_detected": drift_metrics[-1]['drift_detected']
                },
                "retraining_status": {
                    "needed": self.metrics_collector.check_retraining_needed(
                        self.config.performance_threshold,
                        self.config.drift_threshold
                    )
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get metrics summary: {str(e)}"
            )
