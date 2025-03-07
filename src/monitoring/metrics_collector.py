"""
Metrics collector for model monitoring
"""
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from scipy import stats

from ..model.metrics.business_metrics import BusinessMetricsCalculator
from ..processing.data_processor import DataProcessor

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(
        self,
        metrics_dir: str,
        reference_data_path: Optional[str] = None,
        monitoring_window_days: int = 7
    ):
        """
        Initialize metrics collector
        
        Args:
            metrics_dir: Directory to store metrics
            reference_data_path: Path to reference data for drift detection
            monitoring_window_days: Days of data to use for monitoring
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.reference_data_path = reference_data_path
        self.monitoring_window_days = monitoring_window_days
        
        self.business_metrics = BusinessMetricsCalculator()
        self.data_processor = DataProcessor()
        
        # Load reference data if available
        self.reference_distributions = {}
        if reference_data_path:
            self._load_reference_distributions()
            
    def _load_reference_distributions(self):
        """Load reference data distributions for drift detection"""
        try:
            reference_data = pd.read_parquet(self.reference_data_path)
            
            # Calculate distributions for numerical features
            numeric_cols = reference_data.select_dtypes(
                include=['int64', 'float64']
            ).columns
            
            for col in numeric_cols:
                self.reference_distributions[col] = {
                    'mean': reference_data[col].mean(),
                    'std': reference_data[col].std(),
                    'quantiles': reference_data[col].quantile([0.25, 0.5, 0.75])
                }
                
            logger.info("Loaded reference distributions successfully")
            
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            
    def collect_performance_metrics(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        model_version: str
    ) -> Dict:
        """
        Collect model performance metrics
        
        Args:
            predictions: Predicted recommendations
            actuals: Actual user interactions
            model_version: Model version
            
        Returns:
            Dict containing performance metrics
        """
        try:
            # Calculate ranking metrics
            ndcg = self.business_metrics.calculate_ndcg(
                predictions,
                actuals
            )
            
            # Calculate business metrics
            business_metrics = self.business_metrics.calculate_all_metrics(
                predictions,
                actuals
            )
            
            # Calculate uncertainty metrics
            if 'confidence_lower' in predictions.columns:
                uncertainty_metrics = {
                    'avg_confidence_interval': (
                        predictions['confidence_upper'] -
                        predictions['confidence_lower']
                    ).mean(),
                    'coverage': (
                        (actuals['score'] >= predictions['confidence_lower']) &
                        (actuals['score'] <= predictions['confidence_upper'])
                    ).mean()
                }
            else:
                uncertainty_metrics = {}
                
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'model_version': model_version,
                'ndcg': ndcg,
                'business_metrics': business_metrics,
                'uncertainty_metrics': uncertainty_metrics
            }
            
            # Save metrics
            self._save_metrics('performance', metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            raise
            
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        feature_columns: List[str]
    ) -> Dict:
        """
        Detect data drift in features
        
        Args:
            current_data: Current data to check for drift
            feature_columns: Columns to check for drift
            
        Returns:
            Dict containing drift metrics
        """
        try:
            drift_metrics = {}
            
            for col in feature_columns:
                if col not in self.reference_distributions:
                    continue
                    
                ref_dist = self.reference_distributions[col]
                current_dist = {
                    'mean': current_data[col].mean(),
                    'std': current_data[col].std(),
                    'quantiles': current_data[col].quantile([0.25, 0.5, 0.75])
                }
                
                # Kolmogorov-Smirnov test
                ks_statistic, p_value = stats.ks_2samp(
                    current_data[col].dropna(),
                    pd.Series(ref_dist['mean'], index=range(100))  # Reference sample
                )
                
                # Calculate drift metrics
                drift_metrics[col] = {
                    'ks_statistic': float(ks_statistic),
                    'p_value': float(p_value),
                    'mean_difference': float(
                        abs(current_dist['mean'] - ref_dist['mean'])
                    ),
                    'std_difference': float(
                        abs(current_dist['std'] - ref_dist['std'])
                    )
                }
                
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'drift_metrics': drift_metrics,
                'drift_detected': any(
                    m['p_value'] < 0.05 for m in drift_metrics.values()
                )
            }
            
            # Save metrics
            self._save_metrics('drift', metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {e}")
            raise
            
    def check_retraining_needed(
        self,
        performance_threshold: float = 0.1,
        drift_threshold: float = 0.05
    ) -> bool:
        """
        Check if model retraining is needed
        
        Args:
            performance_threshold: Maximum allowed performance degradation
            drift_threshold: Maximum allowed drift p-value
            
        Returns:
            Boolean indicating if retraining is needed
        """
        try:
            # Load recent metrics
            performance_metrics = self._load_recent_metrics('performance')
            drift_metrics = self._load_recent_metrics('drift')
            
            if not performance_metrics or not drift_metrics:
                return False
                
            # Check performance degradation
            baseline_ndcg = performance_metrics[0]['ndcg']
            current_ndcg = performance_metrics[-1]['ndcg']
            performance_degradation = (baseline_ndcg - current_ndcg) / baseline_ndcg
            
            # Check drift
            latest_drift = drift_metrics[-1]
            significant_drift = latest_drift['drift_detected']
            
            # Decision logic
            retraining_needed = (
                performance_degradation > performance_threshold or
                significant_drift
            )
            
            if retraining_needed:
                logger.info(
                    "Retraining needed: "
                    f"performance_degradation={performance_degradation:.2f}, "
                    f"drift_detected={significant_drift}"
                )
                
            return retraining_needed
            
        except Exception as e:
            logger.error(f"Error checking if retraining needed: {e}")
            raise
            
    def _save_metrics(self, metric_type: str, metrics: Dict):
        """Save metrics to file"""
        try:
            metrics_file = self.metrics_dir / f"{metric_type}_metrics.jsonl"
            
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            
    def _load_recent_metrics(
        self,
        metric_type: str,
        days: Optional[int] = None
    ) -> List[Dict]:
        """Load recent metrics from file"""
        try:
            metrics_file = self.metrics_dir / f"{metric_type}_metrics.jsonl"
            if not metrics_file.exists():
                return []
                
            days = days or self.monitoring_window_days
            cutoff_time = datetime.now() - timedelta(days=days)
            
            metrics = []
            with open(metrics_file, 'r') as f:
                for line in f:
                    metric = json.loads(line.strip())
                    metric_time = datetime.fromisoformat(metric['timestamp'])
                    
                    if metric_time >= cutoff_time:
                        metrics.append(metric)
                        
            return metrics
            
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            return []
