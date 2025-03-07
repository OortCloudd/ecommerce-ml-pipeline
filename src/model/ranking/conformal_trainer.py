"""
Module for conformal prediction with CatBoost ranking model
"""
import numpy as np
from mapie.regression import MapieRegressor
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from .ranking_trainer import RankingTrainer
from ..metrics.business_metrics import calculate_business_metrics, generate_uncertainty_report

class ConformalRankingTrainer:
    def __init__(self, alpha=0.1, metric_thresholds=None):
        """
        Initialize conformal ranking trainer
        
        Args:
            alpha: Significance level (1 - confidence). Default 0.1 for 90% confidence
            metric_thresholds: Dictionary of thresholds for business metrics
        """
        self.base_trainer = RankingTrainer()
        self.conformal_model = None
        self.alpha = alpha
        self.metric_thresholds = metric_thresholds or {
            'uncertain_ctr_ratio': 0.3,
            'min_user_coverage': 0.7,
            'revenue_at_risk': 1000,
            'cold_start_uncertainty': 0.5
        }
        
    def train(self, X, y, group_ids, timestamps=None, purchase_values=None, validation_fraction=0.2):
        """
        Train both the base ranking model and the conformal predictor
        
        Args:
            X: Features matrix
            y: Target values (relevance scores)
            group_ids: Group IDs for queries (user_ids)
            timestamps: Optional timestamps for time-based validation
            purchase_values: Optional purchase values for revenue metrics
            validation_fraction: Fraction of data to use for validation
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        # First train the base model
        metrics = self.base_trainer.train(X, y, group_ids, timestamps, validation_fraction)
        
        # Now train the conformal predictor
        if timestamps is None:
            # Use random split if no timestamps provided
            train_size = int((1 - validation_fraction) * len(X))
            indices = np.arange(len(X))
            train_idx = indices[:train_size]
            valid_idx = indices[train_size:]
        else:
            # Use time-based split
            split_time = np.quantile(timestamps, 1 - validation_fraction)
            train_idx = timestamps <= split_time
            valid_idx = timestamps > split_time
            
        # Create conformal predictor
        self.conformal_model = MapieRegressor(
            estimator=self.base_trainer.model,
            method="naive",  # Simple method for time series
            cv="prefit"  # Use already fitted model
        )
        
        # Fit conformal predictor on validation set
        self.conformal_model.fit(X[valid_idx], y[valid_idx])
        
        # Get predictions with confidence intervals
        y_pred, y_pis = self.conformal_model.predict(
            X[valid_idx],
            alpha=self.alpha
        )
        
        # Calculate standard conformal metrics
        in_interval = np.logical_and(
            y[valid_idx] >= y_pis[:, 0, 0],
            y[valid_idx] <= y_pis[:, 1, 0]
        )
        
        metrics['conformal'] = {
            'coverage': np.mean(in_interval),
            'avg_interval_width': np.mean(y_pis[:, 1, 0] - y_pis[:, 0, 0]),
            'target_coverage': 1 - self.alpha
        }
        
        # Calculate business metrics
        if timestamps is not None:
            valid_timestamps = timestamps[valid_idx]
        else:
            valid_timestamps = np.arange(len(y[valid_idx]))
            
        if purchase_values is not None:
            valid_purchase_values = purchase_values[valid_idx]
        else:
            valid_purchase_values = None
            
        business_metrics = calculate_business_metrics(
            y_true=y[valid_idx],
            y_pred=y_pred,
            y_pred_lower=y_pis[:, 0, 0],
            y_pred_upper=y_pis[:, 1, 0],
            group_ids=group_ids[valid_idx],
            timestamps=valid_timestamps,
            purchase_values=valid_purchase_values,
            alpha=self.alpha
        )
        
        metrics['business'] = business_metrics
        
        # Generate uncertainty report
        report, alert = generate_uncertainty_report(
            business_metrics,
            self.metric_thresholds
        )
        metrics['uncertainty_report'] = {
            'text': report,
            'alert': alert
        }
        
        return metrics
    
    def predict(self, X, return_confidence=True, return_metrics=False):
        """
        Make predictions with uncertainty estimates
        
        Args:
            X: Features matrix
            return_confidence: If True, return prediction intervals
            return_metrics: If True, return prediction quality metrics
            
        Returns:
            If return_confidence:
                tuple: (predictions, (lower_bounds, upper_bounds), metrics if return_metrics)
            else:
                array: predictions
        """
        if return_confidence:
            y_pred, y_pis = self.conformal_model.predict(
                X,
                alpha=self.alpha
            )
            
            result = [y_pred, (y_pis[:, 0, 0], y_pis[:, 1, 0])]
            
            if return_metrics:
                # Calculate prediction quality metrics
                uncertainty = y_pis[:, 1, 0] - y_pis[:, 0, 0]
                metrics = {
                    'mean_uncertainty': np.mean(uncertainty),
                    'max_uncertainty': np.max(uncertainty),
                    'uncertain_predictions': np.mean(uncertainty > (1 - self.alpha)),
                    'prediction_range': {
                        'min': np.min(y_pred),
                        'max': np.max(y_pred),
                        'mean': np.mean(y_pred)
                    }
                }
                result.append(metrics)
            
            return tuple(result)
        else:
            return self.base_trainer.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance scores from base model"""
        return self.base_trainer.get_feature_importance()
    
    def save_model(self, path):
        """Save both base and conformal models"""
        self.base_trainer.save_model(path)
        # Save conformal calibration data
        if self.conformal_model is not None:
            np.save(
                path + '.conformal.npy',
                {
                    'alpha': self.alpha,
                    'residuals': self.conformal_model.conformity_scores_,
                    'metric_thresholds': self.metric_thresholds
                }
            )
    
    def load_model(self, path):
        """Load both base and conformal models"""
        self.base_trainer.load_model(path)
        # Load conformal calibration data
        try:
            conformal_data = np.load(path + '.conformal.npy', allow_pickle=True).item()
            self.alpha = conformal_data['alpha']
            self.metric_thresholds = conformal_data.get(
                'metric_thresholds',
                self.metric_thresholds
            )
            self.conformal_model = MapieRegressor(
                estimator=self.base_trainer.model,
                method="naive",
                cv="prefit"
            )
            self.conformal_model.conformity_scores_ = conformal_data['residuals']
        except FileNotFoundError:
            print("Warning: No conformal calibration data found")
        
        return self
