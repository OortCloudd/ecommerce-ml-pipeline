"""
Module for training and evaluating the CatBoost ranking model
"""
import numpy as np
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

class RankingTrainer:
    def __init__(self):
        """Initialize CatBoost ranking trainer"""
        self.model = None
        self.best_params = None
        self.feature_names = None
        
    def _create_pool(self, X, y, group_ids, timestamps=None):
        """Create CatBoost Pool with optional timestamps"""
        return Pool(
            data=X,
            label=y,
            group_id=group_ids,
            timestamp=timestamps
        )
        
    def tune_hyperparameters(self, X, y, group_ids, timestamps, n_trials=100):
        """
        Tune hyperparameters using CatBoost's built-in grid search
        
        Args:
            X: Features matrix
            y: Target values (relevance scores)
            group_ids: Group IDs for queries (user_ids)
            timestamps: Timestamps for time-based splitting
            n_trials: Approximate number of trials to run
        """
        # Create time-based splits
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Sort data by timestamp
        sort_idx = np.argsort(timestamps)
        X = X[sort_idx]
        y = y[sort_idx]
        group_ids = group_ids[sort_idx]
        timestamps = timestamps[sort_idx]
        
        # Get the last split for final evaluation
        splits = list(tscv.split(X))
        train_idx, valid_idx = splits[-1]
        
        # Create train and validation pools
        train_pool = self._create_pool(
            X[train_idx], y[train_idx],
            group_ids[train_idx], timestamps[train_idx]
        )
        valid_pool = self._create_pool(
            X[valid_idx], y[valid_idx],
            group_ids[valid_idx], timestamps[valid_idx]
        )
        
        # Define grid
        grid = {
            'learning_rate': [0.01, 0.03, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5, 7],
            'random_strength': [1, 3, 5],
            'bagging_temperature': [0.0, 0.5, 1.0]
        }
        
        # Initialize model with base parameters
        model = CatBoostRanker(
            iterations=1000,  # Will be limited by early stopping
            loss_function='YetiRank',
            eval_metric='NDCG',
            random_seed=42,
            verbose=False
        )
        
        # Run grid search
        model.grid_search(
            grid,
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=50,
            verbose=False
        )
        
        self.best_params = model.get_params()
        print("Best parameters:", self.best_params)
        print("Best NDCG:", model.get_best_score()['validation']['NDCG'])
        
        return self.best_params
        
    def train(self, X, y, group_ids, timestamps=None, validation_fraction=0.2):
        """
        Train the ranking model with time-based validation
        
        Args:
            X: Features matrix
            y: Target values (relevance scores)
            group_ids: Group IDs for queries (user_ids)
            timestamps: Optional timestamps for time-based validation
            validation_fraction: Fraction of data to use for validation
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
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
        
        # Create train and validation pools
        train_pool = self._create_pool(
            X[train_idx], y[train_idx],
            group_ids[train_idx],
            None if timestamps is None else timestamps[train_idx]
        )
        valid_pool = self._create_pool(
            X[valid_idx], y[valid_idx],
            group_ids[valid_idx],
            None if timestamps is None else timestamps[valid_idx]
        )
        
        # Initialize model with best parameters if available
        params = self.best_params if self.best_params is not None else {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'YetiRank',
            'eval_metric': 'NDCG',
            'random_seed': 42,
            'verbose': False
        }
        
        self.model = CatBoostRanker(**params)
        
        # Train the model
        self.model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Get evaluation metrics
        metrics = {
            'train': {
                'ndcg': self.model.get_best_score()['learn']['NDCG'],
                'map': self.model.get_best_score()['learn']['MAP']
            },
            'validation': {
                'ndcg': self.model.get_best_score()['validation']['NDCG'],
                'map': self.model.get_best_score()['validation']['MAP']
            }
        }
        
        return metrics
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.get_feature_importance()
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, path):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.save_model(path)
        
        # Save best parameters
        if self.best_params is not None:
            np.save(path + '.params.npy', self.best_params)
    
    def load_model(self, path):
        """Load model from disk"""
        self.model = CatBoostRanker()
        self.model.load_model(path)
        
        # Load best parameters if available
        try:
            self.best_params = np.load(path + '.params.npy', allow_pickle=True).item()
        except FileNotFoundError:
            self.best_params = None
        
        return self
