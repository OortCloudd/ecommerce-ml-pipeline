#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Forward Pass

This script runs the forward pass of the recommendation system using the models
and features from the Docker environment.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append('/app')

try:
    from src.model.collaborative.als_trainer import ALSTrainer
    from src.model.ranking.conformal_trainer import ConformalRankingTrainer
except ImportError:
    logger.error("Failed to import model classes. Make sure you're running this in the Docker container.")
    sys.exit(1)


def load_models():
    """Load ALS and ranking models from the Docker environment"""
    model_path = os.getenv("MODEL_PATH", "/app/models")
    allow_pickle = os.getenv("ALLOW_PICKLE", "false").lower() == "true"
    
    logger.info(f"Loading models from {model_path} (allow_pickle={allow_pickle})")
    
    # Initialize models
    als_model = ALSTrainer()
    ranking_model = ConformalRankingTrainer()
    
    # Load models
    als_path = os.path.join(model_path, "als_model.npz")
    ranking_path = os.path.join(model_path, "ranking_model.cbm")
    
    try:
        logger.info(f"Loading ALS model from {als_path}")
        als_model.load_model(als_path)
        logger.info("ALS model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading ALS model: {e}")
        return None, None
    
    try:
        logger.info(f"Loading ranking model from {ranking_path}")
        ranking_model.load_model(ranking_path)
        logger.info("Ranking model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading ranking model: {e}")
        return als_model, None
    
    return als_model, ranking_model


def load_features():
    """Load user and item features from the Docker environment"""
    feature_path = os.getenv("FEATURE_PATH", "/app/features")
    logger.info(f"Loading features from {feature_path}")
    
    user_features_path = os.path.join(feature_path, "user_features.parquet")
    item_features_path = os.path.join(feature_path, "item_features.parquet")
    
    try:
        logger.info(f"Loading user features from {user_features_path}")
        user_features = pd.read_parquet(user_features_path)
        logger.info(f"User features loaded successfully with shape {user_features.shape}")
    except Exception as e:
        logger.error(f"Error loading user features: {e}")
        return None, None
    
    try:
        logger.info(f"Loading item features from {item_features_path}")
        item_features = pd.read_parquet(item_features_path)
        logger.info(f"Item features loaded successfully with shape {item_features.shape}")
    except Exception as e:
        logger.error(f"Error loading item features: {e}")
        return user_features, None
    
    return user_features, item_features


def forward_pass(user_id, n_items, als_model, ranking_model, user_features, item_features):
    """Perform a forward pass through the recommendation system"""
    logger.info(f"Starting forward pass for user_id={user_id}")
    
    # Step 1: Generate candidates using ALS model
    logger.info("Step 1: Generating candidates using ALS model")
    try:
        # Check if we have user factors
        if not hasattr(als_model.model, 'user_factors') or not hasattr(als_model.model, 'item_factors'):
            logger.error("ALS model doesn't have user_factors or item_factors attributes")
            return None
        
        # Create a mapping of user IDs to indices if it doesn't exist
        # In our simple model, we'll just use the user_id directly as an index
        user_idx = int(user_id) % len(als_model.model.user_factors)
        logger.info(f"Using user index {user_idx} for user_id {user_id}")
        
        # Get user factors
        user_factors = als_model.model.user_factors[user_idx]
        logger.info(f"User factors shape: {user_factors.shape}")
        
        # Calculate scores for all items
        scores = als_model.model.item_factors.dot(user_factors)
        logger.info(f"Raw scores shape: {scores.shape}")
        
        # Get top N candidates
        top_n_indices = np.argsort(-scores)[:n_items*2]  # Get 2x candidates for re-ranking
        top_n_scores = scores[top_n_indices]
        
        # Map indices to item IDs (in our simple case, just use the indices as IDs)
        candidate_items = [str(idx) for idx in top_n_indices]
        
        logger.info(f"Generated {len(candidate_items)} candidates")
        
        # Create a dataframe of candidates with scores
        candidates_df = pd.DataFrame({
            'item_id': candidate_items,
            'als_score': top_n_scores
        })
        
        logger.info(f"Candidates:\n{candidates_df.head()}")
    except Exception as e:
        logger.error(f"Error in candidate generation: {e}")
        return None
    
    # Step 2: Re-rank candidates using the ranking model
    logger.info("Step 2: Re-ranking candidates using the ranking model")
    try:
        if ranking_model is None:
            logger.warning("Ranking model not available, using ALS scores only")
            final_scores = top_n_scores
            confidence_lower = top_n_scores - 0.1
            confidence_upper = top_n_scores + 0.1
            final_items = candidate_items[:n_items]
        else:
            # Prepare features for ranking
            ranking_features = []
            
            # Get user features
            if user_features is not None and str(user_id) in user_features.index:
                user_feat = user_features.loc[str(user_id)].to_dict()
            else:
                user_feat = {'user_activity': 0.5, 'avg_session_length': 10}
            
            # Combine with item features for each candidate
            for i, item_id in enumerate(candidate_items):
                item_feat = {}
                if item_features is not None and item_id in item_features.index:
                    item_feat = item_features.loc[item_id].to_dict()
                else:
                    item_feat = {'popularity': 0.5, 'price': 50.0, 'category_id': 1}
                
                # Combine user and item features
                combined_feat = {
                    **user_feat,
                    **item_feat,
                    'als_score': float(top_n_scores[i])
                }
                
                ranking_features.append(combined_feat)
            
            # Convert to DataFrame
            ranking_df = pd.DataFrame(ranking_features)
            logger.info(f"Ranking features:\n{ranking_df.head()}")
            
            # Make predictions with the ranking model
            predictions, lower_bounds, upper_bounds = ranking_model.predict(
                ranking_df, 
                confidence_level=0.9
            )
            
            logger.info(f"Ranking predictions shape: {predictions.shape}")
            
            # Sort by predicted scores
            sorted_indices = np.argsort(-predictions)[:n_items]
            final_items = [candidate_items[i] for i in sorted_indices]
            final_scores = predictions[sorted_indices]
            confidence_lower = lower_bounds[sorted_indices]
            confidence_upper = upper_bounds[sorted_indices]
    except Exception as e:
        logger.error(f"Error in re-ranking: {e}")
        # Fall back to ALS scores
        sorted_indices = np.argsort(-top_n_scores)[:n_items]
        final_items = [candidate_items[i] for i in sorted_indices]
        final_scores = top_n_scores[sorted_indices]
        confidence_lower = final_scores - 0.1
        confidence_upper = final_scores + 0.1
    
    # Step 3: Format final recommendations
    logger.info("Step 3: Formatting final recommendations")
    recommendations = []
    for i, (item_id, score, lower, upper) in enumerate(zip(final_items, final_scores, confidence_lower, confidence_upper)):
        # Get item metadata if available
        metadata = {}
        if item_features is not None and item_id in item_features.index:
            item_data = item_features.loc[item_id]
            metadata = {
                'category': str(item_data.get('category_id', 'unknown')),
                'price': float(item_data.get('price', 0.0)),
                'popularity': float(item_data.get('popularity', 0.0))
            }
        else:
            # Generate some dummy metadata
            metadata = {
                'category': 'electronics',
                'price': 50.0 + (i * 10),
                'popularity': 0.9 - (i * 0.05)
            }
        
        recommendations.append({
            'item_id': str(item_id),
            'score': float(score),
            'confidence_lower': float(max(0, lower)),
            'confidence_upper': float(min(1, upper)),
            'metadata': metadata
        })
    
    logger.info(f"Generated {len(recommendations)} final recommendations")
    return {
        'recommendations': recommendations,
        'model_version': '1.0.0',
        'feature_timestamp': datetime.now().isoformat()
    }


def main():
    """Main function to run a forward pass"""
    import argparse
    parser = argparse.ArgumentParser(description='Run a forward pass through the recommendation system')
    parser.add_argument('--user-id', type=str, default='1', help='User ID to generate recommendations for')
    parser.add_argument('--n-items', type=int, default=5, help='Number of recommendations to generate')
    args = parser.parse_args()
    
    # Load models and features
    als_model, ranking_model = load_models()
    user_features, item_features = load_features()
    
    if als_model is None:
        logger.error("Failed to load ALS model, cannot continue")
        return 1
    
    # Perform forward pass
    result = forward_pass(
        args.user_id, 
        args.n_items, 
        als_model, 
        ranking_model, 
        user_features, 
        item_features
    )
    
    if result is None:
        logger.error("Forward pass failed")
        return 1
    
    # Print results
    print("\nRecommendation Results:")
    print(json.dumps(result, indent=2))
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
