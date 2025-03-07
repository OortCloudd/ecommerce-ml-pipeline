#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Forward Pass Demonstration

This script demonstrates the forward pass of the recommendation system,
showing how data flows through the ALS and ranking models to generate recommendations.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.collaborative.als_trainer import ALSTrainer
from src.model.ranking.conformal_trainer import ConformalRankingTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_models(model_path):
    """Load ALS and ranking models"""
    logger.info("Loading models from %s", model_path)
    
    # Initialize models
    als_model = ALSTrainer()
    ranking_model = ConformalRankingTrainer()
    
    # Load models
    als_path = os.path.join(model_path, "als_model.npz")
    ranking_path = os.path.join(model_path, "ranking_model.cbm")
    
    try:
        logger.info("Loading ALS model from %s", als_path)
        als_model.load_model(als_path)
        logger.info("ALS model loaded successfully")
    except Exception as e:
        logger.error("Error loading ALS model: %s", e)
        return None, None
    
    try:
        logger.info("Loading ranking model from %s", ranking_path)
        ranking_model.load_model(ranking_path)
        logger.info("Ranking model loaded successfully")
    except Exception as e:
        logger.error("Error loading ranking model: %s", e)
        return als_model, None
    
    return als_model, ranking_model


def load_features(feature_path):
    """Load user and item features"""
    logger.info("Loading features from %s", feature_path)
    
    user_features_path = os.path.join(feature_path, "user_features.parquet")
    item_features_path = os.path.join(feature_path, "item_features.parquet")
    
    try:
        logger.info("Loading user features from %s", user_features_path)
        user_features = pd.read_parquet(user_features_path)
        logger.info("User features loaded successfully with shape %s", user_features.shape)
    except Exception as e:
        logger.error("Error loading user features: %s", e)
        return None, None
    
    try:
        logger.info("Loading item features from %s", item_features_path)
        item_features = pd.read_parquet(item_features_path)
        logger.info("Item features loaded successfully with shape %s", item_features.shape)
    except Exception as e:
        logger.error("Error loading item features: %s", e)
        return user_features, None
    
    return user_features, item_features


def forward_pass(user_id, n_items, als_model, ranking_model, user_features, item_features):
    """Perform a forward pass through the recommendation system"""
    logger.info("Starting forward pass for user_id=%s", user_id)
    
    # Step 1: Generate candidates using ALS model
    logger.info("Step 1: Generating candidates using ALS model")
    try:
        # Get user factors
        user_idx = als_model.model.user_items.shape[0] - 1  # Default to last user if not found
        if user_id in als_model.user_mapping:
            user_idx = als_model.user_mapping[user_id]
            logger.info("Found user_id=%s at index %d", user_id, user_idx)
        else:
            logger.warning("User ID %s not found in ALS model, using fallback", user_id)
        
        # Get user factors
        user_factors = als_model.model.user_factors[user_idx]
        logger.info("User factors shape: %s", user_factors.shape)
        
        # Calculate scores for all items
        scores = als_model.model.item_factors.dot(user_factors)
        logger.info("Raw scores shape: %s", scores.shape)
        
        # Get top N candidates
        top_n_indices = np.argsort(-scores)[:n_items*2]  # Get 2x candidates for re-ranking
        top_n_scores = scores[top_n_indices]
        
        # Map indices back to item IDs
        candidate_items = []
        for idx in top_n_indices:
            for item_id, item_idx in als_model.item_mapping.items():
                if item_idx == idx:
                    candidate_items.append(item_id)
                    break
        
        logger.info("Generated %d candidates", len(candidate_items))
        
        # Create a dataframe of candidates with scores
        candidates_df = pd.DataFrame({
            'item_id': candidate_items,
            'als_score': top_n_scores
        })
        
        logger.info("Candidates:\n%s", candidates_df.head())
    except Exception as e:
        logger.error("Error in candidate generation: %s", e)
        return None
    
    # Step 2: Re-rank candidates using the ranking model
    logger.info("Step 2: Re-ranking candidates using the ranking model")
    try:
        if ranking_model is None:
            logger.warning("Ranking model not available, using ALS scores only")
            final_scores = top_n_scores
            confidence_lower = top_n_scores - 0.1
            confidence_upper = top_n_scores + 0.1
        else:
            # Prepare features for ranking
            ranking_features = []
            
            # Get user features
            if user_features is not None and user_id in user_features.index:
                user_feat = user_features.loc[user_id].to_dict()
            else:
                user_feat = {'user_activity': 0.5, 'avg_session_length': 10}
            
            # Combine with item features for each candidate
            for item_id in candidate_items:
                item_feat = {}
                if item_features is not None and item_id in item_features.index:
                    item_feat = item_features.loc[item_id].to_dict()
                else:
                    item_feat = {'popularity': 0.5, 'price': 50.0, 'category_id': 1}
                
                # Combine user and item features
                combined_feat = {
                    **user_feat,
                    **item_feat,
                    'als_score': scores[als_model.item_mapping.get(item_id, 0)]
                }
                
                ranking_features.append(combined_feat)
            
            # Convert to DataFrame
            ranking_df = pd.DataFrame(ranking_features)
            logger.info("Ranking features:\n%s", ranking_df.head())
            
            # Make predictions with the ranking model
            predictions, lower_bounds, upper_bounds = ranking_model.predict(
                ranking_df, 
                confidence_level=0.9
            )
            
            logger.info("Ranking predictions shape: %s", predictions.shape)
            
            # Sort by predicted scores
            sorted_indices = np.argsort(-predictions)[:n_items]
            final_items = [candidate_items[i] for i in sorted_indices]
            final_scores = predictions[sorted_indices]
            confidence_lower = lower_bounds[sorted_indices]
            confidence_upper = upper_bounds[sorted_indices]
    except Exception as e:
        logger.error("Error in re-ranking: %s", e)
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
        
        recommendations.append({
            'item_id': str(item_id),
            'score': float(score),
            'confidence_lower': float(max(0, lower)),
            'confidence_upper': float(min(1, upper)),
            'metadata': metadata
        })
    
    logger.info("Generated %d final recommendations", len(recommendations))
    return {
        'recommendations': recommendations,
        'model_version': '1.0.0',
        'feature_timestamp': datetime.now().isoformat()
    }


def main():
    """Main function to demonstrate a forward pass"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Demonstrate a forward pass through the recommendation system')
    parser.add_argument('--user-id', type=str, default='1', help='User ID to generate recommendations for')
    parser.add_argument('--n-items', type=int, default=5, help='Number of recommendations to generate')
    parser.add_argument('--model-path', type=str, default='./models', help='Path to model files')
    parser.add_argument('--feature-path', type=str, default='./data/processed', help='Path to feature files')
    args = parser.parse_args()
    
    # Load models and features
    als_model, ranking_model = load_models(args.model_path)
    user_features, item_features = load_features(args.feature_path)
    
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
