#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize Forward Pass

This script visualizes the forward pass of the recommendation system,
showing how data flows through the ALS and ranking models with interactive plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import argparse

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.forward_pass import load_models, load_features, forward_pass


def visualize_als_factors(als_model, user_id):
    """Visualize ALS latent factors"""
    plt.figure(figsize=(12, 8))
    
    # Get user factors
    user_idx = als_model.model.user_items.shape[0] - 1  # Default to last user if not found
    if user_id in als_model.user_mapping:
        user_idx = als_model.user_mapping[user_id]
        print(f"Found user_id={user_id} at index {user_idx}")
    else:
        print(f"User ID {user_id} not found in ALS model, using fallback")
    
    # Get user factors
    user_factors = als_model.model.user_factors[user_idx]
    
    # Plot user factors
    plt.subplot(2, 1, 1)
    plt.bar(range(len(user_factors)), user_factors)
    plt.title(f'User {user_id} Latent Factors')
    plt.xlabel('Factor Index')
    plt.ylabel('Factor Value')
    plt.grid(True, alpha=0.3)
    
    # Get top 10 items for this user
    scores = als_model.model.item_factors.dot(user_factors)
    top_indices = np.argsort(-scores)[:10]
    
    # Map indices to item IDs
    top_items = []
    for idx in top_indices:
        for item_id, item_idx in als_model.item_mapping.items():
            if item_idx == idx:
                top_items.append(item_id)
                break
    
    # Plot top item factors
    plt.subplot(2, 1, 2)
    item_factors = []
    item_names = []
    
    for i, item_id in enumerate(top_items):
        if item_id in als_model.item_mapping:
            item_idx = als_model.item_mapping[item_id]
            item_factor = als_model.model.item_factors[item_idx]
            item_factors.append(item_factor)
            item_names.append(f"Item {item_id}")
    
    if item_factors:
        item_factors = np.array(item_factors)
        sns.heatmap(
            item_factors, 
            cmap='viridis', 
            yticklabels=item_names,
            xticklabels=False
        )
        plt.title('Top 10 Item Latent Factors')
        plt.xlabel('Factor Index')
    else:
        plt.text(0.5, 0.5, "No item factors available", ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('als_factors_visualization.png')
    print("Saved ALS factors visualization to als_factors_visualization.png")
    plt.close()


def visualize_recommendation_process(user_id, als_model, ranking_model, user_features, item_features):
    """Visualize the recommendation process from ALS to final ranking"""
    # Get user factors
    user_idx = als_model.model.user_items.shape[0] - 1  # Default to last user if not found
    if user_id in als_model.user_mapping:
        user_idx = als_model.user_mapping[user_id]
    
    # Get user factors
    user_factors = als_model.model.user_factors[user_idx]
    
    # Calculate scores for all items
    scores = als_model.model.item_factors.dot(user_factors)
    
    # Get top 20 candidates for visualization
    top_indices = np.argsort(-scores)[:20]
    top_scores = scores[top_indices]
    
    # Map indices to item IDs
    candidate_items = []
    for idx in top_indices:
        for item_id, item_idx in als_model.item_mapping.items():
            if item_idx == idx:
                candidate_items.append(item_id)
                break
    
    # Create a dataframe of candidates with scores
    candidates_df = pd.DataFrame({
        'item_id': candidate_items,
        'als_score': top_scores
    })
    
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
    
    # Make predictions with the ranking model if available
    if ranking_model is not None:
        predictions, lower_bounds, upper_bounds = ranking_model.predict(
            ranking_df, 
            confidence_level=0.9
        )
        
        # Add predictions to the dataframe
        candidates_df['ranking_score'] = predictions
        candidates_df['confidence_lower'] = lower_bounds
        candidates_df['confidence_upper'] = upper_bounds
    else:
        # If no ranking model, use ALS scores
        candidates_df['ranking_score'] = candidates_df['als_score']
        candidates_df['confidence_lower'] = candidates_df['als_score'] - 0.1
        candidates_df['confidence_upper'] = candidates_df['als_score'] + 0.1
    
    # Sort by ranking score
    candidates_df = candidates_df.sort_values('ranking_score', ascending=False).reset_index(drop=True)
    
    # Visualize the recommendation process
    plt.figure(figsize=(14, 10))
    
    # Plot ALS scores
    plt.subplot(2, 1, 1)
    sns.barplot(x='item_id', y='als_score', data=candidates_df.head(10), palette='viridis')
    plt.title('Top 10 Items by ALS Score')
    plt.xlabel('Item ID')
    plt.ylabel('ALS Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot ranking scores with confidence intervals
    plt.subplot(2, 1, 2)
    ax = sns.barplot(x='item_id', y='ranking_score', data=candidates_df.head(10), palette='viridis')
    
    # Add confidence intervals
    for i, row in candidates_df.head(10).iterrows():
        plt.plot([i, i], [row['confidence_lower'], row['confidence_upper']], color='red', alpha=0.7)
        plt.plot([i-0.1, i+0.1], [row['confidence_lower'], row['confidence_lower']], color='red', alpha=0.7)
        plt.plot([i-0.1, i+0.1], [row['confidence_upper'], row['confidence_upper']], color='red', alpha=0.7)
    
    plt.title('Top 10 Items by Ranking Score with Confidence Intervals')
    plt.xlabel('Item ID')
    plt.ylabel('Ranking Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('recommendation_process_visualization.png')
    print("Saved recommendation process visualization to recommendation_process_visualization.png")
    plt.close()
    
    # Return the top 10 recommendations
    return candidates_df.head(10)


def visualize_feature_importance(ranking_model, feature_names=None):
    """Visualize feature importance from the ranking model"""
    if ranking_model is None or not hasattr(ranking_model.model, 'get_feature_importance'):
        print("Ranking model doesn't support feature importance visualization")
        return
    
    try:
        # Get feature importance
        feature_importance = ranking_model.model.get_feature_importance()
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(feature_importance)],
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title('Feature Importance in Ranking Model')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_importance_visualization.png')
        print("Saved feature importance visualization to feature_importance_visualization.png")
        plt.close()
        
        return importance_df
    except Exception as e:
        print(f"Error visualizing feature importance: {e}")
        return None


def main():
    """Main function to visualize the forward pass"""
    parser = argparse.ArgumentParser(description='Visualize the forward pass of the recommendation system')
    parser.add_argument('--user-id', type=str, default='1', help='User ID to generate recommendations for')
    parser.add_argument('--model-path', type=str, default='./models', help='Path to model files')
    parser.add_argument('--feature-path', type=str, default='./data/processed', help='Path to feature files')
    args = parser.parse_args()
    
    # Load models and features
    als_model, ranking_model = load_models(args.model_path)
    user_features, item_features = load_features(args.feature_path)
    
    if als_model is None:
        print("Failed to load ALS model, cannot continue")
        return 1
    
    # Visualize ALS factors
    visualize_als_factors(als_model, args.user_id)
    
    # Visualize recommendation process
    top_recommendations = visualize_recommendation_process(
        args.user_id, 
        als_model, 
        ranking_model, 
        user_features, 
        item_features
    )
    
    # Visualize feature importance if ranking model is available
    if ranking_model is not None:
        feature_names = None
        if user_features is not None and item_features is not None:
            # Get feature names from the dataframes
            feature_names = list(user_features.columns) + list(item_features.columns) + ['als_score']
        
        visualize_feature_importance(ranking_model, feature_names)
    
    # Print top recommendations
    print("\nTop 10 Recommendations:")
    print(top_recommendations.to_string(index=False))
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
