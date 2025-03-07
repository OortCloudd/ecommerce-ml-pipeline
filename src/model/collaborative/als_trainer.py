"""
Module for training and evaluating ALS (Alternating Least Squares) model
"""
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

class ALSTrainer:
    def __init__(self, factors=100, regularization=0.01, iterations=15):
        """
        Initialize ALS trainer
        
        Args:
            factors (int): Number of latent factors
            regularization (float): Regularization parameter
            iterations (int): Number of iterations to run
        """
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )
        
    def train(self, interaction_matrix):
        """
        Train the ALS model
        
        Args:
            interaction_matrix: User-item interaction matrix in CSR format
            
        Returns:
            self: Trained model instance
        """
        # Convert to float32 as required by implicit
        interaction_matrix = interaction_matrix.astype(np.float32)
        
        # Train the model
        self.model.fit(interaction_matrix)
        
        return self
    
    def get_user_factors(self):
        """Get user latent factors"""
        return self.model.user_factors
    
    def get_item_factors(self):
        """Get item latent factors"""
        return self.model.item_factors
    
    def recommend(self, user_id, n=10, filter_already_liked_items=True):
        """
        Get recommendations for a user
        
        Args:
            user_id (int): User ID to get recommendations for
            n (int): Number of recommendations to return
            filter_already_liked_items (bool): Whether to filter out items the user has already interacted with
            
        Returns:
            list: List of (item_id, score) tuples
        """
        if not hasattr(self.model, 'user_factors'):
            raise ValueError("Model not trained yet")
            
        recommendations = self.model.recommend(
            user_id,
            self.model.user_factors[user_id],
            N=n,
            filter_already_liked_items=filter_already_liked_items
        )
        
        return recommendations
    
    def similar_items(self, item_id, n=10):
        """
        Find similar items
        
        Args:
            item_id (int): Item ID to find similar items for
            n (int): Number of similar items to return
            
        Returns:
            list: List of (item_id, score) tuples
        """
        if not hasattr(self.model, 'item_factors'):
            raise ValueError("Model not trained yet")
            
        similar_items = self.model.similar_items(item_id, N=n)
        return similar_items
    
    def save_model(self, path):
        """Save model to disk"""
        np.savez(
            path,
            user_factors=self.model.user_factors,
            item_factors=self.model.item_factors
        )
    
    def load_model(self, path):
        """Load model from disk"""
        data = np.load(path)
        self.model.user_factors = data['user_factors']
        self.model.item_factors = data['item_factors']
        return self
