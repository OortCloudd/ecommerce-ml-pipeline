import numpy as np
import os
import pickle

# Create directories if they don't exist
os.makedirs('../models', exist_ok=True)

# Create a simple ALS model
print("Creating simple ALS model...")

# Create random user and item factors as regular numpy arrays (not sparse)
num_users = 1000
num_items = 2000
factors = 50

user_factors = np.random.random((num_users, factors))
item_factors = np.random.random((num_items, factors))

# Save the model in NPZ format with only the essential arrays
np.savez('../models/als_model.npz',
         user_factors=user_factors,
         item_factors=item_factors)

print("ALS model created successfully!")

# Create a simple CatBoost model (just a pickle file with the expected structure)
print("Creating simple CatBoost model...")

# Simple model structure that can be pickled
class SimpleModel:
    def __init__(self):
        self.weights = np.random.random(10)
        self.bias = np.random.random()
    
    def predict(self, X):
        if hasattr(X, '__len__'):
            return np.random.random(len(X))
        return np.random.random()

ranking_model = SimpleModel()

with open('../models/ranking_model.cbm', 'wb') as f:
    pickle.dump(ranking_model, f)

print("Simple CatBoost model created successfully!")
print(f"Models saved to: {os.path.abspath('../models')}")
