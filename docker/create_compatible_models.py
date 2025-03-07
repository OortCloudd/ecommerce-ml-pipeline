import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix
import pandas as pd

# Create directories if they don't exist
os.makedirs('../models', exist_ok=True)

# Create a compatible ALS model
print("Creating compatible ALS model...")

# Create sparse matrices for user and item factors
num_users = 1000
num_items = 2000
factors = 50

# Create random user and item factors
user_factors = np.random.random((num_users, factors))
item_factors = np.random.random((num_items, factors))

# Convert to sparse format
user_factors_sparse = csr_matrix(user_factors)
item_factors_sparse = csr_matrix(item_factors)

# Create user and item mappings (required by implicit library)
user_mapping = {str(i): i for i in range(num_users)}
item_mapping = {str(i): i for i in range(num_items)}

# Save the model in NPZ format
np.savez('../models/als_model.npz',
         user_factors=user_factors_sparse,
         item_factors=item_factors_sparse,
         user_mapping=np.array(list(user_mapping.items()), dtype=object),
         item_mapping=np.array(list(item_mapping.items()), dtype=object))

print("ALS model created successfully!")

# Create a compatible CatBoost model
print("Creating compatible CatBoost model...")

try:
    # Try to import CatBoost for proper model creation
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(iterations=10, depth=2, learning_rate=0.1, loss_function='RMSE')
    
    # Generate dummy data for training
    X = np.random.random((100, 10))
    y = np.random.random(100)
    
    # Train the model
    model.fit(X, y, verbose=False)
    
    # Save the model
    model.save_model('../models/ranking_model.cbm')
    print("CatBoost model created with CatBoost library!")
    
except ImportError:
    # If CatBoost is not available, create a pickle file
    print("CatBoost not available, creating a pickle file instead...")
    ranking_model = {
        'model_type': 'catboost',
        'version': '1.0.0',
        'features': ['user_activity', 'item_popularity', 'category_id'],
        'weights': np.random.random(10),
        'bias': np.random.random(1)[0]
    }
    
    with open('../models/ranking_model.cbm', 'wb') as f:
        pickle.dump(ranking_model, f)
    print("Pickle-based model created as a fallback.")

print("Models created successfully!")
print(f"Models saved to: {os.path.abspath('../models')}")
