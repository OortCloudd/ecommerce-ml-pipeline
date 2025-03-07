import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix

# Create directories if they don't exist
os.makedirs('../models', exist_ok=True)
os.makedirs('../features', exist_ok=True)

# Create a dummy ALS model
print("Creating dummy ALS model...")
# ALS models are typically stored as sparse matrices
user_factors = np.random.random((100, 20))  # 100 users, 20 factors
item_factors = np.random.random((200, 20))  # 200 items, 20 factors

# Save in NPZ format (used by implicit library)
np.savez('../models/als_model.npz', 
         user_factors=user_factors,
         item_factors=item_factors)

# Create a dummy CatBoost model
print("Creating dummy CatBoost model...")
# For CatBoost, we'll create a simple pickle file with some metadata
ranking_model = {
    'model_type': 'catboost',
    'version': '1.0.0',
    'features': ['user_activity', 'item_popularity', 'category_id'],
    'weights': np.random.random(10),
    'bias': np.random.random(1)[0]
}

with open('../models/ranking_model.cbm', 'wb') as f:
    pickle.dump(ranking_model, f)

print("Models created successfully!")
print(f"Models saved to: {os.path.abspath('../models')}")
