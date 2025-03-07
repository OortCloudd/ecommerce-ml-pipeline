import pandas as pd
import numpy as np
import os

# Create features directory if it doesn't exist
os.makedirs('../features', exist_ok=True)

print("Creating dummy user features...")
# Create dummy user features
user_ids = [str(i) for i in range(1, 1001)]
user_features = {
    'user_id': user_ids,
    'activity': np.random.random(1000),
    'recency': np.random.random(1000),
    'view_buy_ratio': np.random.random(1000) * 0.3,
    'session_count': np.random.randint(1, 50, 1000),
    'avg_session_length': np.random.random(1000) * 30,
    'last_active': pd.date_range(start='2025-01-01', periods=1000, freq='H')
}

user_df = pd.DataFrame(user_features)
user_df.set_index('user_id', inplace=True)
user_df.to_parquet('../features/user_features.parquet')

print("Creating dummy item features...")
# Create dummy item features
item_ids = [str(i) for i in range(1, 2001)]
categories = np.random.choice(['electronics', 'clothing', 'home', 'sports', 'beauty'], 2000)
item_features = {
    'item_id': item_ids,
    'category': categories,
    'price': np.random.random(2000) * 100,
    'popularity': np.random.random(2000),
    'view_count': np.random.randint(10, 1000, 2000),
    'purchase_count': np.random.randint(0, 100, 2000),
    'conversion_rate': np.random.random(2000) * 0.2,
    'avg_rating': np.random.random(2000) * 5
}

item_df = pd.DataFrame(item_features)
item_df.set_index('item_id', inplace=True)
item_df.to_parquet('../features/item_features.parquet')

print("Creating dummy interaction matrix...")
# Create a sparse interaction matrix for collaborative filtering
user_item_interactions = []
for i in range(5000):  # 5000 interactions
    user_id = np.random.choice(user_ids)
    item_id = np.random.choice(item_ids)
    interaction_value = np.random.random() * 5  # Rating between 0 and 5
    user_item_interactions.append({
        'user_id': user_id,
        'item_id': item_id,
        'interaction': interaction_value
    })

interaction_df = pd.DataFrame(user_item_interactions)
interaction_df.to_parquet('../features/interaction_matrix.parquet')

print("Features created successfully!")
print(f"Features saved to: {os.path.abspath('../features')}")
