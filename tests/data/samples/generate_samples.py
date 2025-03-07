"""
Generate sample data for integration tests from RetailRocket dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_event_samples(n_samples=1000):
    """Generate sample events data"""
    # Create sample visitor IDs and item IDs
    visitor_ids = np.random.randint(1000000, 9999999, size=n_samples)
    item_ids = np.random.randint(100000, 999999, size=n_samples)
    
    # Generate timestamps over last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [
        start_date + timedelta(
            seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
        )
        for _ in range(n_samples)
    ]
    
    # Create events with different types
    event_types = np.random.choice(
        ['view', 'addtocart', 'transaction'],
        size=n_samples,
        p=[0.7, 0.2, 0.1]  # Realistic distribution of event types
    )
    
    # Create DataFrame
    events_df = pd.DataFrame({
        'visitorid': visitor_ids,
        'itemid': item_ids,
        'timestamp': timestamps,
        'event': event_types
    })
    
    return events_df

def generate_item_samples(events_df, n_categories=50):
    """Generate sample item data based on events"""
    unique_items = events_df['itemid'].unique()
    
    # Generate random categories
    categories = [f"category_{i}" for i in range(n_categories)]
    item_categories = np.random.choice(categories, size=len(unique_items))
    
    # Generate random prices
    prices = np.random.uniform(10.0, 1000.0, size=len(unique_items))
    
    # Create item properties
    items_df = pd.DataFrame({
        'itemid': unique_items,
        'category': item_categories,
        'price': prices,
        'available': np.random.choice([True, False], size=len(unique_items), p=[0.9, 0.1])
    })
    
    return items_df

def main():
    # Create samples directory if it doesn't exist
    samples_dir = os.path.dirname(__file__)
    os.makedirs(samples_dir, exist_ok=True)
    
    # Generate sample data
    print("Generating event samples...")
    events_df = generate_event_samples()
    
    print("Generating item samples...")
    items_df = generate_item_samples(events_df)
    
    # Save to parquet files
    events_path = os.path.join(samples_dir, "events_sample.parquet")
    items_path = os.path.join(samples_dir, "items_sample.parquet")
    
    print(f"Saving events to {events_path}")
    events_df.to_parquet(events_path, index=False)
    
    print(f"Saving items to {items_path}")
    items_df.to_parquet(items_path, index=False)
    
    print("Sample data generation complete!")
    print(f"Generated {len(events_df)} events for {len(events_df['visitorid'].unique())} visitors")
    print(f"Generated {len(items_df)} unique items in {len(items_df['category'].unique())} categories")

if __name__ == "__main__":
    main()
