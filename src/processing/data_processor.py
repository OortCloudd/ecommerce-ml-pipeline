"""
Module for data processing and feature engineering for RetailRocket e-commerce dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        """Initialize the data processor"""
        self.event_types = {
            'view': 1,
            'addtocart': 2,
            'transaction': 3
        }

    def clean_events(self, events_df):
        """Clean the events dataframe"""
        logger.info("Cleaning events data...")
        
        # Convert timestamp to datetime
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], unit='ms')
        
        # Remove events with missing product IDs
        events_df = events_df.dropna(subset=['itemid'])
        
        # Convert event types to numeric
        events_df['event_type'] = events_df['event'].map(self.event_types)
        
        # Sort by timestamp
        events_df = events_df.sort_values('timestamp')
        
        return events_df

    def clean_item_properties(self, properties_df):
        """Clean the item properties dataframe"""
        logger.info("Cleaning item properties data...")
        
        # Convert timestamp to datetime
        properties_df['timestamp'] = pd.to_datetime(properties_df['timestamp'], unit='ms')
        
        # Get the latest property values for each item
        latest_properties = (properties_df.sort_values('timestamp')
                           .groupby(['itemid', 'property'])
                           .last()
                           .reset_index())
        
        # Pivot the properties to create a wide format
        item_features = latest_properties.pivot(
            index='itemid',
            columns='property',
            values='value'
        ).reset_index()
        
        return item_features

    def clean_category_tree(self, category_df):
        """Clean the category tree dataframe"""
        logger.info("Cleaning category tree data...")
        
        # Remove any duplicate categories
        category_df = category_df.drop_duplicates()
        
        # Create category path
        category_df['category_path'] = (category_df['parentid'].astype(str) + 
                                      '/' + 
                                      category_df['categoryid'].astype(str))
        
        return category_df

    def engineer_user_features(self, events_df):
        """Create user-level features"""
        logger.info("Engineering user features...")
        
        user_features = pd.DataFrame()
        
        # Basic user activity metrics
        user_activity = events_df.groupby('visitorid').agg({
            'timestamp': ['count', 'min', 'max'],
            'itemid': 'nunique',
            'event': lambda x: (x == 'transaction').sum()
        })
        
        user_activity.columns = [
            'total_events',
            'first_activity',
            'last_activity',
            'unique_items_viewed',
            'total_purchases'
        ]
        
        # Calculate user activity duration in days
        user_activity['activity_duration_days'] = (
            user_activity['last_activity'] - user_activity['first_activity']
        ).dt.total_seconds() / (24 * 3600)
        
        # Calculate event type ratios
        event_counts = events_df.groupby(['visitorid', 'event']).size().unstack(fill_value=0)
        event_counts.columns = [f'event_{col}_count' for col in event_counts.columns]
        
        # Combine all user features
        user_features = pd.concat([user_activity, event_counts], axis=1)
        
        return user_features

    def engineer_item_features(self, events_df, item_properties_df):
        """Create item-level features"""
        logger.info("Engineering item features...")
        
        # Basic item interaction metrics
        item_features = events_df.groupby('itemid').agg({
            'visitorid': ['count', 'nunique'],
            'event': lambda x: (x == 'transaction').sum()
        })
        
        item_features.columns = [
            'total_interactions',
            'unique_visitors',
            'total_purchases'
        ]
        
        # Calculate conversion rate
        item_features['conversion_rate'] = (
            item_features['total_purchases'] / item_features['total_interactions']
        )
        
        # Merge with item properties if available
        if item_properties_df is not None:
            item_features = item_features.merge(
                item_properties_df,
                left_index=True,
                right_on='itemid',
                how='left'
            )
        
        return item_features

    def create_interaction_matrix(self, events_df):
        """Create user-item interaction matrix for ALS"""
        logger.info("Creating interaction matrix...")
        
        # Create interaction strength based on event types
        events_df['interaction_value'] = events_df['event'].map({
            'view': 1,
            'addtocart': 2,
            'transaction': 3
        })
        
        # Aggregate interactions by user-item pairs
        interaction_matrix = (events_df.groupby(['visitorid', 'itemid'])
                            ['interaction_value']
                            .sum()
                            .reset_index())
        
        return interaction_matrix

    def process_all(self, events_df, item_properties_df, category_df):
        """Process all datasets and create features"""
        logger.info("Starting full data processing pipeline...")
        
        # Clean all datasets
        events_clean = self.clean_events(events_df)
        items_clean = self.clean_item_properties(item_properties_df)
        categories_clean = self.clean_category_tree(category_df)
        
        # Engineer features
        user_features = self.engineer_user_features(events_clean)
        item_features = self.engineer_item_features(events_clean, items_clean)
        interaction_matrix = self.create_interaction_matrix(events_clean)
        
        # Merge category information with item features
        if 'categoryid' in items_clean.columns:
            item_features = item_features.merge(
                categories_clean[['categoryid', 'category_path']],
                on='categoryid',
                how='left'
            )
        
        logger.info("Data processing completed successfully")
        
        return {
            'user_features': user_features,
            'item_features': item_features,
            'interaction_matrix': interaction_matrix,
            'events_clean': events_clean
        }
