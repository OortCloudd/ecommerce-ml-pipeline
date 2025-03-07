"""
Module for data processing and feature engineering for RetailRocket e-commerce dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from .feature_engineering import AdvancedFeatureEngineer

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        """Initialize the data processor"""
        self.event_types = {
            'view': 1,
            'addtocart': 2,
            'transaction': 3
        }
        self.feature_engineer = AdvancedFeatureEngineer()

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
        
        # Convert IDs to integers
        events_df['visitorid'] = events_df['visitorid'].astype(int)
        events_df['itemid'] = events_df['itemid'].astype(int)
        
        return events_df

    def clean_item_properties(self, properties_df):
        """Clean the item properties dataframe"""
        logger.info("Cleaning item properties data...")
        
        # Convert timestamp to datetime
        properties_df['timestamp'] = pd.to_datetime(properties_df['timestamp'], unit='ms')
        
        # Convert IDs to integers
        properties_df['itemid'] = properties_df['itemid'].astype(int)
        
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
        
        # Convert numeric columns
        numeric_cols = ['price', 'available']
        for col in numeric_cols:
            if col in item_features.columns:
                item_features[col] = pd.to_numeric(
                    item_features[col],
                    errors='coerce'
                )
        
        return item_features

    def clean_category_tree(self, category_df):
        """Clean the category tree dataframe"""
        logger.info("Cleaning category tree data...")
        
        # Convert IDs to integers
        category_df['categoryid'] = category_df['categoryid'].astype(int)
        category_df['parentid'] = pd.to_numeric(
            category_df['parentid'],
            errors='coerce'
        )
        
        # Remove any duplicate categories
        category_df = category_df.drop_duplicates()
        
        # Create category path
        category_df['category_path'] = (
            category_df['parentid'].fillna('root').astype(str) + 
            '/' + 
            category_df['categoryid'].astype(str)
        )
        
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
        
        # Calculate conversion rates
        user_activity['view_to_cart_rate'] = (
            event_counts['event_addtocart_count'] /
            event_counts['event_view_count'].clip(lower=1)
        )
        user_activity['cart_to_purchase_rate'] = (
            event_counts['event_transaction_count'] /
            event_counts['event_addtocart_count'].clip(lower=1)
        )
        
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
        
        # Calculate conversion rates
        event_counts = events_df.groupby(['itemid', 'event']).size().unstack(fill_value=0)
        event_counts.columns = [f'event_{col}_count' for col in event_counts.columns]
        
        item_features['view_to_cart_rate'] = (
            event_counts['event_addtocart_count'] /
            event_counts['event_view_count'].clip(lower=1)
        )
        item_features['cart_to_purchase_rate'] = (
            event_counts['event_transaction_count'] /
            event_counts['event_addtocart_count'].clip(lower=1)
        )
        
        # Calculate popularity score
        item_features['popularity_score'] = (
            0.4 * item_features['total_interactions'] +
            0.3 * item_features['unique_visitors'] +
            0.3 * item_features['total_purchases']
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
        
        # Create interaction strength based on event types and recency
        events_df = events_df.copy()
        max_timestamp = events_df['timestamp'].max()
        events_df['days_ago'] = (
            (max_timestamp - events_df['timestamp']).dt.total_seconds() /
            (24 * 3600)
        )
        
        # Decay factor for older interactions (30-day half-life)
        events_df['time_weight'] = np.exp(-events_df['days_ago'] * np.log(2) / 30)
        
        # Event type weights
        events_df['event_weight'] = events_df['event'].map({
            'view': 1,
            'addtocart': 2,
            'transaction': 3
        })
        
        # Final interaction value combines event type and recency
        events_df['interaction_value'] = (
            events_df['event_weight'] * events_df['time_weight']
        )
        
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
        
        # Engineer basic features
        user_features = self.engineer_user_features(events_clean)
        item_features = self.engineer_item_features(events_clean, items_clean)
        interaction_matrix = self.create_interaction_matrix(events_clean)
        
        # Engineer advanced features
        advanced_features = self.feature_engineer.engineer_features(
            events_clean,
            items_clean,
            categories_clean
        )
        
        # Merge advanced features
        user_features = user_features.join(
            advanced_features['user_features'],
            how='left'
        )
        
        item_features = item_features.join(
            advanced_features['item_features'],
            how='left'
        )
        
        # Add category information
        if 'categoryid' in items_clean.columns:
            item_features = item_features.merge(
                categories_clean[['categoryid', 'category_path']],
                on='categoryid',
                how='left'
            )
        
        # Store additional features
        additional_features = {
            'session_features': advanced_features['session_features'],
            'category_features': advanced_features['category_features'],
            'sequence_features': advanced_features['sequence_features'],
            'transition_matrix': advanced_features['transition_matrix'],
            'common_sequences': advanced_features['common_sequences']
        }
        
        logger.info("Data processing completed successfully")
        
        return {
            'user_features': user_features,
            'item_features': item_features,
            'interaction_matrix': interaction_matrix,
            'events_clean': events_clean,
            'additional_features': additional_features
        }
