"""
Advanced feature engineering for e-commerce recommendation system
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import networkx as nx
from typing import Dict, List, Tuple

class AdvancedFeatureEngineer:
    def __init__(self):
        """Initialize feature engineer with default parameters"""
        self.time_windows = [1, 7, 30]  # Days for temporal features
        self.price_buckets = [0, 10, 50, 100, 500, float('inf')]
        
    def _calculate_temporal_features(
        self,
        events_df: pd.DataFrame,
        reference_time: datetime = None
    ) -> pd.DataFrame:
        """
        Calculate time-based features for users and items
        
        Args:
            events_df: Clean events dataframe
            reference_time: Reference time for temporal calculations
        """
        if reference_time is None:
            reference_time = events_df['timestamp'].max()
            
        temporal_features = {}
        
        for window in self.time_windows:
            window_start = reference_time - timedelta(days=window)
            window_df = events_df[events_df['timestamp'] >= window_start]
            
            # User activity in time window
            user_window = window_df.groupby('visitorid').agg({
                'timestamp': 'count',
                'itemid': 'nunique',
                'event': lambda x: (x == 'transaction').sum()
            })
            
            user_window.columns = [
                f'user_{window}d_events',
                f'user_{window}d_unique_items',
                f'user_{window}d_purchases'
            ]
            
            # Item popularity in time window
            item_window = window_df.groupby('itemid').agg({
                'visitorid': 'count',
                'event': lambda x: (x == 'transaction').sum()
            })
            
            item_window.columns = [
                f'item_{window}d_views',
                f'item_{window}d_purchases'
            ]
            
            temporal_features[f'user_{window}d'] = user_window
            temporal_features[f'item_{window}d'] = item_window
            
        return temporal_features
    
    def _calculate_session_features(
        self,
        events_df: pd.DataFrame,
        session_timeout: int = 30
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate session-based features
        
        Args:
            events_df: Clean events dataframe
            session_timeout: Minutes to consider for session timeout
        """
        # Sort events by user and time
        events_df = events_df.sort_values(['visitorid', 'timestamp'])
        
        # Calculate time difference between consecutive events
        events_df['time_diff'] = events_df.groupby('visitorid')['timestamp'].diff()
        
        # Mark new sessions when time difference exceeds timeout
        events_df['new_session'] = (
            (events_df['time_diff'] > pd.Timedelta(minutes=session_timeout)) |
            (events_df['time_diff'].isna())
        ).astype(int)
        
        events_df['session_id'] = (
            events_df.groupby('visitorid')['new_session'].cumsum()
        )
        
        # Session-level features
        session_features = events_df.groupby(['visitorid', 'session_id']).agg({
            'timestamp': ['count', lambda x: (x.max() - x.min()).total_seconds()],
            'itemid': 'nunique',
            'event': lambda x: (x == 'transaction').sum()
        })
        
        session_features.columns = [
            'session_events',
            'session_duration',
            'session_unique_items',
            'session_purchases'
        ]
        
        # Aggregate session features by user
        user_session_features = pd.DataFrame()
        
        # Average session metrics
        user_session_agg = session_features.groupby('visitorid').agg({
            'session_events': ['mean', 'std', 'max'],
            'session_duration': ['mean', 'std', 'max'],
            'session_unique_items': ['mean', 'std', 'max'],
            'session_purchases': ['mean', 'sum']
        })
        
        user_session_agg.columns = [
            'avg_session_events', 'std_session_events', 'max_session_events',
            'avg_session_duration', 'std_session_duration', 'max_session_duration',
            'avg_session_items', 'std_session_items', 'max_session_items',
            'avg_session_purchases', 'total_purchases'
        ]
        
        return user_session_agg, session_features
    
    def _calculate_price_features(
        self,
        events_df: pd.DataFrame,
        item_properties_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate price-based features
        
        Args:
            events_df: Clean events dataframe
            item_properties_df: Clean item properties dataframe
        """
        # Extract price from item properties
        price_df = item_properties_df[['itemid', 'price']].copy()
        
        # Create price buckets
        price_df['price_bucket'] = pd.cut(
            price_df['price'],
            bins=self.price_buckets,
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        # User price preferences
        user_price_features = events_df.merge(
            price_df,
            on='itemid',
            how='left'
        ).groupby('visitorid').agg({
            'price': ['mean', 'std', 'max', 'min'],
            'price_bucket': lambda x: x.mode().iloc[0] if len(x) > 0 else None
        })
        
        user_price_features.columns = [
            'avg_price_viewed',
            'std_price_viewed',
            'max_price_viewed',
            'min_price_viewed',
            'preferred_price_range'
        ]
        
        # Item price features relative to category
        item_price_features = price_df.merge(
            item_properties_df[['itemid', 'categoryid']],
            on='itemid',
            how='left'
        )
        
        category_stats = item_price_features.groupby('categoryid')['price'].agg([
            'mean', 'std'
        ])
        
        item_price_features = item_price_features.merge(
            category_stats,
            on='categoryid',
            how='left'
        )
        
        item_price_features['price_vs_category'] = (
            (item_price_features['price'] - item_price_features['mean']) /
            item_price_features['std'].clip(lower=1e-6)
        )
        
        return user_price_features, item_price_features
    
    def _calculate_category_features(
        self,
        events_df: pd.DataFrame,
        item_properties_df: pd.DataFrame,
        category_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate category-based features
        
        Args:
            events_df: Clean events dataframe
            item_properties_df: Clean item properties dataframe
            category_df: Clean category tree dataframe
        """
        # Build category graph
        G = nx.DiGraph()
        for _, row in category_df.iterrows():
            if pd.notna(row['parentid']):
                G.add_edge(row['parentid'], row['categoryid'])
                
        # Calculate category depth and number of subcategories
        category_features = pd.DataFrame()
        category_features['depth'] = {
            node: len(nx.ancestors(G, node)) for node in G.nodes()
        }
        category_features['subcategories'] = {
            node: len(nx.descendants(G, node)) for node in G.nodes()
        }
        
        # User category preferences
        items_with_categories = item_properties_df[['itemid', 'categoryid']].merge(
            events_df[['visitorid', 'itemid', 'event']],
            on='itemid',
            how='right'
        )
        
        user_categories = items_with_categories.groupby(
            ['visitorid', 'categoryid']
        ).size().unstack(fill_value=0)
        
        # Calculate category diversity
        user_category_features = pd.DataFrame(index=user_categories.index)
        user_category_features['category_diversity'] = -(
            (user_categories.div(user_categories.sum(axis=1), axis=0) + 1e-10)
            .apply(lambda x: np.sum(x * np.log(x)), axis=1)
        )
        
        return user_category_features, category_features
    
    def _calculate_sequence_features(
        self,
        events_df: pd.DataFrame,
        session_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate sequence-based features
        
        Args:
            events_df: Clean events dataframe
            session_features: Session-level features
        """
        sequence_features = pd.DataFrame()
        
        # Calculate transition probabilities between event types
        events_df = events_df.sort_values(['visitorid', 'timestamp'])
        events_df['next_event'] = events_df.groupby('visitorid')['event'].shift(-1)
        
        transition_matrix = pd.crosstab(
            events_df['event'],
            events_df['next_event'],
            normalize='index'
        )
        
        # Calculate common sequences
        events_df['event_sequence'] = (
            events_df.groupby('visitorid')['event']
            .transform(lambda x: '->'.join(x))
        )
        
        common_sequences = (
            events_df.groupby('visitorid')['event_sequence']
            .first()
            .value_counts()
            .head(10)
            .index
            .tolist()
        )
        
        return sequence_features, transition_matrix, common_sequences
    
    def engineer_features(
        self,
        events_df: pd.DataFrame,
        item_properties_df: pd.DataFrame,
        category_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Engineer all advanced features
        
        Args:
            events_df: Clean events dataframe
            item_properties_df: Clean item properties dataframe
            category_df: Clean category tree dataframe
            
        Returns:
            Dictionary containing all engineered feature dataframes
        """
        # Calculate temporal features
        temporal_features = self._calculate_temporal_features(events_df)
        
        # Calculate session features
        user_session_features, session_features = self._calculate_session_features(
            events_df
        )
        
        # Calculate price features
        user_price_features, item_price_features = self._calculate_price_features(
            events_df,
            item_properties_df
        )
        
        # Calculate category features
        user_category_features, category_features = self._calculate_category_features(
            events_df,
            item_properties_df,
            category_df
        )
        
        # Calculate sequence features
        sequence_features, transition_matrix, common_sequences = self._calculate_sequence_features(
            events_df,
            session_features
        )
        
        # Combine all user features
        user_features = pd.concat([
            temporal_features.get('user_1d', pd.DataFrame()),
            temporal_features.get('user_7d', pd.DataFrame()),
            temporal_features.get('user_30d', pd.DataFrame()),
            user_session_features,
            user_price_features,
            user_category_features
        ], axis=1)
        
        # Combine all item features
        item_features = pd.concat([
            temporal_features.get('item_1d', pd.DataFrame()),
            temporal_features.get('item_7d', pd.DataFrame()),
            temporal_features.get('item_30d', pd.DataFrame()),
            item_price_features.set_index('itemid')
        ], axis=1)
        
        return {
            'user_features': user_features,
            'item_features': item_features,
            'session_features': session_features,
            'category_features': category_features,
            'sequence_features': sequence_features,
            'transition_matrix': transition_matrix,
            'common_sequences': common_sequences
        }
