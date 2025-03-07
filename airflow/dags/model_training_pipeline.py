"""
DAG for training recommendation models
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from src.model.collaborative.als_trainer import ALSTrainer
from src.model.ranking.ranking_trainer import RankingTrainer
from src.utils.s3_utils import download_from_s3, upload_to_s3

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 7),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def load_data(**context):
    """Load processed data from S3"""
    # Download processed data
    data_files = [
        'interaction_matrix.parquet',
        'user_features.parquet',
        'item_features.parquet'
    ]
    
    for file in data_files:
        download_from_s3(
            f'processed/{file}',
            f'/tmp/{file}'
        )
    
    # Load interaction matrix
    interactions = pd.read_parquet('/tmp/interaction_matrix.parquet')
    
    # Convert to sparse matrix
    interaction_matrix = csr_matrix(
        (interactions['value'].values,
         (interactions['user_id'].values, interactions['item_id'].values))
    )
    
    # Push to XCom
    context['task_instance'].xcom_push('interaction_matrix', interaction_matrix)
    context['task_instance'].xcom_push('interactions_df', interactions)

def train_als(**context):
    """Train ALS model"""
    # Get interaction matrix from XCom
    interaction_matrix = context['task_instance'].xcom_pull(
        task_ids='load_data',
        key='interaction_matrix'
    )
    
    # Train model
    als_trainer = ALSTrainer()
    als_trainer.train(interaction_matrix)
    
    # Save model
    als_trainer.save_model('/tmp/als_model.npz')
    
    # Upload to S3
    upload_to_s3(
        '/tmp/als_model.npz',
        'models/als_model.npz'
    )
    
    # Push user and item factors to XCom
    context['task_instance'].xcom_push(
        'user_factors',
        als_trainer.get_user_factors()
    )
    context['task_instance'].xcom_push(
        'item_factors',
        als_trainer.get_item_factors()
    )

def generate_candidates(**context):
    """Generate candidate items using ALS"""
    # Get data from XCom
    interactions_df = context['task_instance'].xcom_pull(
        task_ids='load_data',
        key='interactions_df'
    )
    user_factors = context['task_instance'].xcom_pull(
        task_ids='train_als',
        key='user_factors'
    )
    item_factors = context['task_instance'].xcom_pull(
        task_ids='train_als',
        key='item_factors'
    )
    
    # Load user and item features
    user_features = pd.read_parquet('/tmp/user_features.parquet')
    item_features = pd.read_parquet('/tmp/item_features.parquet')
    
    # Generate candidates for each user
    candidates = []
    for user_id in range(len(user_factors)):
        # Get top 100 items
        scores = user_factors[user_id].dot(item_factors.T)
        top_items = np.argsort(-scores)[:100]
        
        # Create feature vectors for each user-item pair
        for item_id in top_items:
            candidates.append({
                'user_id': user_id,
                'item_id': item_id,
                'als_score': scores[item_id],
                **user_features.loc[user_id].to_dict(),
                **item_features.loc[item_id].to_dict()
            })
    
    # Convert to DataFrame
    candidates_df = pd.DataFrame(candidates)
    
    # Add actual interaction scores
    interactions_matrix = pd.pivot_table(
        interactions_df,
        values='value',
        index='user_id',
        columns='item_id',
        fill_value=0
    )
    candidates_df['actual_score'] = [
        interactions_matrix.loc[row.user_id, row.item_id]
        if (row.user_id in interactions_matrix.index and 
            row.item_id in interactions_matrix.columns)
        else 0
        for _, row in candidates_df.iterrows()
    ]
    
    # Save candidates
    candidates_df.to_parquet('/tmp/ranking_candidates.parquet')
    
    # Push to XCom
    context['task_instance'].xcom_push('candidates_df', candidates_df)

def train_ranker(**context):
    """Train ranking model with hyperparameter tuning"""
    # Get candidates from XCom
    candidates_df = context['task_instance'].xcom_pull(
        task_ids='generate_candidates',
        key='candidates_df'
    )
    
    # Prepare features
    feature_cols = [col for col in candidates_df.columns
                   if col not in ['user_id', 'item_id', 'actual_score']]
    X = candidates_df[feature_cols].values
    y = candidates_df['actual_score'].values
    group_ids = candidates_df['user_id'].values
    
    # Get timestamps from interaction data
    interactions_df = context['task_instance'].xcom_pull(
        task_ids='load_data',
        key='interactions_df'
    )
    timestamps = interactions_df.groupby(['user_id', 'item_id'])['timestamp'].max()
    timestamps = candidates_df.apply(
        lambda row: timestamps.get((row['user_id'], row['item_id']), 
                                 timestamps.min()),
        axis=1
    ).values
    
    # Train model
    ranker = RankingTrainer()
    
    # Tune hyperparameters
    print("Starting hyperparameter tuning...")
    best_params = ranker.tune_hyperparameters(
        X, y, group_ids, timestamps,
        n_trials=50  # Adjust based on your time constraints
    )
    print("Best parameters:", best_params)
    
    # Train final model
    print("Training final model...")
    metrics = ranker.train(X, y, group_ids, timestamps)
    
    # Save model
    ranker.save_model('/tmp/ranking_model.cbm')
    
    # Upload to S3
    upload_to_s3(
        '/tmp/ranking_model.cbm',
        'models/ranking_model.cbm'
    )
    
    # Log metrics
    print("Ranking model metrics:", metrics)
    
    # Save feature importance
    importance_df = ranker.get_feature_importance()
    importance_df.to_csv('/tmp/feature_importance.csv', index=False)
    upload_to_s3(
        '/tmp/feature_importance.csv',
        'models/feature_importance.csv'
    )

# Create DAG
dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Train recommendation models',
    schedule_interval=timedelta(days=1)
)

# Define tasks
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

train_als_task = PythonOperator(
    task_id='train_als',
    python_callable=train_als,
    dag=dag
)

generate_candidates_task = PythonOperator(
    task_id='generate_candidates',
    python_callable=generate_candidates,
    dag=dag
)

train_ranker_task = PythonOperator(
    task_id='train_ranker',
    python_callable=train_ranker,
    dag=dag
)

# Set dependencies
load_data_task >> train_als_task >> generate_candidates_task >> train_ranker_task
