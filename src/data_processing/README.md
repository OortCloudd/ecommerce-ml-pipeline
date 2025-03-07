# Data Processing

This directory contains the data processing components for the e-commerce ML pipeline, focusing on transforming raw RetailRocket data into features for model training.

## Components

### Data Processor (`data_processor.py`)

Handles the core data processing tasks:
- Event data cleaning (timestamps, missing values)
- Item properties processing (latest values)
- Category tree handling
- Data validation and quality checks

### Feature Engineering (`feature_engineering.py`)

Implements feature creation for recommendation models:
- User feature engineering (activity metrics, event ratios)
- Item feature engineering (interaction metrics, conversion rates)
- Interaction matrix creation for ALS collaborative filtering

## Output Features

The processing pipeline produces several key feature files:
- `user_features.parquet`: User-level features
- `item_features.parquet`: Item-level features
- `interaction_matrix.parquet`: Sparse user-item interaction matrix
- `events_clean.parquet`: Processed event data

## Integration with Airflow

These components are used in the Airflow DAG with the following structure:
1. `extract_data`: Downloads from S3 (raw/)
2. `transform_data`: Processes using DataProcessor
3. `load_to_s3`: Uploads to S3 (processed/)
4. `cleanup`: Removes temporary files
