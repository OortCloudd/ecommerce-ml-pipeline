from datetime import datetime, timedelta
import os
import pandas as pd
import logging
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from src.processing.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS S3 configuration
S3_BUCKET = "ecommerce-ml-pipeline-data"
RAW_PREFIX = "raw"
PROCESSED_PREFIX = "processed"

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 7),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'ecommerce_data_pipeline',
    default_args=default_args,
    description='E-commerce data pipeline for ML processing',
    schedule_interval=timedelta(days=1),
)

def download_from_s3(key, local_path):
    """Download a file from S3"""
    try:
        s3_hook = S3Hook()
        s3_hook.get_key(key, S3_BUCKET).download_file(local_path)
        logger.info(f"Successfully downloaded {key} to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {key}: {str(e)}")
        raise

def upload_to_s3(local_path, key):
    """Upload a file to S3"""
    try:
        s3_hook = S3Hook()
        s3_hook.load_file(
            filename=local_path,
            key=key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        logger.info(f"Successfully uploaded {local_path} to {key}")
        return True
    except Exception as e:
        logger.error(f"Error uploading {local_path}: {str(e)}")
        raise

def extract_data(**context):
    """Extract data from S3 source"""
    logger.info("Starting data extraction...")
    
    # Create temp directory if it doesn't exist
    os.makedirs('tmp', exist_ok=True)
    
    # Download all required files
    files = {
        f"{RAW_PREFIX}/events.csv": "tmp/events.csv",
        f"{RAW_PREFIX}/item_properties_part1.csv": "tmp/item_properties_1.csv",
        f"{RAW_PREFIX}/item_properties_part2.csv": "tmp/item_properties_2.csv",
        f"{RAW_PREFIX}/category_tree.csv": "tmp/category_tree.csv"
    }
    
    for s3_key, local_path in files.items():
        download_from_s3(s3_key, local_path)
    
    logger.info("Data extraction completed")
    
    # Pass the file paths to the next task
    return {
        'events_path': 'tmp/events.csv',
        'properties_1_path': 'tmp/item_properties_1.csv',
        'properties_2_path': 'tmp/item_properties_2.csv',
        'category_path': 'tmp/category_tree.csv'
    }

def transform_data(**context):
    """Transform the extracted data"""
    logger.info("Starting data transformation...")
    
    # Get file paths from previous task
    ti = context['task_instance']
    file_paths = ti.xcom_pull(task_ids='extract_data')
    
    # Read the data
    events_df = pd.read_csv(file_paths['events_path'])
    
    # Combine item properties parts
    properties_1 = pd.read_csv(file_paths['properties_1_path'])
    properties_2 = pd.read_csv(file_paths['properties_2_path'])
    properties_df = pd.concat([properties_1, properties_2])
    
    category_df = pd.read_csv(file_paths['category_path'])
    
    # Process the data
    processor = DataProcessor()
    processed_data = processor.process_all(
        events_df=events_df,
        item_properties_df=properties_df,
        category_df=category_df
    )
    
    # Save processed data
    output_paths = {}
    for name, df in processed_data.items():
        output_path = f"tmp/{name}.parquet"
        df.to_parquet(output_path, index=False)
        output_paths[name] = output_path
    
    logger.info("Data transformation completed")
    return output_paths

def load_to_s3(**context):
    """Load transformed data to S3"""
    logger.info("Starting data upload to S3...")
    
    # Get file paths from previous task
    ti = context['task_instance']
    file_paths = ti.xcom_pull(task_ids='transform_data')
    
    # Upload each file to S3
    for name, local_path in file_paths.items():
        s3_key = f"{PROCESSED_PREFIX}/{name}.parquet"
        upload_to_s3(local_path, s3_key)
    
    # Clean up temporary files
    for path in file_paths.values():
        if os.path.exists(path):
            os.remove(path)
    
    logger.info("Data upload completed")

def cleanup(**context):
    """Clean up temporary files"""
    logger.info("Cleaning up temporary files...")
    
    if os.path.exists('tmp'):
        for file in os.listdir('tmp'):
            file_path = os.path.join('tmp', file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir('tmp')
    
    logger.info("Cleanup completed")

# Create tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    provide_context=True,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    provide_context=True,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_to_s3',
    python_callable=load_to_s3,
    provide_context=True,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
extract_task >> transform_task >> load_task >> cleanup_task
