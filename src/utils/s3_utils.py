import boto3
import os
from botocore.exceptions import ClientError
import configparser
from pathlib import Path

def get_aws_region():
    """Get AWS region from credentials file."""
    credentials_path = Path.home() / '.aws' / 'credentials'
    if credentials_path.exists():
        config = configparser.ConfigParser()
        config.read(credentials_path)
        return config['default'].get('region', 'us-east-1')
    return 'us-east-1'

def create_bucket(bucket_name, region=None):
    """Create an S3 bucket."""
    if region is None:
        region = get_aws_region()
    
    try:
        s3_client = boto3.client('s3', region_name=region)
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f"Created bucket: {bucket_name} in region: {region}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(f"Bucket {bucket_name} already exists and is owned by you")
            return True
        print(f"Error creating bucket: {e}")
        return False

def upload_file(file_path, bucket, s3_key):
    """Upload a file to S3."""
    try:
        s3_client = boto3.client('s3', region_name=get_aws_region())
        print(f"Uploading {file_path} to s3://{bucket}/{s3_key}")
        s3_client.upload_file(file_path, bucket, s3_key)
        print(f"Successfully uploaded {file_path}")
        return True
    except ClientError as e:
        print(f"Error uploading file: {e}")
        return False

def upload_dataset(local_dir, bucket_name, prefix="raw"):
    """Upload all files from a directory to S3."""
    success = True
    for file in os.listdir(local_dir):
        if file.endswith('.csv'):
            local_path = os.path.join(local_dir, file)
            s3_key = f"{prefix}/{file}"
            if not upload_file(local_path, bucket_name, s3_key):
                success = False
    return success

def main():
    # Configuration
    BUCKET_NAME = "ecommerce-ml-pipeline-data"
    DATASET_DIR = "data/raw/RetailRocketDataset"
    
    # Create bucket
    if not create_bucket(BUCKET_NAME):
        print("Failed to create/verify bucket. Exiting.")
        return
    
    # Upload dataset
    print(f"\nUploading dataset from {DATASET_DIR} to S3...")
    if upload_dataset(DATASET_DIR, BUCKET_NAME):
        print("\nSuccessfully uploaded all files to S3!")
        print(f"Data is now available at s3://{BUCKET_NAME}/raw/")
    else:
        print("\nSome files failed to upload. Please check the errors above.")

if __name__ == "__main__":
    main()
