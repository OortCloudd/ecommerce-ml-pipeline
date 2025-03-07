"""
Module for handling S3 data ingestion operations
"""
import boto3

class S3Connector:
    def __init__(self, bucket_name):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name

    def upload_file(self, file_path, s3_key):
        """Upload a file to S3"""
        self.s3_client.upload_file(file_path, self.bucket_name, s3_key)

    def download_file(self, s3_key, local_path):
        """Download a file from S3"""
        self.s3_client.download_file(self.bucket_name, s3_key, local_path)
