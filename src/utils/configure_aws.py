import os
import configparser
from pathlib import Path

def configure_aws_credentials(access_key, secret_key, region='us-east-1'):
    """Configure AWS credentials file."""
    # Get the AWS credentials file path
    aws_dir = Path.home() / '.aws'
    aws_dir.mkdir(exist_ok=True)
    credentials_path = aws_dir / 'credentials'
    
    # Create a new credentials file from scratch
    config = configparser.ConfigParser()
    config['default'] = {
        'aws_access_key_id': access_key,
        'aws_secret_access_key': secret_key,
        'region': region
    }
    
    # Save the credentials with UTF-8 encoding
    with open(credentials_path, 'w', encoding='utf-8') as f:
        config.write(f)
    
    print(f"AWS credentials configured successfully in {credentials_path}")

if __name__ == "__main__":
    print("Please enter your AWS credentials:")
    access_key = input("Access Key ID: ").strip()
    secret_key = input("Secret Access Key: ").strip()
    region = input("Region (press Enter for us-east-1): ").strip() or 'us-east-1'
    
    if region == 'eu':
        region = 'eu-west-1'  # Default to Ireland region if just 'eu' is entered
    
    configure_aws_credentials(access_key, secret_key, region)
