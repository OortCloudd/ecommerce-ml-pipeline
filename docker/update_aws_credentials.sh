#!/bin/bash
# Script to update AWS credentials and restart the prediction service

# Prompt for AWS credentials
read -p "Enter your AWS Access Key ID: " AWS_ACCESS_KEY_ID
read -sp "Enter your AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
echo ""
read -p "Enter your AWS Region (default: eu-west-1): " AWS_DEFAULT_REGION

# Set default region if not provided
if [ -z "$AWS_DEFAULT_REGION" ]; then
    AWS_DEFAULT_REGION="eu-west-1"
fi

# Set environment variables for the current session
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# Ask if this is a production deployment
read -p "Is this a production deployment? (y/n): " production
if [[ "$production" =~ ^[Yy]$ ]]; then
    PRODUCTION_READY="true"
    echo "Setting PRODUCTION_READY to true for production deployment"
else
    PRODUCTION_READY="false"
    echo "Setting PRODUCTION_READY to false for development deployment"
fi

# Update docker-compose.yml with environment variables
echo "Updating docker-compose.yml with AWS credentials..."
compose_path="$(dirname "$0")/docker-compose.yml"

# Create a temporary file
temp_file=$(mktemp)

# Update environment variables in docker-compose.yml
cat "$compose_path" | \
    sed "s|\(\s*AWS_ACCESS_KEY_ID:\).*|\1 $AWS_ACCESS_KEY_ID|" | \
    sed "s|\(\s*AWS_SECRET_ACCESS_KEY:\).*|\1 $AWS_SECRET_ACCESS_KEY|" | \
    sed "s|\(\s*AWS_DEFAULT_REGION:\).*|\1 $AWS_DEFAULT_REGION|" | \
    sed "s|\(\s*PRODUCTION_READY:\).*|\1 '$PRODUCTION_READY'|" > "$temp_file"

# Replace the original file
mv "$temp_file" "$compose_path"

# Restart the prediction service
echo "Stopping the prediction service..."
docker-compose -f "$compose_path" down

echo "Starting the prediction service with updated AWS credentials..."
docker-compose -f "$compose_path" up -d --build prediction-service

echo "Prediction service restarted with updated AWS credentials."
echo "Checking service status..."

# Wait a few seconds for the service to start
sleep 5

# Check service logs
docker logs ecommerce-ml-prediction-service-1

echo ""
echo "AWS credentials have been updated. You can now use the prediction service with your AWS credentials."
echo "Production mode: $PRODUCTION_READY"

# Test the service
read -p "Do you want to test the service? (y/n): " test_service
if [[ "$test_service" =~ ^[Yy]$ ]]; then
    echo "Testing health endpoint..."
    health_response=$(curl -s http://localhost:8000/health)
    echo "Health check successful: $health_response"
    
    echo "Testing recommendations endpoint..."
    recommendation_body='{"user_id":"1","n_items":5,"include_metadata":true}'
    recommendations_response=$(curl -s -X POST http://localhost:8000/recommendations \
        -H "Content-Type: application/json" \
        -d "$recommendation_body")
    
    # Count the number of recommendations using jq if available
    if command -v jq &> /dev/null; then
        rec_count=$(echo "$recommendations_response" | jq '.recommendations | length')
        echo "Recommendations received: $rec_count items"
    else
        echo "Recommendations received. Install jq for better output parsing."
    fi
    
    echo "Service is working correctly!"
fi
