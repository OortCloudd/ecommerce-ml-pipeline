#!/bin/bash
# Script to run the forward pass inside the Docker container

# Default parameters
USER_ID="1"
N_ITEMS=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --user-id)
      USER_ID="$2"
      shift 2
      ;;
    --n-items)
      N_ITEMS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Running forward pass for user_id=$USER_ID with n_items=$N_ITEMS"

# Copy the script to the container
docker cp run_forward_pass.py ecommerce-ml-prediction-service-1:/app/

# Run the script inside the container
docker exec ecommerce-ml-prediction-service-1 python /app/run_forward_pass.py --user-id "$USER_ID" --n-items "$N_ITEMS"

echo "Forward pass completed."
