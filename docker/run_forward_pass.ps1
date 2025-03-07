# PowerShell script to run the forward pass inside the Docker container

param (
    [string]$UserId = "1",
    [int]$NItems = 5
)

Write-Host "Running forward pass for user_id=$UserId with n_items=$NItems"

# Copy the script to the container
Write-Host "Copying script to container..."
docker cp .\run_forward_pass.py ecommerce-ml-prediction-service-1:/app/

# Run the script inside the container
Write-Host "Executing forward pass..."
docker exec ecommerce-ml-prediction-service-1 python /app/run_forward_pass.py --user-id "$UserId" --n-items "$NItems"

Write-Host "Forward pass completed."
