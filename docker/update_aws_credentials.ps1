# Script to update AWS credentials and restart the prediction service

# Prompt for AWS credentials
$AWS_ACCESS_KEY_ID = Read-Host -Prompt "Enter your AWS Access Key ID"
$AWS_SECRET_ACCESS_KEY = Read-Host -Prompt "Enter your AWS Secret Access Key" -AsSecureString
$AWS_DEFAULT_REGION = Read-Host -Prompt "Enter your AWS Region (default: eu-west-1)" 

# Set default region if not provided
if ([string]::IsNullOrWhiteSpace($AWS_DEFAULT_REGION)) {
    $AWS_DEFAULT_REGION = "eu-west-1"
}

# Convert secure string to plain text
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($AWS_SECRET_ACCESS_KEY)
$AWS_SECRET_ACCESS_KEY_PLAIN = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

# Set environment variables for the current session
$env:AWS_ACCESS_KEY_ID = $AWS_ACCESS_KEY_ID
$env:AWS_SECRET_ACCESS_KEY = $AWS_SECRET_ACCESS_KEY_PLAIN
$env:AWS_DEFAULT_REGION = $AWS_DEFAULT_REGION

# Ask if this is a production deployment
$production = Read-Host -Prompt "Is this a production deployment? (y/n)"
$PRODUCTION_READY = $production.ToLower() -eq "y"

# Update docker-compose.yml with environment variables
Write-Host "Updating docker-compose.yml with AWS credentials..."
$composePath = "$PSScriptRoot\docker-compose.yml"
$composeContent = Get-Content $composePath -Raw

# Replace environment variables in docker-compose.yml
$composeContent = $composeContent -replace '(\s+AWS_ACCESS_KEY_ID:).*', "$1 $AWS_ACCESS_KEY_ID"
$composeContent = $composeContent -replace '(\s+AWS_SECRET_ACCESS_KEY:).*', "$1 $AWS_SECRET_ACCESS_KEY_PLAIN"
$composeContent = $composeContent -replace '(\s+AWS_DEFAULT_REGION:).*', "$1 $AWS_DEFAULT_REGION"

# Update PRODUCTION_READY flag
if ($PRODUCTION_READY) {
    $composeContent = $composeContent -replace '(\s+PRODUCTION_READY:).*', "$1 'true'"
    Write-Host "Setting PRODUCTION_READY to true for production deployment"
} else {
    $composeContent = $composeContent -replace '(\s+PRODUCTION_READY:).*', "$1 'false'"
    Write-Host "Setting PRODUCTION_READY to false for development deployment"
}

# Save updated docker-compose.yml
Set-Content -Path $composePath -Value $composeContent

# Restart the prediction service
Write-Host "Stopping the prediction service..."
docker-compose -f $composePath down

Write-Host "Starting the prediction service with updated AWS credentials..."
docker-compose -f $composePath up -d --build prediction-service

Write-Host "Prediction service restarted with updated AWS credentials."
Write-Host "Checking service status..."

# Wait a few seconds for the service to start
Start-Sleep -Seconds 5

# Check service logs
docker logs ecommerce-ml-prediction-service-1

Write-Host ""
Write-Host "AWS credentials have been updated. You can now use the prediction service with your AWS credentials."
Write-Host "Production mode: $PRODUCTION_READY"

# Test the service
$testService = Read-Host -Prompt "Do you want to test the service? (y/n)"
if ($testService.ToLower() -eq "y") {
    Write-Host "Testing health endpoint..."
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
        Write-Host "Health check successful: $($health | ConvertTo-Json -Depth 1)"
        
        Write-Host "Testing recommendations endpoint..."
        $recommendationBody = @{
            user_id = "1"
            n_items = 5
            include_metadata = $true
        } | ConvertTo-Json
        
        $recommendations = Invoke-RestMethod -Uri "http://localhost:8000/recommendations" -Method Post -Body $recommendationBody -ContentType "application/json"
        Write-Host "Recommendations received: $($recommendations.recommendations.Count) items"
        
        Write-Host "Service is working correctly!"
    } catch {
        Write-Host "Error testing service: $_"
    }
}
