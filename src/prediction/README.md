# Prediction Service

This directory contains the FastAPI-based prediction service for serving real-time recommendations in the e-commerce ML pipeline.

## Components

### API Service (`app.py`)

Implements the FastAPI application with the following endpoints:
- `/recommendations/{user_id}`: Get personalized recommendations for a user
- `/event`: Record user interaction events (views, add-to-cart, purchases)
- `/health`: Service health check endpoint
- `/metrics`: Prometheus metrics endpoint

### Features

- **Real-time Recommendations**: Serves personalized product recommendations with low latency
- **Production/Development Modes**: Configurable modes with `DEVELOPMENT_MODE` and `PRODUCTION_READY` flags
- **Event Recording**: Captures user interaction events for model retraining
- **Error Handling**: Robust error handling with background retries for failed operations
- **Monitoring**: Prometheus metrics for service performance and recommendation quality

## Configuration

The service can be configured using environment variables:

```
# AWS Configuration
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-secret>
AWS_DEFAULT_REGION=eu-west-1

# Development/Production Flags
DEVELOPMENT_MODE=false  # Set to true for development environment
PRODUCTION_READY=true   # Set to true for production-quality recommendations
ALLOW_PICKLE=false      # Set to true only if needed for legacy models
```

## Docker Integration

The service is containerized using Docker, with configuration in the `docker/prediction/` directory. The service can be updated and restarted using the `update_aws_credentials.ps1` or `update_aws_credentials.sh` scripts in the `docker/` directory.
