# E-commerce ML Pipeline

Real-time e-commerce recommendation system using collaborative filtering and learning-to-rank models.

## Architecture

### Components
- **Data Pipeline**: Airflow DAGs for data processing and model training
- **Prediction Service**: FastAPI service for real-time recommendations
- **Monitoring Stack**: Prometheus and Grafana for observability

### Models
- **ALS (Alternating Least Squares)**
  - Collaborative filtering for candidate generation
  - User and item latent factors
  - Efficient sparse matrix implementation

- **CatBoost Ranker**
  - Learning-to-rank with YetiRank loss
  - Conformal prediction for uncertainty estimates
  - Feature importance analysis

### Features
- Advanced user behavior analysis
- Session-based features
- Temporal patterns
- Category tree analysis
- Price-based signals

## Project Structure
```
ecommerce-ml-pipeline/
├── airflow/                 # Airflow DAGs and configs
├── data/                    # Data storage (gitignored)
│   ├── raw/                # Raw RetailRocket data
│   └── processed/          # Processed features
├── docker/                 # Docker configurations
│   ├── airflow/           # Airflow service
│   ├── monitoring/        # Prometheus & Grafana
│   ├── prediction/        # FastAPI service
│   └── requirements/      # Python dependencies
├── models/                # Model artifacts
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── model/           # Model implementations
│   ├── processing/      # Data processing
│   ├── prediction/      # FastAPI service
│   └── monitoring/      # Metrics collection
└── infrastructure/      # IaC and K8s configs
```

## Setup

### Prerequisites
- Docker and Docker Compose
- Python 3.10+
- AWS account with S3 access

### Environment Variables
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-secret>
AWS_DEFAULT_REGION=eu-west-1

# Development/Production Flags
DEVELOPMENT_MODE=false  # Set to true for development environment
PRODUCTION_READY=true   # Set to true for production-quality recommendations
ALLOW_PICKLE=false      # Set to true only if needed for legacy models

# Grafana (optional)
GRAFANA_ADMIN_PASSWORD=<your-password>
```

### Quick Start
1. Clone the repository
2. Set up environment variables
3. Start the services:
   ```bash
   docker compose up -d
   ```
4. Access the services:
   - Prediction API: http://localhost:8000
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000

### Production Deployment
1. Update AWS credentials and configure production mode:
   ```bash
   cd docker
   ./update_aws_credentials.ps1  # On Windows
   # Or on Linux/Mac:
   # ./update_aws_credentials.sh
   ```
2. Follow the prompts to enter your AWS credentials and set production mode
3. The script will automatically:
   - Update the docker-compose.yml file with your credentials
   - Set the PRODUCTION_READY flag based on your selection
   - Restart the prediction service with the new configuration
   - Test the service to ensure it's working correctly

4. Verify the deployment:
   - Check the health endpoint: http://localhost:8000/health
   - Test recommendations: http://localhost:8000/recommendations
   - Monitor logs: `docker logs ecommerce-ml-prediction-service-1`

## Data Pipeline

### Data Sources
- RetailRocket e-commerce dataset
- Stored in AWS S3: `ecommerce-ml-pipeline-data`
- Raw data files:
  - `events.csv`
  - `item_properties.csv`
  - `category_tree.csv`

### Feature Engineering
- Event data cleaning
- Item properties processing
- Category tree handling
- User feature engineering
- Item feature engineering
- Interaction matrix creation

### Model Training
1. Data preprocessing
2. ALS model training
3. Candidate generation
4. CatBoost ranking model
5. Model evaluation and storage

## Monitoring

### Metrics
- Prediction latency
- Request throughput
- Feature update status
- Model performance
- System health

### Dashboards
- Real-time prediction metrics
- Model performance tracking
- Feature drift detection
- System resource usage

## API Endpoints

### Prediction Service
- `POST /recommendations`
  - Get personalized recommendations
  - Parameters:
    - `user_id`: User identifier
    - `n_items`: Number of items (default: 10)
    - `include_metadata`: Include item details
    - `min_confidence`: Minimum prediction confidence

- `POST /events`
  - Record user interactions
  - Parameters:
    - `user_id`: User identifier
    - `item_id`: Item identifier
    - `event_type`: Interaction type
    - `timestamp`: Event timestamp

- `GET /health`
  - Service health check
  - Returns:
    - Model version
    - Feature freshness
    - System status

- `GET /metrics`
  - Prometheus metrics endpoint

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License
MIT License
