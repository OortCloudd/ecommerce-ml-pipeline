# Changelog - E-commerce ML Pipeline Setup

## [Unreleased]

### Added
- Real-time prediction service with FastAPI
  - ALS model integration for candidate generation
  - CatBoost ranking model with confidence intervals
  - Real-time feature updates from S3
  - Event tracking and logging
  - Prometheus metrics integration

- Comprehensive monitoring stack
  - Prometheus server for metrics collection
  - Grafana dashboards for visualization
  - Custom metrics for model performance
  - Feature update monitoring
  - Request latency tracking

### Changed
- Updated project structure for better organization
  - Separated Docker configurations by service
  - Organized requirements by component
  - Improved code modularity

### Fixed
- Docker service configurations
  - Added proper health checks
  - Implemented secure user permissions
  - Fixed volume mounts for data persistence

## [1.0.0] - 2025-03-07

### Added
- Initial project setup
- Data processing pipeline for RetailRocket dataset
- AWS S3 integration
  - Raw data storage
  - Processed features
  - Model artifacts
- Airflow DAG implementation
  - Data extraction from S3
  - Feature engineering
  - Model training
  - Result upload
- Model implementations
  - ALS for collaborative filtering
  - CatBoost for ranking
  - Conformal prediction for uncertainty

### Changed
- Organized project structure
  - Source code organization
  - Docker configurations
  - Documentation

### Security
- Implemented secure configurations
  - Non-root container users
  - Environment variable management
  - AWS credential handling

## [2025-03-07] - Project Setup and Data Management

### Added
1. Project Structure:
   - Created essential directories for Airflow (dags, logs, plugins, config)
   - Set up data directories (raw, processed)
   - Added utility scripts in src/utils/

2. Docker Configuration:
   - Implemented docker-compose.yaml for Airflow services
   - Created airflow.env for environment variables
   - Successfully deployed Airflow with PostgreSQL backend

3. Data Management:
   - Added RetailRocket dataset to data/raw/
   - Created data sampling utility (src/utils/create_samples.py)
   - Generated 1000-row samples for all dataset files in data/raw/samples/
   - Added AWS integration utilities (src/utils/s3_utils.py)
   - Added AWS credentials configuration script (src/utils/configure_aws.py)

4. Dependencies:
   - Updated requirements.txt with necessary packages:
     - apache-airflow and AWS provider
     - pandas for data processing
     - boto3 for AWS integration
     - psycopg2-binary for PostgreSQL

### Technical Decisions
1. Docker Implementation:
   - Chose Docker for consistent development environment
   - Set up multi-container architecture with PostgreSQL
   - Implemented custom initialization script

2. Data Strategy:
   - Decided to keep sample data in git for development
   - Plan to store full dataset in S3 (pending AWS setup)
   - Implemented .gitignore rules to exclude large data files

3. Development Workflow:
   - Created utility scripts for repeatable operations
   - Set up local development with sample data
   - Prepared AWS integration tools

### Current Status
- Airflow running in Docker
- Sample data generated and stored
- AWS utilities prepared
- AWS S3 setup pending
- Full dataset upload pending

### Next Steps
1. Complete AWS Setup:
   - Create IAM user with S3 access
   - Configure AWS credentials
   - Create S3 bucket
   - Upload full dataset

2. Data Pipeline Development:
   - Create first Airflow DAG
   - Implement data processing logic
   - Set up model training pipeline

### Notes
- All configuration follows industry best practices
- Infrastructure defined as code (IaC)
- Development environment mirrors production setup

## [2025-03-07] - AWS Setup and Dataset Upload

### Added
1. AWS Configuration:
   - Created IAM user with S3 access
   - Implemented AWS credentials configuration script
   - Added region handling in configuration utilities

2. S3 Integration:
   - Created S3 bucket (ecommerce-ml-pipeline-data) in eu-west-1
   - Enhanced s3_utils.py with better region handling
   - Added AWS region detection from credentials

3. Dataset Upload:
   - Successfully uploaded RetailRocket dataset to S3:
     - category_tree.csv (14.4 KB)
     - events.csv (94.2 MB)
     - item_properties_part1.csv (484.3 MB)
     - item_properties_part2.csv (408.9 MB)
   - Organized under 'raw/' prefix in S3

### Technical Decisions
1. AWS Configuration:
   - Used eu-west-1 (Ireland) region for lower latency
   - Implemented proper UTF-8 encoding for credentials
   - Added automatic region detection from credentials

2. Data Organization:
   - Used 'raw/' prefix for unprocessed data
   - Uploaded unzipped CSV files directly
   - Maintained original file structure

### Current Status
- AWS credentials configured
- S3 bucket created and accessible
- Full dataset uploaded to S3
- Local development environment ready

### Next Steps
1. Data Pipeline Development:
   - Create first Airflow DAG
   - Implement data processing logic
   - Set up model training pipeline

## [2025-03-07] - Model Training Pipeline Development

### Added
- Implemented ALS (Alternating Least Squares) model for collaborative filtering
  - Added ALSTrainer class with training, prediction, and model persistence
  - Support for user-item recommendations and similar items
  
- Implemented CatBoost ranking model
  - Added RankingTrainer class with YetiRank loss function
  - Support for group-wise learning and evaluation metrics (NDCG, MAP)
  
- Created model training pipeline DAG with tasks:
  1. load_data: Load processed data from S3
  2. train_als: Train and save ALS model
  3. generate_candidates: Generate candidate items using ALS
  4. train_ranker: Train and save ranking model
  
- Added new dependencies:
  - implicit for ALS implementation
  - catboost for ranking model
  - scikit-learn for data splitting
  - scipy for sparse matrix operations

## [2025-03-07] - Model Training Pipeline Enhancements

### Added
- Enhanced ranking model with Optuna for Bayesian optimization
- Added conformal prediction support using MAPIE
- Improved model reliability with uncertainty quantification
- Updated dependencies in requirements.txt and Dockerfile

## [2025-03-07] - Advanced Feature Engineering

### Added
- Advanced feature engineering module with sophisticated e-commerce features:
  - Temporal features with multi-window analysis (1/7/30 days)
  - Session-based features with configurable timeout
  - Price-based features with category-relative analysis
  - Category graph analysis using NetworkX
  - Sequence analysis with transition probabilities
  - Time-decay interaction scoring

### Enhanced
- DataProcessor with improved data cleaning:
  - Proper type conversion for IDs
  - Numeric column handling
  - Enhanced conversion rate calculations
  - Integration with advanced features

### Dependencies
- Added NetworkX for category graph analysis
- Added tqdm for progress tracking

## [2025-03-07] - Real-time Prediction Service

### Added
- FastAPI service for real-time recommendations:
  - Real-time and batch recommendation endpoints
  - Event ingestion API
  - Health check and metrics endpoints
  - Confidence interval support from MAPIE
  - Background feature updates

- Prediction Service Features:
  - Feature caching with configurable update intervals
  - Asynchronous feature computation
  - Model version management
  - Combined ALS and CatBoost predictions
  - Real-time feature engineering

### Dependencies
- Added FastAPI and uvicorn for API service
- Added pydantic for data validation
- Added prometheus-client for metrics

### Infrastructure
- Added Dockerfile for prediction service
- Configured environment variables for deployment
- Added cache and model volume mounts

## [2025-03-07] - AWS EKS Deployment

### Added
- Kubernetes deployment configuration:
  - 3 replicas with rolling updates
  - Resource requests and limits
  - Health checks and readiness probes
  - Horizontal Pod Autoscaler (3-10 pods)
  - EFS storage for models and cache

- AWS Infrastructure:
  - EKS cluster integration
  - EFS for shared model storage
  - LoadBalancer service type
  - Prometheus metrics support

### Enhanced
- FastAPI service improvements:
  - OpenAPI documentation
  - Error handling and validation
  - Service metrics and health checks
  - Type hints and response models

### Dependencies
- Updated requirements.txt with exact versions
- Added prometheus-client for metrics

### Next Steps
- Set up AWS EFS storage class
- Configure AWS Load Balancer Controller
- Implement CI/CD pipeline
- Add monitoring with Prometheus/Grafana

## [2025-03-07] - Model Monitoring Implementation

### Added
- Model monitoring service:
  - Real-time performance tracking
  - Data drift detection using KS test
  - Automated retraining triggers
  - S3 integration for metrics storage

- Monitoring components:
  - MetricsCollector for data collection
  - MonitoringService for metrics processing
  - FastAPI endpoints for monitoring
  - Kubernetes deployment configs

### Enhanced
- Project infrastructure:
  - EFS storage for metrics persistence
  - Prometheus metrics integration
  - Service health checks
  - Background tasks for metrics collection

### Dependencies
- Added scipy for statistical tests
- Added boto3 for AWS integration

### Next Steps
- Implement CI/CD pipeline
- Set up A/B testing framework
- Configure Prometheus/Grafana dashboards
- Complete AWS infrastructure setup

## [2025-03-07] - Service and Docker Configuration Updates

### Added
1. Docker and Service Configuration Updates:
   - Updated `docker-compose.yml` to remove obsolete version field and add proper service dependencies.
   - Created `requirements.service.txt` to separate service dependencies from Airflow dependencies.
   - Updated `Dockerfile.service` and `Dockerfile.monitoring` to use `requirements.service.txt`.

2. Service Enhancements:
   - Implemented health checks for prediction and monitoring services.
   - Improved Docker build process by resolving dependency conflicts.

3. Next Steps:
   - Verify service health and operational status.
   - Monitor logs for any runtime errors.
   - Ensure all environment variables are correctly set for AWS and local operations.

## [2025-03-07] - Project Structure Reorganization and Docker Services

### Added
1. Docker Service Organization:
   - Created dedicated docker/ directory with clear service separation:
     - airflow/: Airflow service with Python 3.10
     - prediction/: FastAPI prediction service
     - monitoring/: Prometheus monitoring service
     - requirements/: Service-specific dependencies

2. Requirements Organization:
   - Split requirements into modular files:
     - base.txt: Core ML and data processing
     - airflow.txt: Airflow service dependencies
     - prediction.txt: FastAPI service dependencies
     - monitoring.txt: Prometheus and metrics

3. Project Structure:
   - Organized according to best practices:
     - airflow/: DAGs and configurations
     - data/: Raw and processed data
     - docker/: Service configurations
     - docs/: Documentation
     - infrastructure/: Terraform and K8s
     - models/: Model artifacts
     - notebooks/: Jupyter notebooks
     - src/: Main source code
     - tests/: Test suite

### Technical Decisions
1. Service Architecture:
   - Python 3.10 as base for all services
   - Separated concerns between services
   - Modular dependency management
   - Prometheus for monitoring

2. Development Workflow:
   - Clear separation of development and production
   - Service-specific configurations
   - Centralized Docker management

### Current Status (60% Complete)
✅ Completed:
- Project structure and organization
- Docker service configurations
- Requirements management
- Basic service setup (Airflow, Prediction, Monitoring)
- AWS S3 data setup and upload
- Data processing pipeline implementation
- Airflow DAG for data processing

🔄 In Progress:
- Model training setup
- Service deployment
- Monitoring setup

❌ Pending:
- Real-time prediction service implementation
- Model training pipeline optimization
- Production deployment
- Grafana dashboards

### Next Steps
1. Model Training Pipeline:
   - Optimize ALS and CatBoost training
   - Implement model versioning
   - Add model evaluation metrics

2. Service Development:
   - Complete FastAPI prediction service
   - Add health checks and metrics
   - Set up Prometheus monitoring
   - Add Grafana dashboards

3. Testing and Documentation:
   - Add unit tests for core components
   - Write API documentation
   - Create deployment guides

### Notes
- All services configured for Python 3.10
- Infrastructure ready for AWS deployment
- Data pipeline operational with RetailRocket dataset
- Monitoring strategy in place with Prometheus
