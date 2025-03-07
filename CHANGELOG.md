# Changelog

All notable changes to the E-commerce ML Pipeline project will be documented in this file.

## [1.0.0] - 2025-03-07

### Added
- Forward pass implementation for recommendation visualization
  - Added `forward_pass.py` to demonstrate the recommendation process
  - Created `visualize_forward_pass.py` for interactive visualizations
  - Added Docker integration scripts for running in containers
- Production deployment scripts
  - Created `update_aws_credentials.ps1` for Windows users
  - Created `update_aws_credentials.sh` for Linux/Mac users
  - Added support for PRODUCTION_READY flag configuration
- Enhanced recommendation service
  - Improved error handling for event recording
  - Added background retries for failed S3 uploads
  - Enhanced dummy recommendations for development mode
- Comprehensive documentation
  - Added detailed README files for key components
  - Created documentation for forward pass visualization
  - Updated main README with production deployment instructions

### Changed
- Consolidated data processing code
  - Merged `processing/` and `data_processing/` directories
  - Improved organization of feature engineering code
- Standardized project structure
  - Added consistent README files across directories
  - Improved naming conventions for better clarity
  - Enhanced documentation for key components

### Fixed
- Resolved model loading issues in Docker environment
- Fixed error handling in recommendation service
- Addressed AWS credential management security concerns

## [0.9.0] - 2025-03-01

### Added
- Initial implementation of recommendation system
  - ALS collaborative filtering for candidate generation
  - CatBoost ranking model with conformal prediction
- Data processing pipeline for RetailRocket dataset
  - Event data cleaning and preprocessing
  - Feature engineering for users and items
  - Interaction matrix creation for collaborative filtering
- Docker containerization
  - FastAPI prediction service
  - Airflow for orchestration
  - Prometheus and Grafana for monitoring
- AWS integration
  - S3 storage for data and models
  - Credential management
  - Event logging
