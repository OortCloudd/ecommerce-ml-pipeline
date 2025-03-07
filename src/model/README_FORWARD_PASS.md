# Recommendation System Forward Pass

This directory contains tools to demonstrate and visualize the forward pass of our e-commerce recommendation system, showing how data flows through the ALS collaborative filtering model and the CatBoost ranking model to generate personalized recommendations.

## Components

### 1. Forward Pass Implementation (`forward_pass.py`)

This script implements the core forward pass logic for the recommendation system:

- **Model Loading**: Loads the trained ALS and ranking models from disk
- **Feature Loading**: Loads user and item features from processed data files
- **Candidate Generation**: Uses the ALS model to generate candidate items based on user-item similarity
- **Re-ranking**: Applies the ranking model to re-rank candidates based on additional features
- **Confidence Intervals**: Provides confidence intervals for recommendations using conformal prediction

### 2. Forward Pass Visualization (`visualize_forward_pass.py`)

This script creates visualizations to help understand the recommendation process:

- **ALS Factor Visualization**: Shows the latent factors for users and items
- **Recommendation Process Visualization**: Displays the transition from ALS scores to final rankings
- **Feature Importance Visualization**: Shows which features have the most impact on the ranking model

### 3. Docker Integration (`run_forward_pass.py`)

This script is designed to run inside the Docker container to demonstrate the forward pass with production models:

- **Environment-aware**: Uses environment variables to locate models and features
- **Detailed Logging**: Provides step-by-step logging of the recommendation process
- **Error Handling**: Gracefully handles missing models or features with appropriate fallbacks

## Usage

### Visualizing the Forward Pass

```bash
python visualize_forward_pass.py --user-id 1 --model-path ./models --feature-path ./data/processed
```

This will generate three visualization files:
- `als_factors_visualization.png`: Shows the user's latent factors and top items' factors
- `recommendation_process_visualization.png`: Shows the transition from ALS to ranking scores
- `feature_importance_visualization.png`: Shows the importance of each feature in the ranking model

### Running Forward Pass in Docker

On Windows (PowerShell):
```powershell
.\run_forward_pass.ps1 -UserId 1 -NItems 10
```

On Linux/Mac:
```bash
./run_forward_pass.sh --user-id 1 --n-items 10
```

This will run the forward pass inside the Docker container and display the recommendations in JSON format.

## Understanding the Output

The recommendation output includes:

- **item_id**: Unique identifier for the recommended item
- **score**: Predicted score/relevance for the user
- **confidence_lower/upper**: Confidence interval bounds for the prediction
- **metadata**: Additional item information (category, price, popularity)

## Extending the Forward Pass

To extend the forward pass implementation:

1. **Add New Features**: Incorporate additional user or item features in the ranking model
2. **Implement New Models**: Add new candidate generation or ranking models
3. **Customize Visualizations**: Create additional visualizations for specific aspects of the model

## Troubleshooting

- **Missing Models**: Ensure models are trained and saved in the correct location
- **Feature Format**: Verify that feature files are in the expected format (Parquet)
- **Docker Environment**: Check that environment variables are correctly set in the Docker container

## Related Components

- **Data Processing Pipeline**: Generates features used by the recommendation system
- **Model Training Pipeline**: Trains the ALS and ranking models
- **Prediction Service**: Serves recommendations via API endpoints
