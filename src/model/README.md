# Model Implementation

This directory contains the model implementations for the e-commerce recommendation system, including collaborative filtering, ranking, and forward pass visualization.

## Components

### Collaborative Filtering (`collaborative/`)

- **ALS Trainer**: Implements Alternating Least Squares for matrix factorization
- Handles user and item latent factor generation
- Provides methods for model saving/loading and recommendation generation

### Ranking (`ranking/`)

- **Conformal Ranking Trainer**: Implements CatBoost-based ranking with uncertainty estimates
- Provides calibrated prediction intervals using conformal prediction
- Includes feature importance analysis

### Forward Pass (`forward_pass.py`)

Demonstrates the step-by-step process of generating recommendations:
- Loads ALS and CatBoost ranking models
- Performs candidate generation using collaborative filtering
- Re-ranks candidates with the ranking model
- Handles error cases gracefully with fallbacks
- Formats final recommendations with confidence intervals

### Visualization (`visualize_forward_pass.py`)

Creates visual representations of the recommendation process:
- Visualizes ALS latent factors for users and items
- Shows the transition from ALS scores to final rankings
- Displays confidence intervals for predictions
- Visualizes feature importance in the ranking model

### Metrics (`metrics/`)

- Implements evaluation metrics for recommendation quality
- Includes ranking metrics (NDCG, MRR) and classification metrics (precision, recall)

## Usage

See the detailed usage instructions in [README_FORWARD_PASS.md](./README_FORWARD_PASS.md) for information on running the forward pass and visualization tools.
