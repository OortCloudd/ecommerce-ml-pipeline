"""
Module for model training and evaluation
"""
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

class ModelTrainer:
    def __init__(self):
        self.model = None

    def train(self, X, y, test_size=0.2):
        """Train the model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.model = CatBoostRegressor(verbose=False)
        self.model.fit(X_train, y_train)
        
        return self.model.score(X_test, y_test)

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
