"""
Traditional machine learning models
(Random Forest, XGBoost, SVM, MLP)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
import joblib
from typing import Dict, Optional


class TraditionalMLClassifier:
    """
    Wrapper for traditional ML models

    Supports:
        - Random Forest
        - XGBoost (if installed)
        - Gradient Boosting
        - SVM
        - MLP (sklearn neural network)
    """

    def __init__(
        self,
        model_type: str = 'random_forest',
        **kwargs
    ):
        """
        Parameters:
            model_type: 'random_forest', 'xgboost', 'gradient_boosting', 'svm', 'mlp'
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 20),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1)
            )

        elif model_type == 'xgboost':
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Install with: pip install xgboost")
            self.model = xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1)
            )

        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42)
            )

        elif model_type == 'svm':
            self.model = SVC(
                C=kwargs.get('C', 10),
                kernel=kwargs.get('kernel', 'rbf'),
                gamma=kwargs.get('gamma', 'scale'),
                random_state=kwargs.get('random_state', 42),
                probability=True  # Enable probability estimates
            )

        elif model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (64, 32)),
                activation=kwargs.get('activation', 'relu'),
                max_iter=kwargs.get('max_iter', 500),
                random_state=kwargs.get('random_state', 42)
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, scale: bool = True):
        """
        Train model

        Parameters:
            X: Feature matrix, shape (N, num_features)
            y: Labels, shape (N,)
            scale: Whether to scale features
        """
        if scale:
            X = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Predict labels"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if scale:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if scale:
            X = self.scaler.transform(X)

        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray, scale: bool = True) -> float:
        """Calculate accuracy score"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if scale:
            X = self.scaler.transform(X)

        return self.model.score(X, y)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance (for tree-based models)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None

    def grid_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict,
        cv: int = 5,
        scale: bool = True
    ):
        """
        Perform grid search for hyperparameter tuning

        Parameters:
            X: Feature matrix
            y: Labels
            param_grid: Parameter grid
            cv: Number of cross-validation folds
            scale: Whether to scale features
        """
        if scale:
            X = self.scaler.fit_transform(X)

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True

        return grid_search.best_params_, grid_search.best_score_

    def save_model(self, filepath: str):
        """Save model to file"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }, filepath)

    def load_model(self, filepath: str):
        """Load model from file"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data['model_type']
        self.is_fitted = data['is_fitted']
