"""
Classical baseline models for comparison.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from typing import Dict
import pickle


class ClassicalBaseline:
    """Wrapper for classical machine learning baselines."""
    
    def __init__(self, config: Dict, model_type: str = 'random_forest'):
        """
        Args:
            config: Configuration dictionary
            model_type: 'random_forest', 'xgboost', or 'mlp'
        """
        self.config = config
        self.model_type = model_type
        self.model = self._create_model()
        
    def _create_model(self):
        """Create the specified model."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=3,  # Adjusted for 3:1 class imbalance (3501 neg / 1195 pos)
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.01,  # L2 regularization
                batch_size=32,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train the model."""
        print(f"Training {self.model_type} baseline...")
        
        if self.model_type == 'xgboost' and X_val is not None:
            # Use validation set for early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        print(f"{self.model_type} training complete")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X_test)
    
    def save(self, path: str):
        """Save model."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: str):
        """Load model."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
