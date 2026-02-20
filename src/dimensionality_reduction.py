"""
Dimensionality reduction for quantum-suitable features.
Reduces features to 8-20 dimensions for quantum circuits.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple


class DimensionalityReducer:
    """Reduce feature dimensions for quantum circuits."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()  # Add StandardScaler
        self.pca = None
        self.feature_selector = None
        self.quantum_n_features = config['dim_reduction']['quantum_features']['n_features']
        
    def fit_pca(self, features: np.ndarray) -> np.ndarray:
        """
        Apply PCA for initial dimensionality reduction.
        
        Args:
            features: (num_samples, num_features)
            
        Returns:
            Reduced features (num_samples, pca_components)
        """
        n_components = self.config['dim_reduction']['pca']['n_components']
        variance_threshold = self.config['dim_reduction']['pca']['variance_threshold']
        
        # Adaptive: PCA components can't exceed num_samples - 1 or num_features
        max_components = min(features.shape[0] - 1, features.shape[1], n_components)
        
        self.pca = PCA(n_components=max_components)
        features_pca = self.pca.fit_transform(features)
        
        # Check explained variance
        explained_var = np.cumsum(self.pca.explained_variance_ratio_)
        n_components_needed = np.argmax(explained_var >= variance_threshold) + 1
        
        print(f"PCA: {n_components_needed} components explain {variance_threshold*100}% variance")
        print(f"Reduced from {features.shape[1]} to {features_pca.shape[1]} features")
        
        return features_pca
    
    def transform_pca(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted PCA."""
        return self.pca.transform(features)
    
    def select_quantum_features(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Select most informative features for quantum circuit.
        
        Args:
            features: (num_samples, pca_components)
            labels: (num_samples,)
            
        Returns:
            Selected features (num_samples, quantum_n_features)
        """
        selection_method = self.config['dim_reduction']['quantum_features']['selection_method']
        
        if selection_method == 'mutual_info':
            # Mutual information-based selection
            n_features = features.shape[1]
            k = min(self.quantum_n_features, n_features)
            
            print(f"DEBUG: select_quantum_features input shape: {features.shape}")
            print(f"DEBUG: quantum_n_features config: {self.quantum_n_features}")
            print(f"DEBUG: Calculated k: {k}")
            
            if k < self.quantum_n_features:
                print(f"Warning: Reduced k from {self.quantum_n_features} to {k} due to limited features")
            
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif,
                k=k
            )
            features_selected = self.feature_selector.fit_transform(features, labels)
            
        elif selection_method == 'random_forest':
            # Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, labels)
            
            importances = rf.feature_importances_
            top_indices = np.argsort(importances)[-self.quantum_n_features:]
            features_selected = features[:, top_indices]
            
            self.feature_selector = top_indices  # Store indices
            
        else:
            # Default: take first n features
            features_selected = features[:, :self.quantum_n_features]
            self.feature_selector = None
        
        print(f"Selected {self.quantum_n_features} features for quantum circuit")
        print(f"Final feature shape: {features_selected.shape}")
        
        return features_selected
    
    def transform_quantum_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted selector."""
        if self.feature_selector is None:
            return features[:, :self.quantum_n_features]
        elif isinstance(self.feature_selector, np.ndarray):
            # Random forest indices
            return features[:, self.feature_selector]
        else:
            # SelectKBest
            return self.feature_selector.transform(features)
    
    def normalize_for_quantum(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, π] for quantum encoding.
        
        Args:
            features: (num_samples, quantum_n_features)
            
        Returns:
            Normalized features in [0, π]
        """
        # Min-max normalization to [0, 1]
        features_min = features.min(axis=0, keepdims=True)
        features_max = features.max(axis=0, keepdims=True)
        features_norm = (features - features_min) / (features_max - features_min + 1e-8)
        
        # Scale to [0, π]
        features_norm = features_norm * np.pi
        
        return features_norm
    
    def fit_transform_pca_only(self, features: np.ndarray) -> np.ndarray:
        """
        Scaling -> PCA only (no SelectKBest, no quantum normalization).
        Returns all PCA components so the hybrid model gets the full classical signal.
        
        Args:
            features: (num_samples, num_features)
            
        Returns:
            PCA-reduced features (num_samples, pca_components)
        """
        features_scaled = self.scaler.fit_transform(features)
        print(f"Applied StandardScaler to {features.shape[1]} features")
        features_pca = self.fit_pca(features_scaled)
        print(f"PCA-only output: {features_pca.shape} (all components kept for hybrid model)")
        return features_pca
    
    def transform_pca_only(self, features: np.ndarray) -> np.ndarray:
        """Transform new features using fitted scaler + PCA only."""
        features_scaled = self.scaler.transform(features)
        return self.transform_pca(features_scaled)
    
    def fit_transform(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Full pipeline: Scaling -> PCA -> Feature Selection -> Normalization.
        
        Args:
            features: (num_samples, num_features)
            labels: (num_samples,)
            
        Returns:
            Quantum-ready features (num_samples, quantum_n_features) in [0, π]
        """
        # Step 1: StandardScaler (NEW - critical for SVM convergence)
        features_scaled = self.scaler.fit_transform(features)
        print(f"Applied StandardScaler to {features.shape[1]} features")
        
        # Step 2: PCA
        features_pca = self.fit_pca(features_scaled)
        
        # Step 3: Feature selection
        features_selected = self.select_quantum_features(features_pca, labels)
        
        # Step 4: Normalization for quantum
        features_quantum = self.normalize_for_quantum(features_selected)
        
        return features_quantum
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform new features using fitted reducers."""
        features_scaled = self.scaler.transform(features)
        features_pca = self.transform_pca(features_scaled)
        features_selected = self.transform_quantum_features(features_pca)
        features_quantum = self.normalize_for_quantum(features_selected)
        return features_quantum
