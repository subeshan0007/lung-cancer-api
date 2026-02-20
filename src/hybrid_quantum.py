"""
Hybrid Classical-Quantum Model (Enhanced)
Combines classical preprocessing with an upgraded quantum feature encoder.

Architecture:
    Classical (85%): ResNet3D → PCA → Feature Selection → SVM
    Quantum (15%): Enhanced quantum encoder for 12 selected features
    
Improvements over v1:
    - 6 qubits, 5 layers (was 4 qubits, 3 layers)
    - Multi-basis measurements: PauliX, PauliY, PauliZ → 18 quantum features (was 8)
    - Data reuploading per layer
    - Trainable variational parameters
    - SMOTE oversampling for class imbalance
    - GridSearchCV for SVM hyperparameter tuning
    - Optimized decision threshold
    - Quantum-classical feature interaction terms
"""
import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
import pickle


class QuantumFeatureEncoder:
    """
    Enhanced quantum circuit for feature encoding.
    
    Uses a deeper circuit (6 qubits, 5 layers) with:
    - Data reuploading per layer
    - Trainable variational parameters
    - Multi-basis measurements (PauliX, PauliY, PauliZ)
    → Produces 18 quantum features (6 qubits × 3 measurement bases)
    """
    
    def __init__(self, n_qubits=6, n_layers=5, n_input_features=12):
        """
        Initialize quantum feature encoder.
        
        Args:
            n_qubits: Number of qubits (default: 6)
            n_layers: Number of circuit layers (default: 5)
            n_input_features: Number of classical features to encode (default: 12)
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_input_features = n_input_features
        
        # Output: 3 measurement bases × n_qubits
        self.n_output_features = 3 * n_qubits  # 18 for 6 qubits
        
        # Trainable variational parameters
        # 3 rotations per qubit per layer
        self.n_params = n_qubits * n_layers * 3
        np.random.seed(42)
        self.params = np.random.randn(self.n_params) * 0.1
        
        # Create quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Create quantum circuit
        self.circuit = qml.QNode(self._quantum_circuit, self.dev)
        
        print(f"Initialized Quantum Feature Encoder: {n_qubits} qubits, {n_layers} layers")
        print(f"  Output features: {self.n_output_features} (3 bases × {n_qubits} qubits)")
    
    def _quantum_circuit(self, features, params):
        """
        Enhanced quantum circuit with data reuploading and trainable parameters.
        
        Circuit structure per layer:
        1. Data encoding: RY(feature) + RZ(feature) on each qubit
        2. Trainable rotations: RX(θ) + RY(θ) + RZ(θ) per qubit
        3. Entanglement: CNOT ring topology
        
        Args:
            features: Classical features to encode (length n_input_features)
            params: Trainable variational parameters
            
        Returns:
            List of expectation values (PauliX, PauliY, PauliZ on all qubits)
        """
        features_per_qubit = max(len(features) // self.n_qubits, 1)
        param_idx = 0
        
        for layer in range(self.n_layers):
            # === Data encoding layer (re-uploaded each layer) ===
            for i in range(self.n_qubits):
                idx = i * features_per_qubit
                
                if idx < len(features):
                    # Bound angles to [-π, π] using tanh
                    angle1 = np.pi * np.tanh(features[idx])
                    qml.RY(angle1, wires=i)
                    
                    if idx + 1 < len(features):
                        angle2 = np.pi * np.tanh(features[idx + 1])
                        qml.RZ(angle2, wires=i)
            
            # === Trainable variational layer ===
            for i in range(self.n_qubits):
                qml.RX(params[param_idx], wires=i)
                qml.RY(params[param_idx + 1], wires=i)
                qml.RZ(params[param_idx + 2], wires=i)
                param_idx += 3
            
            # === Entanglement layer (ring topology) ===
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])  # Close the ring
        
        # === Multi-basis measurements ===
        # PauliZ on all qubits
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        # PauliX on all qubits
        measurements += [qml.expval(qml.PauliX(i)) for i in range(self.n_qubits)]
        # PauliY on all qubits
        measurements += [qml.expval(qml.PauliY(i)) for i in range(self.n_qubits)]
        
        return measurements
    
    def encode(self, features):
        """
        Encode classical features using quantum circuit.
        
        Args:
            features: Array of shape (n_samples, n_input_features) or (n_input_features,)
            
        Returns:
            Quantum-encoded features: (n_samples, n_output_features) or (n_output_features,)
        """
        if features.ndim == 1:
            measurements = self.circuit(features, self.params)
            return np.array(measurements)
        else:
            encoded = []
            for sample in features:
                measurements = self.circuit(sample, self.params)
                encoded.append(np.array(measurements))
            return np.array(encoded)

    def train_params(self, X, y, n_epochs=30, lr=0.05, batch_size=32):
        """
        Optimise variational parameters via gradient descent.

        Objective: maximise separation between class-0 and class-1 quantum
        features (Fisher-like criterion on the quantum output space).

        Args:
            X: Selected features for quantum encoding (n_samples, n_input_features)
            y: Binary labels (n_samples,)
            n_epochs: Number of optimisation epochs
            lr: Learning rate for Adam optimiser
            batch_size: Mini-batch size
        """
        print(f"  Training quantum encoder params ({self.n_params} params, {n_epochs} epochs, lr={lr})...")

        # We use PennyLane's numpy wrapper for auto-differentiation
        from pennylane import numpy as pnp
        params = pnp.array(self.params, requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=lr)

        n = len(y)
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]

        rng = np.random.RandomState(42)

        def cost_fn(params, X_batch, y_batch):
            """Fisher-like loss: minimise intra-class variance, maximise inter-class distance."""
            encodings = []
            for sample in X_batch:
                meas = self.circuit(sample, params)
                encodings.append(pnp.array(meas))
            encodings = pnp.stack(encodings)

            mask_0 = y_batch == 0
            mask_1 = y_batch == 1

            if mask_0.sum() == 0 or mask_1.sum() == 0:
                return pnp.array(0.0)

            mean_0 = pnp.mean(encodings[mask_0], axis=0)
            mean_1 = pnp.mean(encodings[mask_1], axis=0)

            inter_class = pnp.sum((mean_0 - mean_1) ** 2)
            intra_0 = pnp.mean(pnp.sum((encodings[mask_0] - mean_0) ** 2, axis=1))
            intra_1 = pnp.mean(pnp.sum((encodings[mask_1] - mean_1) ** 2, axis=1))

            loss = (intra_0 + intra_1) / (inter_class + 1e-6)
            return loss

        best_loss = float('inf')
        best_params = params.copy()

        for epoch in range(n_epochs):
            # Balanced mini-batch: equal samples from each class
            half = batch_size // 2
            b0 = rng.choice(idx_0, size=min(half, len(idx_0)), replace=len(idx_0) < half)
            b1 = rng.choice(idx_1, size=min(half, len(idx_1)), replace=len(idx_1) < half)
            batch_idx = np.concatenate([b0, b1])
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            params, loss_val = opt.step_and_cost(
                lambda p: cost_fn(p, X_batch, y_batch), params
            )

            loss_val = float(loss_val)
            if loss_val < best_loss:
                best_loss = loss_val
                best_params = params.copy()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{n_epochs}: loss={loss_val:.4f} (best={best_loss:.4f})")

        self.params = np.array(best_params)
        print(f"  Quantum encoder training complete. Best loss: {best_loss:.4f}")


class HybridQuantumClassifier:
    """
    Enhanced Hybrid Classical-Quantum classifier.
    
    Pipeline:
        1. Classical features (from ResNet3D + PCA) → 50 features
        2. Select top 12 features for quantum encoding
        3. Quantum encode → 18 quantum features (6 qubits × 3 bases)
        4. Create interaction terms: quantum × top classical → ~18 more features
        5. Concatenate: [50 classical] + [18 quantum] + [18 interactions] = 86 total
        6. StandardScaler on hybrid features
        7. SMOTE oversampling for class imbalance
        8. GridSearchCV to tune SVM hyperparameters
        9. Optimize decision threshold for F1
    """
    
    def __init__(self, config):
        """
        Initialize hybrid quantum classifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Quantum encoder settings
        hybrid_config = config.get('hybrid', {})
        self.n_qubits = hybrid_config.get('n_qubits', 6)
        self.n_layers = hybrid_config.get('n_layers', 5)
        self.n_quantum_features = hybrid_config.get('n_quantum_features', 12)
        self.use_feature_interactions = hybrid_config.get('use_feature_interactions', True)
        self.use_smote = hybrid_config.get('use_smote', True)
        self.tune_svm = hybrid_config.get('tune_svm', True)
        self.optimize_threshold = hybrid_config.get('optimize_threshold', True)
        
        # Initialize quantum encoder
        self.quantum_encoder = QuantumFeatureEncoder(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_input_features=self.n_quantum_features
        )
        
        # Feature selector (will be fit during training)
        self.feature_selector = None
        
        # Scaler for hybrid features
        self.hybrid_scaler = StandardScaler()
        
        # Decision threshold (default 0.5, optimized during training)
        self.decision_threshold = 0.5
        
        # Top feature indices for interaction terms
        self.top_classical_indices = None
        
        # Classical SVM classifier (will be tuned or use defaults)
        self.classifier = None
        
        print(f"Hybrid Quantum Classifier initialized")
        print(f"  Quantum features: {self.n_quantum_features} → {self.quantum_encoder.n_output_features} encoded")
        print(f"  Circuit: {self.n_qubits} qubits, {self.n_layers} layers")
        print(f"  SMOTE: {self.use_smote}, SVM tuning: {self.tune_svm}")
    
    def fit(self, X, y):
        """
        Train hybrid quantum classifier with all enhancements.
        
        Args:
            X: Classical features (n_samples, n_features) [typically 50 after PCA]
            y: Labels (n_samples,)
        """
        print(f"\nTraining Enhanced Hybrid Quantum Classifier on {len(X)} samples...")
        print(f"Input features: {X.shape[1]}")
        
        # Step 1: Select most important features for quantum encoding
        print(f"\nStep 1: Selecting {self.n_quantum_features} features for quantum encoding...")
        n_features_in = X.shape[1]
        k = min(self.n_quantum_features, n_features_in)
        if k < self.n_quantum_features:
            print(f"  Warning: Reduced quantum features from {self.n_quantum_features} to {k}")
            # Update encoder input size
            self.quantum_encoder = QuantumFeatureEncoder(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                n_input_features=k
            )

        self.feature_selector = SelectKBest(
            mutual_info_classif,
            k=k
        )
        X_for_quantum = self.feature_selector.fit_transform(X, y)
        print(f"  Selected features shape: {X_for_quantum.shape}")
        
        # Step 1.5: Train quantum encoder parameters via gradient descent
        train_quantum = self.config.get('hybrid', {}).get('train_quantum_params', True)
        n_epochs_q = self.config.get('hybrid', {}).get('quantum_train_epochs', 30)
        lr_q = self.config.get('hybrid', {}).get('quantum_train_lr', 0.05)
        if train_quantum:
            print(f"\nStep 1.5: Training quantum encoder parameters...")
            self.quantum_encoder.train_params(
                X_for_quantum, y,
                n_epochs=n_epochs_q, lr=lr_q, batch_size=min(32, len(y))
            )
        else:
            print(f"\nStep 1.5: Skipped (train_quantum_params=False in config)")
        
        # Step 2: Quantum encoding (now using optimised parameters)
        print(f"\nStep 2: Encoding features with enhanced quantum circuit...")
        print(f"  {self.n_qubits} qubits, {self.n_layers} layers, 3 measurement bases")
        X_quantum_encoded = self.quantum_encoder.encode(X_for_quantum)
        print(f"  Quantum-encoded features shape: {X_quantum_encoded.shape}")
        
        # Step 3: Create feature interaction terms
        if self.use_feature_interactions:
            print(f"\nStep 3: Creating quantum-classical interaction features...")
            # Select top classical features by mutual information
            mi_scores = mutual_info_classif(X, y, random_state=42)
            n_interaction = min(self.quantum_encoder.n_output_features, X.shape[1])
            self.top_classical_indices = np.argsort(mi_scores)[-n_interaction:]
            X_top_classical = X[:, self.top_classical_indices]
            
            # Interaction: quantum_feature × classical_feature (element-wise)
            n_interact = min(X_quantum_encoded.shape[1], X_top_classical.shape[1])
            X_interactions = X_quantum_encoded[:, :n_interact] * X_top_classical[:, :n_interact]
            
            X_hybrid = np.concatenate([X, X_quantum_encoded, X_interactions], axis=1)
            print(f"  Interaction features: {X_interactions.shape[1]}")
        else:
            X_hybrid = np.concatenate([X, X_quantum_encoded], axis=1)
        
        print(f"\nStep 4: Creating hybrid feature set...")
        print(f"  Final hybrid features: {X_hybrid.shape}")
        print(f"    - Classical features: {X.shape[1]}")
        print(f"    - Quantum features:   {X_quantum_encoded.shape[1]}")
        if self.use_feature_interactions:
            print(f"    - Interaction terms:  {X_interactions.shape[1]}")
        print(f"    - Total:              {X_hybrid.shape[1]}")
        
        # Step 5: Scale hybrid features
        print(f"\nStep 5: Scaling hybrid features...")
        X_hybrid_scaled = self.hybrid_scaler.fit_transform(X_hybrid)
        
        # Step 6: SMOTE oversampling
        if self.use_smote:
            print(f"\nStep 6: Applying SMOTE oversampling...")
            try:
                from imblearn.over_sampling import SMOTE
                class_counts = np.bincount(y.astype(int))
                print(f"  Before SMOTE: {dict(enumerate(class_counts))}")
                
                smote = SMOTE(random_state=42)
                X_hybrid_scaled, y_train = smote.fit_resample(X_hybrid_scaled, y)
                
                class_counts_after = np.bincount(y_train.astype(int))
                print(f"  After SMOTE:  {dict(enumerate(class_counts_after))}")
            except ImportError:
                print("  WARNING: imbalanced-learn not installed. Skipping SMOTE.")
                print("  Install with: pip install imbalanced-learn")
                y_train = y
        else:
            y_train = y
        
        # Step 7: Train SVM with optional GridSearchCV
        if self.tune_svm:
            print(f"\nStep 7: Tuning SVM hyperparameters with GridSearchCV...")
            param_grid = {
                'C': [1, 10],
                'gamma': ['scale', 0.01],
                'kernel': ['rbf', 'linear']
            }
            
            import warnings
            
            base_svm = SVC(
                probability=True,
                class_weight='balanced',
                max_iter=20000,
                cache_size=1000
            )
            
            n_combos = len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])
            print(f"  Grid: {n_combos} combinations × 3 folds = {n_combos * 3} fits")
            
            grid_search = GridSearchCV(
                base_svm, param_grid, 
                cv=3, scoring='f1', 
                n_jobs=-1, verbose=1
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                grid_search.fit(X_hybrid_scaled, y_train)
            
            self.classifier = grid_search.best_estimator_
            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Best CV F1:  {grid_search.best_score_:.4f}")
        else:
            print(f"\nStep 7: Training SVM on hybrid features...")
            self.classifier = SVC(
                kernel='rbf', C=10.0, gamma='scale',
                probability=True, class_weight='balanced',
                max_iter=-1, cache_size=1000
            )
            self.classifier.fit(X_hybrid_scaled, y_train)
        
        # Step 7.5: Probability calibration (isotonic regression)
        # SVM Platt scaling can be poorly calibrated; isotonic regression on
        # the original (non-SMOTE) data improves predicted probabilities and
        # makes them generalize better to unseen/external images.
        print(f"\nStep 7.5: Calibrating probabilities (isotonic regression)...")
        X_hybrid_orig = self.hybrid_scaler.transform(
            np.concatenate([X, X_quantum_encoded] + 
                          ([X_interactions] if self.use_feature_interactions else []), axis=1)
        )
        try:
            cal_clf = CalibratedClassifierCV(
                self.classifier, method='isotonic', cv='prefit'
            )
            cal_clf.fit(X_hybrid_orig, y)
            self.calibrator = cal_clf
            print(f"  Probability calibration fitted.")
        except Exception as e:
            print(f"  Calibration failed: {e}. Using raw SVM probabilities.")
            self.calibrator = None
        
        # Step 8: Optimize decision threshold
        if self.optimize_threshold:
            print(f"\nStep 8: Optimizing decision threshold...")
            # Use original (non-SMOTE) data for threshold optimization
            X_hybrid_orig = self.hybrid_scaler.transform(
                np.concatenate([X, X_quantum_encoded] + 
                              ([X_interactions] if self.use_feature_interactions else []), axis=1)
            )
            y_proba = self.classifier.predict_proba(X_hybrid_orig)[:, 1]
            
            from sklearn.metrics import f1_score as f1_metric
            best_f1 = 0
            best_thresh = 0.5
            for thresh in np.arange(0.3, 0.7, 0.01):
                y_pred_thresh = (y_proba >= thresh).astype(int)
                f1 = f1_metric(y, y_pred_thresh, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            
            self.decision_threshold = best_thresh
            print(f"  Optimal threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")
        
        print("\nEnhanced Hybrid Quantum Classifier training complete ✓")
        return self
    
    def predict(self, X):
        """
        Predict labels using optimized threshold.
        
        Args:
            X: Classical features (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.decision_threshold).astype(int)
    
    def predict_proba(self, X, quantum_weight=1.0):
        """
        Predict class probabilities.
        
        Args:
            X: Classical features (n_samples, n_features)
            quantum_weight: Float in [0, 1]. Reduces quantum noise influence
                on out-of-distribution inputs (default 1.0 = unchanged).
            
        Returns:
            Class probabilities (n_samples, 2)
        """
        X_hybrid = self._prepare_hybrid_features(X, quantum_weight=quantum_weight)
        # Use calibrated probabilities if available (better generalisation)
        if getattr(self, 'calibrator', None) is not None:
            return self.calibrator.predict_proba(X_hybrid)
        return self.classifier.predict_proba(X_hybrid)
    
    def _prepare_hybrid_features(self, X, quantum_weight=1.0):
        """
        Prepare hybrid features for prediction.
        
        Args:
            X: Classical features
            quantum_weight: Float in [0, 1]. Controls influence of quantum-encoded
                features during inference. 1.0 = full quantum contribution (default,
                matches training). Lower values blend quantum features toward their
                training mean, reducing quantum noise on out-of-distribution inputs.
            
        Returns:
            Scaled hybrid features (classical + quantum + interactions)
        """
        # Ensure proper numpy array (fixes version mismatch between pickle and runtime)
        X = np.asarray(X, dtype=np.float64)
        
        # Select features for quantum encoding
        X_for_quantum = self.feature_selector.transform(X)
        
        # Quantum encoding
        X_quantum_encoded = self.quantum_encoder.encode(X_for_quantum)
        
        # Feature interactions
        if self.use_feature_interactions and self.top_classical_indices is not None:
            X_top_classical = X[:, self.top_classical_indices]
            n_interact = min(X_quantum_encoded.shape[1], X_top_classical.shape[1])
            X_interactions = X_quantum_encoded[:, :n_interact] * X_top_classical[:, :n_interact]
            X_hybrid = np.concatenate([X, X_quantum_encoded, X_interactions], axis=1)
        else:
            X_hybrid = np.concatenate([X, X_quantum_encoded], axis=1)
        
        # Reduce quantum noise for out-of-distribution inputs by blending
        # quantum + interaction features toward their training mean.
        # After StandardScaler this effectively scales their contribution by
        # quantum_weight, leaving classical PCA features untouched.
        if quantum_weight < 1.0 and hasattr(self.hybrid_scaler, 'mean_'):
            n_classical = X.shape[1]
            X_hybrid[:, n_classical:] = (
                X_hybrid[:, n_classical:] * quantum_weight
                + self.hybrid_scaler.mean_[n_classical:] * (1.0 - quantum_weight)
            )
        
        # Scale
        X_hybrid_scaled = self.hybrid_scaler.transform(X_hybrid)
        
        return X_hybrid_scaled
    
    def save(self, path: str):
        """Save hybrid model to file."""
        model_data = {
            'config': self.config,
            'feature_selector': self.feature_selector,
            'classifier': self.classifier,
            'hybrid_scaler': self.hybrid_scaler,
            'decision_threshold': self.decision_threshold,
            'top_classical_indices': self.top_classical_indices,
            'use_feature_interactions': self.use_feature_interactions,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'n_quantum_features': self.n_quantum_features,
            'params': self.quantum_encoder.params,
            'calibrator': getattr(self, 'calibrator', None),
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Hybrid model saved to {path}")
    
    def load(self, path: str):
        """Load hybrid model from file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.feature_selector = model_data['feature_selector']
        self.classifier = model_data['classifier']
        self.n_qubits = model_data['n_qubits']
        self.n_layers = model_data['n_layers']
        self.n_quantum_features = model_data['n_quantum_features']
        
        # Load new fields with backward compatibility
        self.hybrid_scaler = model_data.get('hybrid_scaler', StandardScaler())
        self.decision_threshold = model_data.get('decision_threshold', 0.5)
        self.top_classical_indices = model_data.get('top_classical_indices', None)
        self.use_feature_interactions = model_data.get('use_feature_interactions', False)
        self.calibrator = model_data.get('calibrator', None)
        
        # Reinitialize quantum encoder with saved params
        self.quantum_encoder = QuantumFeatureEncoder(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_input_features=self.n_quantum_features
        )
        if 'params' in model_data:
            self.quantum_encoder.params = model_data['params']
        
        print(f"Hybrid model loaded from {path}")
        if self.calibrator is not None:
            print(f"  Probability calibrator loaded (isotonic regression)")
