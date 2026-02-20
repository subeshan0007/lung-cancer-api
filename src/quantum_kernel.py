"""
Quantum Kernel Method (QKM) for lung nodule classification.
Primary approach using quantum feature maps and classical SVM.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Dict, List
from sklearn.svm import SVC
from tqdm import tqdm
import pickle


class QuantumKernel:
    """Quantum kernel using PennyLane for computing feature map overlaps."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_qubits = config['quantum']['n_qubits']
        self.n_layers = config['quantum']['n_layers']
        self.entanglement = config['quantum']['entanglement']
        self.use_data_reuploading = config['quantum'].get('use_data_reuploading', False)
        
        # Initialize quantum device
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        
        # Choose feature map based on configuration
        if self.use_data_reuploading:
            feature_map_func = self.feature_map_data_reuploading
            print(f"Using data re-uploading feature map")
        else:
            feature_map_func = self._quantum_feature_map
        
        # Create quantum circuit
        self.kernel_circuit = qml.QNode(feature_map_func, self.dev)
        
        print(f"Initialized Quantum Kernel: {self.n_qubits} qubits, {self.n_layers} layers")
    
    def _quantum_feature_map(self, x1: np.ndarray, x2: np.ndarray):
        """
        Quantum feature map with data re-uploading.
        
        Args:
            x1: First feature vector (n_features,)
            x2: Second feature vector (n_features,)
        
        Returns:
            Fidelity between quantum states
        """
        n_features = len(x1)
        features_per_qubit = int(np.ceil(n_features / self.n_qubits))
        
        # Encode first vector with data re-uploading
        for layer in range(self.n_layers):
            # Encode features
            for i, qubit in enumerate(range(self.n_qubits)):
                feat_idx = (i * features_per_qubit + layer) % n_features
                qml.RY(x1[feat_idx], wires=qubit)
                qml.RZ(x1[feat_idx], wires=qubit)
            
            # Entanglement layer
            if layer < self.n_layers - 1:
                self._entanglement_layer()
        
        # Apply inverse of second vector encoding (for fidelity calculation)
        for layer in range(self.n_layers - 1, -1, -1):
            if layer < self.n_layers - 1:
                self._entanglement_layer()
            
            # Inverse encoding
            for i in range(self.n_qubits - 1, -1, -1):
                qubit = i
                feat_idx = (i * features_per_qubit + layer) % n_features
                qml.RZ(-x2[feat_idx], wires=qubit)
                qml.RY(-x2[feat_idx], wires=qubit)
        
        # Return all probabilities - we'll extract [0] outside the QNode
        return qml.probs(wires=range(self.n_qubits))
    
    def _entanglement_layer(self):
        """Create entanglement between qubits."""
        if self.entanglement == 'full':
            # Full entanglement
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.CNOT(wires=[i, j])
        
        elif self.entanglement == 'linear':
            # Linear chain
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        elif self.entanglement == 'circular':
            # Ring topology
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[self.n_qubits-1, 0])
    
    def feature_map_data_reuploading(self, x1, x2):
        """
        Advanced data re-uploading feature map.
        Re-uploads data in each layer for better expressivity.
        Uses normalized inputs to quantum rotation ranges.
        """
        n_features = len(x1)
        features_per_qubit = max(n_features // self.n_qubits, 1)
        
        # Encode first vector with data re-uploading
        for layer in range(self.n_layers):
            # Data encoding - reupload features in each layer
            for i in range(self.n_qubits):
                start_idx = i * features_per_qubit
                end_idx = min(start_idx + 2, n_features)
                
                if start_idx < n_features:
                    # Normalize to [-π, π] using tanh
                    angle1 = np.pi * np.tanh(x1[start_idx])
                    qml.RY(angle1, wires=i)
                    
                    if start_idx + 1 < n_features:
                        angle2 = np.pi * np.tanh(x1[start_idx + 1])
                        qml.RZ(angle2, wires=i)
            
            # Entangling layer after each data encoding
            if layer < self.n_layers - 1:
                self._entanglement_layer()
        
        # Inverse encoding with second vector
        for layer in range(self.n_layers - 1, -1, -1):
            if layer < self.n_layers - 1:
                self._entanglement_layer()
            
            # Inverse data encoding
            for i in range(self.n_qubits - 1, -1, -1):
                start_idx = i * features_per_qubit
                
                if start_idx < n_features:
                    if start_idx + 1 < n_features:
                        angle2 = np.pi * np.tanh(x2[start_idx + 1])
                        qml.RZ(-angle2, wires=i)
                    
                    angle1 = np.pi * np.tanh(x2[start_idx])
                    qml.RY(-angle1, wires=i)
        
        return qml.probs(wires=range(self.n_qubits))
    
    def compute_kernel_element(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute single kernel element K(x1, x2).
        
        Args:
            x1: First feature vector
            x2: Second feature vector
            
        Returns:
            Kernel value (fidelity)
        """
        try:
            probs = self.kernel_circuit(x1, x2)
            # Extract probability of all qubits in |0⟩ state (first element)
            fidelity = probs[0]
            return float(fidelity)
        except Exception as e:
            print(f"Error computing kernel element: {e}")
            return 0.0
    
    def normalize_kernel(self, K):
        """
        Normalize kernel matrix to improve conditioning.
        Uses cosine normalization (diagonal normalization).
        
        Args:
            K: Kernel matrix (n_samples, n_samples)
            
        Returns:
            Normalized kernel matrix
        """
        # Diagonal normalization (cosine normalization)
        K_diag = np.diag(K)
        K_diag_sqrt = np.sqrt(np.outer(K_diag, K_diag))
        
        # Avoid division by zero
        K_normalized = K / (K_diag_sqrt + 1e-10)
        
        # Ensure symmetry (averaging with transpose)
        K_normalized = (K_normalized + K_normalized.T) / 2
        
        return K_normalized
    
    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray = None, normalize=True) -> np.ndarray:
        """
        Compute kernel matrix between two sets of samples.
        
        Args:
            X1: (n_samples_1, n_features)
            X2: (n_samples_2, n_features) or None (uses X1)
            normalize: Whether to normalize the kernel matrix
            
        Returns:
            Kernel matrix (n_samples_1, n_samples_2)
        """
        if X2 is None:
            X2 = X1
            symmetric = True
        else:
            symmetric = False
        
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        
        print(f"Computing quantum kernel matrix ({n1} x {n2})...")
        
        total_elements = n1 * n2 if not symmetric else n1 * (n1 + 1) // 2
        pbar = tqdm(total=total_elements, desc="Kernel computation")
        
        for i in range(n1):
            start_j = i if symmetric else 0
            for j in range(start_j, n2):
                K[i, j] = self.compute_kernel_element(X1[i], X2[j])
                
                if symmetric and i != j:
                    K[j, i] = K[i, j]  # Use symmetry
                
                pbar.update(1)
        
        pbar.close()
        
        # Normalize kernel if requested and matrix is square (training)
        if normalize and symmetric:
            K = self.normalize_kernel(K)
            print("✓ Kernel matrix normalized")
        
        return K


class QuantumKernelClassifier:
    """
    Quantum Kernel Method classifier combining quantum kernel with SVM.
    PRIMARY APPROACH for robust generalization.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.quantum_kernel = QuantumKernel(config)
        self.svm = None
        self.X_train = None  # Store training data for kernel computation
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, C: float = None, 
            tune_C: bool = True, enable_diagnostics: bool = True):
        """
        Train quantum kernel classifier with optional C tuning and diagnostics.
        
        Args:
            X_train: Training features (n_samples, n_quantum_features)
            y_train: Training labels (n_samples,)
            C: SVM regularization parameter (None for auto-tuning)
            tune_C: Whether to tune C parameter via cross-validation
            enable_diagnostics: Whether to run kernel diagnostics
        """
        print(f"Training Quantum Kernel Classifier on {len(X_train)} samples...")
        import sys
        
        # Compute quantum kernel matrix for training data (with normalization)
        K_train = self.quantum_kernel.compute_kernel_matrix(X_train, X_train, normalize=True)
        
        print(f"Kernel matrix computed. Shape: {K_train.shape}")
        
        # Run diagnostics if enabled
        if enable_diagnostics:
            from .quantum_diagnostics import QuantumDiagnostics
            diag = QuantumDiagnostics()
            kernel_stats = diag.analyze_kernel(K_train, y_train, 
                                              save_path='results/kernel_diagnostics.png')
            self.kernel_stats = kernel_stats
        
        print(f"Starting SVM training with precomputed kernel...")
        sys.stdout.flush()
        
        # Compute class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', 
                                             classes=np.unique(y_train), 
                                             y=y_train)
        sample_weights = np.array([class_weights[y] for y in y_train])
        
        print(f"Class weights: {dict(zip(np.unique(y_train), class_weights))}")
        sys.stdout.flush()
        
        # C parameter tuning
        if C is None or tune_C:
            from sklearn.model_selection import cross_val_score
            
            C_candidates = [0.01, 0.1, 0.5, 1.0, 2.0]
            best_score = -1
            best_C = 1.0
            
            print("\nTuning SVM C parameter...")
            for C_test in C_candidates:
                svm_test = SVC(kernel='precomputed', C=C_test, probability=True,
                              class_weight='balanced', max_iter=-1)
                
                # Quick 2-fold CV for speed
                try:
                    scores = cross_val_score(svm_test, K_train, y_train, cv=2, 
                                           scoring='roc_auc', n_jobs=1,
                                           fit_params={'sample_weight': sample_weights})
                    mean_score = scores.mean()
                    print(f"  C={C_test}: AUC-ROC={mean_score:.4f}")
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_C = C_test
                except Exception as e:
                    print(f"  C={C_test}: Failed ({str(e)})")
            
            C = best_C
            print(f"\n✓ Best C parameter: {best_C} (AUC-ROC: {best_score:.4f})")
            sys.stdout.flush()
        
        # Train SVM with best C, increased max_iter for better convergence
        self.svm = SVC(kernel='precomputed', C=C, probability=True, 
                      class_weight='balanced', max_iter=-1, verbose=True)
        self.svm.fit(K_train, y_train, sample_weight=sample_weights)
        
        print("\nSVM training completed!")
        sys.stdout.flush()
        
        # Store training data for future predictions
        self.X_train = X_train
        self.best_C = C
        
        print("Quantum Kernel Classifier training complete")
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict labels for test samples.
        
        Args:
            X_test: Test features (n_samples, n_quantum_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        # Compute kernel matrix between test and train
        K_test = self.quantum_kernel.compute_kernel_matrix(X_test, self.X_train)
        
        # Predict using SVM
        predictions = self.svm.predict(K_test)
        
        return predictions
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for test samples.
        
        Args:
            X_test: Test features (n_samples, n_quantum_features)
            
        Returns:
            Class probabilities (n_samples, 2)
        """
        K_test = self.quantum_kernel.compute_kernel_matrix(X_test, self.X_train)
        probabilities = self.svm.predict_proba(K_test)
        
        return probabilities
    
    def save(self, path: str):
        """Save model to file."""
        model_data = {
            'svm': self.svm,
            'X_train': self.X_train,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svm = model_data['svm']
        self.X_train = model_data['X_train']
        print(f"Model loaded from {path}")


def draw_quantum_circuit(config: Dict, sample_features: np.ndarray):
    """
    Draw and save quantum circuit diagram.
    
    Args:
        config: Configuration dictionary
        sample_features: Sample feature vector for visualization
    """
    n_qubits = config['quantum']['n_qubits']
    n_layers = config['quantum']['n_layers']
    
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(x):
        n_features = len(x)
        features_per_qubit = int(np.ceil(n_features / n_qubits))
        
        for layer in range(n_layers):
            for i, qubit in enumerate(range(n_qubits)):
                feat_idx = (i * features_per_qubit + layer) % n_features
                qml.RY(x[feat_idx], wires=qubit)
                qml.RZ(x[feat_idx], wires=qubit)
            
            if layer < n_layers - 1:
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
        
        return qml.probs(wires=range(n_qubits))
    
    # Run circuit to generate drawer
    circuit(sample_features)
    
    # Draw circuit
    print("\n" + "="*60)
    print("Quantum Circuit Diagram:")
    print("="*60)
    print(qml.draw(circuit)(sample_features))
    print("="*60 + "\n")
