import numpy as np
from squlearn.kernel.matrix import FidelityKernel
from squlearn.kernel.ml import QGPR
from squlearn import Executor
from squlearn.encoding_circuit import HubregtsenEncodingCircuit
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer import Aer
import joblib
import os
import time
import pickle
import hashlib

class QuantumGaussianRegression:
    def __init__(self, alpha=1e-10, n_qubits=None, shots=1024, normalize_y=False, model_dir="models"):
        """
        Initializes the QuantumGaussianRegression model using squlearn's QGPR.
        
        Parameters:
          alpha (float): Regularization parameter for the Gaussian process.
          n_qubits (int): Number of qubits to use in the quantum circuit. If None, will be determined from data dimensions.
          shots (int): Number of shots for quantum circuit execution.
          normalize_y (bool): Whether to normalize target values by removing mean and scaling to unit variance.
          model_dir (str): Directory to save/load model files.
        """
        self.alpha = alpha
        self.n_qubits = n_qubits
        self.shots = shots
        self.normalize_y = normalize_y
        self.model = None
        self.kernel = None
        self.model_dir = model_dir
        self.encoding_circuit = None
        
        # Store training data explicitly
        self.X_train = None
        self.y_train = None
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "training_data"), exist_ok=True)
        
        # Set random seed for reproducibility
        algorithm_globals.random_seed = 42
    
    def _generate_model_hash(self, X):
        """
        Generate a hash for the model based on its parameters and data dimensions.
        
        Parameters:
          X (np.ndarray): Feature matrix
        
        Returns:
          str: A unique hash string for the model
        """
        n_qubits = self.n_qubits or X.shape[1]
        hash_str = f"qgpr_nqubits{n_qubits}_alpha{self.alpha}_normalize{self.normalize_y}_samples{X.shape[0]}_features{X.shape[1]}"
        return hash_str
    
    def _get_model_paths(self, model_hash):
        """
        Get paths for model files.
        
        Parameters:
          model_hash (str): The model hash string
        
        Returns:
          tuple: Paths for model parameters, training features, and training targets
        """
        model_path = os.path.join(self.model_dir, f"{model_hash}.joblib")
        features_path = os.path.join(self.model_dir, "training_data", f"{model_hash}_features.npy")
        targets_path = os.path.join(self.model_dir, "training_data", f"{model_hash}_targets.npy")
        return model_path, features_path, targets_path
    
    def fit(self, X, y, force_retrain=False):
        """
        Fit the quantum Gaussian process regression model. If a saved model exists,
        it will be loaded instead of retraining unless force_retrain is True.
        
        Parameters:
          X (np.ndarray): Feature matrix with shape (n_samples, n_features)
          y (np.ndarray): Target values with shape (n_samples,)
          force_retrain (bool): If True, retrain the model even if a saved version exists
        
        Returns:
          self: The fitted model
        """
        # Validate input data
        if X is None or y is None:
            raise ValueError("Training data X and y must not be None")
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data X and y must not be empty")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same number of samples. Got X: {len(X)}, y: {len(y)}")
        
        # Generate model hash and paths
        model_hash = self._generate_model_hash(X)
        model_path, features_path, targets_path = self._get_model_paths(model_hash)
        
        # Store the training data
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Try to load a pre-trained model if it exists and we're not forcing a retrain
        if not force_retrain and os.path.exists(model_path):
            print(f"Loading pre-trained QGPR model from {model_path}")
            if self._load_model(model_path, features_path, targets_path):
                return self
        
        print("Training new QGPR model...")
        start_time = time.time()
        
        # Determine number of qubits if not specified
        if self.n_qubits is None:
            self.n_qubits = min(X.shape[1], 8)  # Use up to 8 qubits based on features
        
        # Create an encoding circuit and executor for squlearn
        num_features = X.shape[1]
        
        # Create squlearn's encoding circuit
        self.encoding_circuit = HubregtsenEncodingCircuit(
            num_qubits=self.n_qubits,
            num_features=num_features,
            num_layers=2
        )
        
        # Create squlearn's executor
        executor = Executor(
            backend=Aer.get_backend('qasm_simulator'),
            shots=self.shots
        )
        
        # Create the FidelityKernel with the encoding circuit and executor
        self.kernel = FidelityKernel(
            encoding_circuit=self.encoding_circuit,
            executor=executor
        )
        
        # Assign random parameters to the kernel
        self.kernel.assign_parameters(np.random.rand(self.encoding_circuit.num_parameters))
        
        # Create and fit the QGPR model
        self.model = QGPR(
            quantum_kernel=self.kernel, 
            sigma=self.alpha,
            normalize_y=self.normalize_y,
            full_regularization=True
        )
        
        self.model.fit(X, y)
        
        training_time = time.time() - start_time
        print(f"Quantum Gaussian Process Regression model training complete in {training_time:.2f} seconds.")
        
        # Save the model and training data
        self._save_model(model_path, features_path, targets_path)
        
        return self
    
    def predict(self, X):
        """
        Predict using the quantum Gaussian process regression model.
        
        Parameters:
          X (np.ndarray): Feature matrix with shape (n_samples, n_features)
        
        Returns:
          y_pred (np.ndarray): Predicted values.
          sigma (np.ndarray): Standard deviation of the predictions.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Get predictions and standard deviations
        y_pred, sigma = self.model.predict(X, return_std=True)
        return y_pred, sigma
    
    def get_kernel(self):
        """
        Returns the kernel used by the Gaussian Process.
        """
        if self.kernel is None:
            raise ValueError("Kernel has not been initialized yet. Call fit() first.")
        return self.kernel
    
    def _save_model(self, model_path, features_path, targets_path):
        """
        Save the trained model and training data to separate files.
        
        Parameters:
          model_path (str): Path to save the model parameters
          features_path (str): Path to save the training features
          targets_path (str): Path to save the training targets
        """
        if self.model is None:
            raise ValueError("Cannot save: model has not been fitted yet.")
        
        # Save training data
        np.save(features_path, self.X_train)
        np.save(targets_path, self.y_train)
        
        # Save model parameters
        model_params = {
            'alpha': self.alpha,
            'n_qubits': self.n_qubits,
            'shots': self.shots,
            'normalize_y': self.normalize_y,
            'kernel_parameters': self.kernel.get_parameters(),
            'model_params': {
                'alpha': self.model.alpha,
                'normalize_y': self.model.normalize_y
            }
        }
        
        joblib.dump(model_params, model_path)
        print(f"Model saved to {model_path}")
    
    def _load_model(self, model_path, features_path, targets_path):
        """
        Load a trained model from disk.
        
        Parameters:
          model_path (str): Path to the saved model parameters
          features_path (str): Path to the saved training features
          targets_path (str): Path to the saved training targets
        
        Returns:
          bool: True if model was loaded successfully, False otherwise
        """
        try:
            # Load model parameters
            model_params = joblib.load(model_path)
            
            # Restore model parameters
            self.alpha = model_params['alpha']
            self.n_qubits = model_params['n_qubits']
            self.shots = model_params['shots']
            self.normalize_y = model_params['normalize_y']
            
            # Load training data
            self.X_train = np.load(features_path)
            self.y_train = np.load(targets_path)
            
            # Recreate the model with saved parameters
            num_features = self.X_train.shape[1]
            self.encoding_circuit = HubregtsenEncodingCircuit(
                num_qubits=self.n_qubits,
                num_features=num_features,
                num_layers=2
            )
            
            executor = Executor(
                backend=Aer.get_backend('qasm_simulator'),
                shots=self.shots
            )
            
            self.kernel = FidelityKernel(
                encoding_circuit=self.encoding_circuit,
                executor=executor
            )
            
            self.kernel.assign_parameters(model_params['kernel_parameters'])
            
            self.model = QGPR(
                quantum_kernel=self.kernel,
                sigma=self.alpha,
                normalize_y=self.normalize_y,
                full_regularization=True
            )
            
            # Restore model parameters
            self.model.alpha = model_params['model_params']['alpha']
            self.model.normalize_y = model_params['model_params']['normalize_y']
            
            print("Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

if __name__ == "__main__":
    print("QuantumGaussianRegression module - this module provides quantum-enhanced regression for methane concentration prediction")
    print("Import and use this module in other scripts rather than running it directly.")
