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
        
        # Assign random parameters to the kernel (as shown in the documentation example)
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
        
        # Save the model and training data for future use
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
        try:
            y_pred, sigma = self.model.predict(X, return_std=True)
            return y_pred, sigma
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Fallback to a simple mean prediction if the model fails
            if self.y_train is not None:
                mean_value = np.mean(self.y_train)
                std_value = np.std(self.y_train) if len(self.y_train) > 1 else 1.0
                print(f"Falling back to mean prediction: {mean_value} with std: {std_value}")
                return np.full(len(X), mean_value), np.full(len(X), std_value)
            else:
                raise ValueError("Cannot make fallback prediction as training data is missing")
    
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
        
        # First save the training data separately using numpy's efficient file format
        try:
            print(f"Saving training features to {features_path}")
            np.save(features_path, self.X_train)
            
            print(f"Saving training targets to {targets_path}")
            np.save(targets_path, self.y_train)
        except Exception as e:
            print(f"Error saving training data: {e}")
            # Don't return here, still try to save the model parameters
        
        # Now save the model parameters (without the actual training data)
        try:
            # Get the kernel parameters, with a fallback for compatibility
            kernel_params = None
            if hasattr(self.kernel, 'get_parameters'):
                kernel_params = self.kernel.get_parameters()
            
            # Create a dictionary with only the essential model parameters
            model_params = {
                'alpha': self.alpha,
                'n_qubits': self.n_qubits,
                'shots': self.shots,
                'normalize_y': self.normalize_y,
                'kernel_parameters': kernel_params,
                'model_params': {
                    'alpha': getattr(self.model, 'alpha', None),
                    'normalize_y': getattr(self.model, 'normalize_y', None)
                },
                # Add a data reference instead of the actual data
                'data_reference': {
                    'features_path': features_path,
                    'targets_path': targets_path,
                    'X_shape': self.X_train.shape if self.X_train is not None else None,
                    'y_shape': self.y_train.shape if self.y_train is not None else None
                }
            }
            
            # Try to save using joblib first
            try:
                joblib.dump(model_params, model_path)
                print(f"Model parameters successfully saved to {model_path}")
            except Exception as joblib_error:
                print(f"Error saving with joblib: {joblib_error}")
                # Fall back to pickle if joblib fails
                pickle_path = model_path.replace('.joblib', '.pkl')
                with open(pickle_path, 'wb') as f:
                    pickle.dump(model_params, f)
                print(f"Model parameters saved using pickle to {pickle_path}")
        except Exception as e:
            print(f"Error saving model parameters: {e}")
            print("Continuing without saving the model parameters...")
    
    def _load_model(self, model_path, features_path, targets_path):
        """
        Load a trained model and its training data from files.
        
        Parameters:
          model_path (str): Path to the model parameters file
          features_path (str): Path to the training features file
          targets_path (str): Path to the training targets file
        
        Returns:
          bool: True if model was successfully loaded
        """
        # First try to load the model parameters
        try:
            # Try joblib first
            try:
                model_params = joblib.load(model_path)
            except Exception as joblib_error:
                print(f"Error loading with joblib: {joblib_error}")
                # Fall back to pickle if joblib fails
                pickle_path = model_path.replace('.joblib', '.pkl')
                if os.path.exists(pickle_path):
                    with open(pickle_path, 'rb') as f:
                        model_params = pickle.load(f)
                else:
                    raise FileNotFoundError(f"Neither {model_path} nor {pickle_path} found")
            
            # Restore model parameters
            self.alpha = model_params['alpha']
            self.n_qubits = model_params['n_qubits']
            self.shots = model_params['shots']
            self.normalize_y = model_params['normalize_y']
            
            # Now load the training data from the separate files
            try:
                # Load training features
                if os.path.exists(features_path):
                    print(f"Loading training features from {features_path}")
                    self.X_train = np.load(features_path)
                else:
                    print(f"Warning: Training features file {features_path} not found")
                    self.X_train = None
                
                # Load training targets
                if os.path.exists(targets_path):
                    print(f"Loading training targets from {targets_path}")
                    self.y_train = np.load(targets_path)
                else:
                    print(f"Warning: Training targets file {targets_path} not found")
                    self.y_train = None
            except Exception as data_error:
                print(f"Error loading training data: {data_error}")
                self.X_train = None
                self.y_train = None
            
            # Check if we have valid training data
            if self.X_train is None or self.y_train is None:
                print(f"Error: Could not load training data. X_train: {self.X_train is not None}, y_train: {self.y_train is not None}")
                self.model = None
                return False
            
            # Validate loaded data shapes
            expected_X_shape = model_params.get('data_reference', {}).get('X_shape')
            expected_y_shape = model_params.get('data_reference', {}).get('y_shape')
            
            if expected_X_shape and self.X_train.shape != expected_X_shape:
                print(f"Warning: Loaded X_train shape {self.X_train.shape} does not match expected shape {expected_X_shape}")
            
            if expected_y_shape and self.y_train.shape != expected_y_shape:
                print(f"Warning: Loaded y_train shape {self.y_train.shape} does not match expected shape {expected_y_shape}")
            
            # Recreate the encoding circuit
            num_features = self.X_train.shape[1]
            self.encoding_circuit = HubregtsenEncodingCircuit(
                num_qubits=self.n_qubits,
                num_features=num_features,
                num_layers=2
            )
            
            # Recreate executor
            executor = Executor(
                backend=Aer.get_backend('qasm_simulator'),
                shots=self.shots
            )
            
            # Recreate the kernel
            self.kernel = FidelityKernel(
                encoding_circuit=self.encoding_circuit,
                executor=executor
            )
            
            # Restore kernel parameters if available
            if model_params['kernel_parameters'] is not None:
                self.kernel.assign_parameters(model_params['kernel_parameters'])
            
            # Recreate the model
            self.model = QGPR(
                quantum_kernel=self.kernel, 
                sigma=self.alpha,
                normalize_y=self.normalize_y,
                full_regularization=True
            )
            
            # Fit with loaded data
            print("Fitting model with loaded training data...")
            self.model.fit(self.X_train, self.y_train)
            print(f"Model successfully loaded and recreated")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Reset the model to None to ensure accurate state reporting
            self.model = None
            return False

if __name__ == "__main__":
    print("QuantumGaussianRegression module - this module provides quantum-enhanced regression for methane concentration prediction")
    print("Import and use this module in other scripts rather than running it directly.")
