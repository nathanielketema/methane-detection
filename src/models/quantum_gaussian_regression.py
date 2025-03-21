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
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        algorithm_globals.random_seed = 42
    
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
        # Generate a model filename based on data characteristics and hyperparameters
        model_hash = f"qgpr_nqubits{self.n_qubits or X.shape[1]}_alpha{self.alpha}_normalize{self.normalize_y}_samples{X.shape[0]}_features{X.shape[1]}"
        model_path = os.path.join(self.model_dir, f"{model_hash}.joblib")
        
        # Try to load a pre-trained model if it exists and we're not forcing a retrain
        if not force_retrain and os.path.exists(model_path):
            print(f"Loading pre-trained QGPR model from {model_path}")
            self._load_model(model_path)
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
        
        # Save the model for future use
        self._save_model(model_path)
        
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
    
    def _save_model(self, path):
        """
        Save the trained model to a file.
        
        Parameters:
          path (str): Path to save the model file
        """
        if self.model is None:
            raise ValueError("Cannot save: model has not been fitted yet.")
        
        # Create a dictionary with all necessary components to recreate the model
        model_data = {
            'model': self.model,
            'kernel': self.kernel,
            'alpha': self.alpha,
            'n_qubits': self.n_qubits,
            'shots': self.shots,
            'normalize_y': self.normalize_y,
            'encoding_circuit': self.encoding_circuit
        }
        
        # Save model to file
        try:
            joblib.dump(model_data, path)
            print(f"Model successfully saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _load_model(self, path):
        """
        Load a trained model from a file.
        
        Parameters:
          path (str): Path to the model file
        
        Returns:
          bool: True if model was successfully loaded
        """
        try:
            # Load model data from file
            model_data = joblib.load(path)
            
            # Restore model components
            self.model = model_data['model']
            self.kernel = model_data['kernel']
            self.alpha = model_data['alpha']
            self.n_qubits = model_data['n_qubits']
            self.shots = model_data['shots']
            self.normalize_y = model_data['normalize_y']
            self.encoding_circuit = model_data['encoding_circuit']
            
            print(f"Model successfully loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

if __name__ == "__main__":
    print("QuantumGaussianRegression module - this module provides quantum-enhanced regression for methane concentration prediction")
    print("Import and use this module in other scripts rather than running it directly.")
