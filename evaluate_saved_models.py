#!/usr/bin/env python
"""
Script to evaluate the saved models using the model_evaluation module.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from src.model_evaluation import ClassificationEvaluator, RegressionEvaluator, plot_geospatial_errors

# Path to models
MODEL_DIR = 'models'
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'methane_leak_classifier.joblib')
REGRESSOR_PATH = os.path.join(MODEL_DIR, 'qgpr_nqubits3_alpha1e-10_normalizeFalse_samples10000_features3.joblib')

# Create output directory for saved figures
os.makedirs('evaluation_results', exist_ok=True)

def load_or_generate_test_data(n_samples=200, n_features=3):
    """
    Load or generate sample test data for model evaluation.
    In a real scenario, you would load your actual test data.
    
    Returns:
        X: Features for testing
        y_clf: Binary labels for classification (1=leak, 0=no leak)
        y_reg: Continuous values for regression (tracer concentration)
        lat: Latitude coordinates for spatial visualization
        lon: Longitude coordinates for spatial visualization
    """
    print("Generating synthetic test data...")
    
    # Generate synthetic data
    X = np.random.rand(n_samples, n_features)
    
    # Regression target: tracer concentration (continuous values)
    y_reg = np.exp(-((X[:, 0] - 0.5)**2 + (X[:, 1] - 0.5)**2) / 0.1) + 0.05 * np.random.randn(n_samples)
    
    # Classification target: leak detection (binary, 1=leak, 0=no leak)
    y_clf = (y_reg > 0.3).astype(int)
    
    # Spatial coordinates for maps
    lat = 37.7749 + 0.01 * X[:, 0]  # San Francisco approximate coordinates
    lon = -122.4194 + 0.01 * X[:, 1]
    
    return X, y_clf, y_reg, lat, lon

def evaluate_classifier():
    """
    Evaluate the saved methane leak classifier.
    """
    print("\n=== Evaluating Methane Leak Classifier ===")
    
    try:
        # Load the classifier model
        print(f"Loading classifier from {CLASSIFIER_PATH}...")
        classifier = joblib.load(CLASSIFIER_PATH)
        print("Classifier loaded successfully.")
        
        # Generate test data
        X, y_clf, _, lat, lon = load_or_generate_test_data()
        
        # Create evaluator
        print("Evaluating classification performance...")
        evaluator = ClassificationEvaluator(model=classifier)
        
        # Evaluate model
        metrics = evaluator.evaluate(X, y_clf)
        
        # Print report
        evaluator.print_report()
        
        # Plot confusion matrix
        print("Generating confusion matrix plot...")
        fig_cm, _ = evaluator.plot_confusion_matrix(normalize=True)
        fig_cm.savefig('evaluation_results/classifier_confusion_matrix.png', bbox_inches='tight', dpi=300)
        
        # Plot ROC curve
        print("Generating ROC curve plot...")
        fig_roc, _ = evaluator.plot_roc_curve(X, y_clf)
        fig_roc.savefig('evaluation_results/classifier_roc_curve.png', bbox_inches='tight', dpi=300)
        
        print("Classification evaluation complete. Visualizations saved to 'evaluation_results' directory.")
        
    except Exception as e:
        print(f"Error evaluating classifier: {e}")

def evaluate_regressor():
    """
    Evaluate the saved QGPR regressor model.
    """
    print("\n=== Evaluating QGPR Regression Model ===")
    
    try:
        # Load the regressor model
        print(f"Loading regressor from {REGRESSOR_PATH}...")
        regressor = joblib.load(REGRESSOR_PATH)
        print("Regressor loaded successfully.")
        
        # Generate test data
        X, _, y_reg, lat, lon = load_or_generate_test_data()
        
        # Create evaluator
        print("Evaluating regression performance...")
        evaluator = RegressionEvaluator(model=regressor)
        
        # Make predictions and extract uncertainty if available
        print("Making predictions...")
        try:
            predictions = regressor.predict(X)
            
            # Check if predictions include uncertainty estimates
            if isinstance(predictions, tuple) and len(predictions) >= 2:
                y_pred = predictions[0]
                uncertainties = predictions[1]
                print("Uncertainty estimates are available in predictions.")
            else:
                y_pred = predictions
                uncertainties = None
                print("No uncertainty estimates are available in predictions.")
                
        except Exception as prediction_error:
            print(f"Error making predictions: {prediction_error}")
            print("Evaluating with direct predictions...")
            evaluator.evaluate(X, y_reg)
            y_pred = evaluator.y_pred
            uncertainties = None
        
        # Evaluate model
        if 'y_pred' in locals():
            metrics = evaluator.evaluate(X, y_reg, y_pred)
        
        # Print report
        evaluator.print_report()
        
        # Plot predictions
        print("Generating prediction scatter plot...")
        fig_pred, _ = evaluator.plot_predictions()
        fig_pred.savefig('evaluation_results/regressor_predictions.png', bbox_inches='tight', dpi=300)
        
        # Plot residuals
        print("Generating residual plots...")
        fig_resid = evaluator.plot_residuals()
        fig_resid.savefig('evaluation_results/regressor_residuals.png', bbox_inches='tight', dpi=300)
        
        # Select a subset for geospatial visualization
        test_indices = np.random.choice(range(len(X)), size=min(100, len(X)), replace=False)
        X_map = X[test_indices]
        y_true_map = y_reg[test_indices]
        y_pred_map = y_pred[test_indices]
        lat_map = lat[test_indices]
        lon_map = lon[test_indices]
        
        # Get uncertainties for the map if available
        if uncertainties is not None:
            uncertainties_map = uncertainties[test_indices]
        else:
            uncertainties_map = None
        
        # Plot geospatial error map
        print("Generating geospatial error maps...")
        fig_geo = plot_geospatial_errors(lat_map, lon_map, y_true_map, y_pred_map, uncertainties_map)
        fig_geo.savefig('evaluation_results/regressor_geospatial_errors.png', bbox_inches='tight', dpi=300)
        
        print("Regression evaluation complete. Visualizations saved to 'evaluation_results' directory.")
        
    except Exception as e:
        print(f"Error evaluating regressor: {e}")

if __name__ == "__main__":
    print("Starting model evaluation...")
    
    # Evaluate classifier
    evaluate_classifier()
    
    # Evaluate regressor
    evaluate_regressor()
    
    print("\nAll evaluations completed. Results saved to 'evaluation_results' directory.") 