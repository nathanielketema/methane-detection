"""
Example script demonstrating the use of the model_evaluation module
for evaluating methane leak detection models.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom modules
from src.model_evaluation import ClassificationEvaluator, RegressionEvaluator, plot_geospatial_errors
from src.models.methane_leak_classifier import MethaneLeakClassifier

# Create output directory for saved figures
os.makedirs('outputs', exist_ok=True)

def load_sample_data():
    """
    Load or generate sample data for the example.
    In a real scenario, you would load your actual data.
    """
    # Generate synthetic data for demonstration
    n_samples = 200
    
    # Features: lat, lon, wind_speed, pressure, etc.
    X = np.random.rand(n_samples, 5)
    
    # Regression target: tracer concentration (continuous values)
    y_reg = np.exp(-((X[:, 0] - 0.5)**2 + (X[:, 1] - 0.5)**2) / 0.1) + 0.05 * np.random.randn(n_samples)
    
    # Classification target: leak detection (binary, 1=leak, 0=no leak)
    y_clf = (y_reg > 0.3).astype(int)
    
    # Spatial coordinates for maps
    lat = 37.7749 + 0.01 * X[:, 0]  # San Francisco approximate coordinates
    lon = -122.4194 + 0.01 * X[:, 1]
    
    return X, y_clf, y_reg, lat, lon

def classification_example():
    """
    Demonstrate classification model evaluation.
    """
    print("\n=== Classification Model Evaluation Example ===")
    
    # Load data
    X, y_clf, _, lat, lon = load_sample_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.3, random_state=42)
    
    # Train a classifier
    print("Training methane leak classifier...")
    classifier = MethaneLeakClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    
    # Create evaluator and evaluate model
    print("Evaluating classification performance...")
    evaluator = ClassificationEvaluator(model=classifier)
    metrics = evaluator.evaluate(X_test, y_test)
    
    # Print evaluation report
    evaluator.print_report()
    
    # Plot confusion matrix
    print("Generating confusion matrix plot...")
    fig_cm, _ = evaluator.plot_confusion_matrix(normalize=True)
    fig_cm.savefig('outputs/confusion_matrix.png', bbox_inches='tight', dpi=300)
    
    # Plot ROC curve
    print("Generating ROC curve plot...")
    fig_roc, _ = evaluator.plot_roc_curve(X_test, y_test)
    fig_roc.savefig('outputs/roc_curve.png', bbox_inches='tight', dpi=300)
    
    print("Classification evaluation complete. Visualizations saved to 'outputs' directory.")
    
    return classifier, evaluator, X_test, y_test

def regression_example():
    """
    Demonstrate regression model evaluation.
    """
    print("\n=== Regression Model Evaluation Example ===")
    
    # Load data
    X, _, y_reg, lat, lon = load_sample_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)
    
    # Train a regressor (using RandomForestRegressor as a stand-in for QGPR)
    print("Training regression model...")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    
    # Create evaluator and evaluate model
    print("Evaluating regression performance...")
    evaluator = RegressionEvaluator(model=regressor)
    metrics = evaluator.evaluate(X_test, y_test)
    
    # Print evaluation report
    evaluator.print_report()
    
    # Plot predictions
    print("Generating prediction scatter plot...")
    fig_pred, _ = evaluator.plot_predictions()
    fig_pred.savefig('outputs/prediction_scatter.png', bbox_inches='tight', dpi=300)
    
    # Plot residuals
    print("Generating residual plots...")
    fig_resid = evaluator.plot_residuals()
    fig_resid.savefig('outputs/residuals.png', bbox_inches='tight', dpi=300)
    
    # Make predictions for geospatial visualization
    test_indices = np.random.choice(range(len(X)), size=100, replace=False)
    X_map = X[test_indices]
    y_true = y_reg[test_indices]
    y_pred = regressor.predict(X_map)
    lat_map = lat[test_indices]
    lon_map = lon[test_indices]
    
    # Generate artificial uncertainty (would come from QGPR model in real use)
    uncertainties = 0.1 * np.abs(y_pred - y_true) + 0.05 * np.random.rand(len(y_pred))
    
    # Plot geospatial error map
    print("Generating geospatial error maps...")
    fig_geo = plot_geospatial_errors(lat_map, lon_map, y_true, y_pred, uncertainties)
    fig_geo.savefig('outputs/geospatial_errors.png', bbox_inches='tight', dpi=300)
    
    print("Regression evaluation complete. Visualizations saved to 'outputs' directory.")
    
    return regressor, evaluator, X_test, y_test

if __name__ == "__main__":
    print("Running model evaluation examples...")
    
    # Run classification example
    clf_model, clf_evaluator, clf_X_test, clf_y_test = classification_example()
    
    # Run regression example
    reg_model, reg_evaluator, reg_X_test, reg_y_test = regression_example()
    
    print("\nAll examples completed successfully. Evaluation visualizations saved to 'outputs' directory.") 