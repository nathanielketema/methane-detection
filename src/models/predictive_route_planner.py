import numpy as np
import pandas as pd
from src.models.quantum_gaussian_regression import QuantumGaussianRegression

def plan_next_location(model: QuantumGaussianRegression, candidate_features: pd.DataFrame):
    """
    Given a trained model and candidate features DataFrame, predict tracer concentration for each candidate
    and select the candidate with the highest predicted value.
    
    Parameters:
      model (QuantumGaussianRegression): The regression model that has been trained.
      candidate_features (pd.DataFrame): DataFrame that must include the features used by the model.
        It should also contain 'latitude' and 'longitude' columns for navigation.
    
    Returns:
      selected_candidate (pd.Series): The row of the candidate with the highest predicted concentration.
      prediction (float): The predicted tracer concentration for that candidate.
      uncertainty (float): The standard deviation (uncertainty) of the prediction.
    """
    # Extract features for prediction; assume that the model expects all columns except for latitude/longitude.
    features = candidate_features.drop(columns=['latitude', 'longitude'])
    X_candidate = features.values  # Convert to numpy array for prediction
    
    # Get predictions and uncertainties
    predictions, uncertainties = model.predict(X_candidate)
    
    # Select candidate with the highest predicted tracer concentration
    max_idx = np.argmax(predictions)
    selected_candidate = candidate_features.iloc[max_idx]
    
    return selected_candidate, predictions[max_idx], uncertainties[max_idx]

if __name__ == "__main__":
    print("Predictive Route Planner module - this module helps determine the best location for drone navigation")
    print("Import and use this module in other scripts rather than running it directly.")
