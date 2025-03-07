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
    # Test the planner with dummy candidate data.
    import matplotlib.pyplot as plt
    
    # Dummy candidate data including navigation coordinates and some features.
    candidate_data = pd.DataFrame({
        'elapsed_minutes': [30, 45, 60],
        'wind_speed': [2.5, 3.0, 1.5],
        'temperature (C)': [22, 23, 21],
        'latitude': [35.0, 35.1, 35.2],
        'longitude': [-120.0, -120.1, -120.2]
    })
    
    # Create a dummy regression model and train it on a small dummy dataset.
    model = QuantumGaussianRegression()
    import numpy as np
    X_dummy = np.array([[10, 2.0, 20], [20, 2.5, 21], [30, 3.0, 22], [40, 3.5, 23]])
    y_dummy = np.array([0.1, 0.3, 0.5, 0.7])
    model.fit(X_dummy, y_dummy)
    
    selected_candidate, pred, uncert = plan_next_location(model, candidate_data)
    print("Selected candidate:")
    print(selected_candidate)
    print("Predicted tracer concentration:", pred)
    print("Prediction uncertainty:", uncert)
