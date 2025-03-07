import os
import pandas as pd

from src.data_processing import process_data
from src.feature_engineering import extract_features
from src.models.quantum_gaussian_regression import QuantumGaussianRegression
from src.drone_navigation import simulate_drone_navigation
from src.visualization import plot_methane_distribution

def main():
    print("Starting Methane Leak Detection Project...")

    # 1. Process the raw data
    print("Processing raw data...")
    processed_data = process_data()

    # 2. Feature Engineering
    print("Extracting features from processed data...")
    features = extract_features(processed_data)
    
    # Check that target variable exists
    if 'tracer concentration' not in features.columns:
        raise ValueError("Column 'tracer concentration' not found in data.")
    
    # 3. Prepare training data
    # For demonstration, we'll assume the following columns are features:
    # 'elapsed_minutes', 'wind_speed', 'temperature (C)'.
    # We drop columns that aren't used as features (like spatial coordinates and time).
    feature_columns = ['elapsed_minutes', 'wind_speed', 'temperature (C)']
    # Ensure these columns exist in the dataset (add more if needed)
    missing = [col for col in feature_columns if col not in features.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    X = features[feature_columns].values
    y = features['tracer concentration'].values
    
    # 4. Train the regression model
    print("Training the regression model...")
    model = QuantumGaussianRegression()
    model.fit(X, y)
    
    # 5. Simulate drone navigation decision
    # For this demonstration, use a subset of candidate locations.
    candidate_data = features[['elapsed_minutes', 'wind_speed', 'temperature (C)', 'latitude', 'longitude']].head(5)
    current_position = (features['latitude'].iloc[0], features['longitude'].iloc[0])
    print("Simulating drone navigation...")
    next_waypoint, prediction, uncertainty = simulate_drone_navigation(model, candidate_data, current_position)
    
    # 6. Visualization
    print("Generating visualization of methane distribution...")
    # Use the earliest time slice for the static plot if available.
    if 'Time (UTC)' in processed_data.columns:
        sample_time = pd.to_datetime(processed_data['Time (UTC)']).min()
        plot_methane_distribution(processed_data, time_filter=sample_time)
    else:
        plot_methane_distribution(processed_data)
    
    print("Project execution complete.")

if __name__ == "__main__":
    main()
