import os
import pandas as pd
import numpy as np

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
    if 'tracer_concentration' not in features.columns:
        raise ValueError("Column 'tracer_concentration' not found in data.")
    
    # 3. Prepare training data
    print("Preparing training data...")
    
    # Define feature columns for model training
    feature_columns = ['time_numeric', 'u_west_to_east_wind', 'temprature']
    
    # Sample a smaller subset of data for training
    sample_size = min(10000, len(features))
    features = features.sample(n=sample_size, random_state=42)
    
    # Convert time column to numeric (seconds since first timestamp)
    features['time_numeric'] = pd.to_datetime(features['time']).astype(np.int64) // 10**9
    features['time_numeric'] = features['time_numeric'] - features['time_numeric'].min()
    
    # Get complete rows only and ensure we have at least 5 valid points
    required_columns = ['latitude', 'longitude', 'time', 'u_west_to_east_wind', 'temprature']
    
    # First ensure we have enough valid rows in processed_data
    valid_data = processed_data.dropna(subset=required_columns)
    
    # Sort by time to ensure we get the first 5 chronological points
    valid_data = valid_data.sort_values('time')
    
    if len(valid_data) < 5:
        raise ValueError("Not enough valid data points for navigation")
        
    # Convert time to numeric for the first 5 points
    valid_times = pd.to_datetime(valid_data['time'].iloc[:5]).astype(np.int64) // 10**9
    time_numeric = valid_times - valid_times.min()
    
    navigation_data = {
        'latitude': valid_data['latitude'].iloc[:5],
        'longitude': valid_data['longitude'].iloc[:5],
        'time_numeric': time_numeric,
        'u_west_to_east_wind': valid_data['u_west_to_east_wind'].iloc[:5],
        'temprature': valid_data['temprature'].iloc[:5]
    }
    
    # Verify no NaN values in navigation data
    nav_df = pd.DataFrame(navigation_data)
    if nav_df.isna().any().any():
        print("Columns with NaN values:", nav_df.columns[nav_df.isna().any()].tolist())
        raise ValueError("NaN values found in navigation data")
    
    # Prepare model input data
    X = features[feature_columns].values
    y = features['tracer_concentration'].values
    
    # Clear features DataFrame to free memory
    del features
    
    # 4. Train the regression model
    print("Training the regression model...")
    model = QuantumGaussianRegression()
    model.fit(X, y)
    
    # 5. Simulate drone navigation decision
    print("Simulating drone navigation...")
    current_position = (navigation_data['latitude'].iloc[0], navigation_data['longitude'].iloc[0])
    next_waypoint, prediction, uncertainty = simulate_drone_navigation(model, pd.DataFrame(navigation_data), current_position)
    
    # 6. Visualization
    print("Generating visualization of methane distribution...")
    # Use the earliest time slice for the static plot if available.
    if 'Time (UTC)' in processed_data.columns:
        sample_time = pd.to_datetime(processed_data['Time (UTC)']).min()
        plot_methane_distribution(processed_data, time_filter=sample_time, tracer_column='tracer_concentration')
    else:
        plot_methane_distribution(processed_data, tracer_column='tracer_concentration')
    
    print("Project execution complete.")

if __name__ == "__main__":
    main()
