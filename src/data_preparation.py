import pandas as pd
import numpy as np

def prepare_training_data(features, processed_data, sample_size=10000):
    """
    Prepare training data for the methane leak detection models.
    
    Parameters:
        features (pd.DataFrame): Features extracted from processed data
        processed_data (pd.DataFrame): Processed sensor data
        sample_size (int): Maximum number of samples to use for training
    
    Returns:
        dict: Dictionary containing:
            - X: Feature matrix for model training
            - y: Regression target values (tracer concentration)
            - y_binary: Binary classification target (1 if leak, 0 if no leak)
            - feature_columns: List of feature column names
            - nav_df: DataFrame with navigation data for drone routing
    """
    # Define feature columns for model training
    feature_columns = ['time_numeric', 'u_west_to_east_wind', 'temprature']
    
    # Sample a smaller subset of data for training
    sample_size = min(sample_size, len(features))
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
    
    # Create binary target for leak detection (1 if concentration > 0, 0 otherwise)
    y_binary = (y > 0).astype(int)
    
    return {
        'X': X,
        'y': y,
        'y_binary': y_binary,
        'feature_columns': feature_columns,
        'nav_df': nav_df
    } 