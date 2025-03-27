import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

def balance_data(data: pd.DataFrame, target_col: str = 'tracer_concentration') -> pd.DataFrame:
    """
    Balance the dataset to have equal number of samples with positive and zero values.
    
    Parameters:
        data (pd.DataFrame): Input data
        target_col (str): Column name for the target variable
        
    Returns:
        pd.DataFrame: Balanced dataset
    """
    positive_samples = data[data[target_col] > 0]
    zero_samples = data[data[target_col] == 0]
    
    print(f"Original data distribution:")
    print(f"  - Samples with {target_col} > 0: {len(positive_samples)}")
    print(f"  - Samples with {target_col} = 0: {len(zero_samples)}")
    
    target_size = min(len(positive_samples), len(zero_samples))
    
    if len(positive_samples) > target_size:
        positive_samples = positive_samples.sample(n=target_size, random_state=42)
    
    if len(zero_samples) > target_size:
        zero_samples = zero_samples.sample(n=target_size, random_state=42)
    
    balanced_data = pd.concat([positive_samples, zero_samples])
    
    # Shuffle the data
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced data distribution:")
    print(f"  - Samples with {target_col} > 0: {len(balanced_data[balanced_data[target_col] > 0])}")
    print(f"  - Samples with {target_col} = 0: {len(balanced_data[balanced_data[target_col] == 0])}")
    
    return balanced_data

def get_feature_columns() -> List[str]:
    """
    Get the list of feature columns to use for model training.
    
    Returns:
        List[str]: List of feature column names
    """
    return [
        'time_numeric',
        'u_west_to_east_wind',
        'v_south_to_north_wind',
        'temperature',
        'relative_humidity',
        'wind_speed',
        'wind_direction',
        'grid_distance',
        'hour_sin',
        'hour_cos'
    ]

def prepare_navigation_data(data: pd.DataFrame, n_points: int = 5) -> pd.DataFrame:
    """
    Prepare navigation data for drone routing.
    
    Parameters:
        data (pd.DataFrame): Input data
        n_points (int): Number of points to use for navigation
        
    Returns:
        pd.DataFrame: Navigation data
    """
    required_columns = ['latitude', 'longitude', 'time', 'u_west_to_east_wind', 'temperature']
    
    # Ensure we have enough valid rows
    valid_data = data.dropna(subset=required_columns)
    valid_data = valid_data.sort_values('time')
    
    if len(valid_data) < n_points:
        raise ValueError(f"Not enough valid data points for navigation. Need {n_points}, got {len(valid_data)}")
    
    # Convert time to numeric for the first n_points
    valid_times = pd.to_datetime(valid_data['time'].iloc[:n_points]).astype(np.int64) // 10**9
    time_numeric = valid_times - valid_times.min()
    
    navigation_data = {
        'latitude': valid_data['latitude'].iloc[:n_points],
        'longitude': valid_data['longitude'].iloc[:n_points],
        'time_numeric': time_numeric,
        'u_west_to_east_wind': valid_data['u_west_to_east_wind'].iloc[:n_points],
        'temperature': valid_data['temperature'].iloc[:n_points]
    }
    
    # Create DataFrame and verify no NaN values
    nav_df = pd.DataFrame(navigation_data)
    if nav_df.isna().any().any():
        print("Columns with NaN values:", nav_df.columns[nav_df.isna().any()].tolist())
        raise ValueError("NaN values found in navigation data")
    
    return nav_df

def prepare_training_data(features: pd.DataFrame, processed_data: pd.DataFrame, 
                         sample_size: int = 10000) -> Dict[str, Any]:
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
    # Balance the data
    balanced_data = balance_data(features)
    
    # Sample a smaller subset of data for training
    sample_size = min(sample_size, len(balanced_data))
    balanced_data = balanced_data.sample(n=sample_size, random_state=42)
    
    # Convert time column to numeric (seconds since first timestamp)
    balanced_data['time_numeric'] = pd.to_datetime(balanced_data['time']).astype(np.int64) // 10**9
    balanced_data['time_numeric'] = balanced_data['time_numeric'] - balanced_data['time_numeric'].min()
    
    # Get feature columns
    feature_columns = get_feature_columns()
    
    # Prepare navigation data
    nav_df = prepare_navigation_data(processed_data)
    
    # Prepare model input data
    X = balanced_data[feature_columns].values
    y = balanced_data['tracer_concentration'].values
    y_binary = (y > 0).astype(int)
    
    return {
        'X': X,
        'y': y,
        'y_binary': y_binary,
        'feature_columns': feature_columns,
        'nav_df': nav_df
    } 