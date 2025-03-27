import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
import os
from src.config import PROCESSED_DATA_PATH, FEATURE_ENGINEERING_CONFIG

def analyze_feature_importance(data: pd.DataFrame, target_col: str = 'tracer_concentration') -> Dict[str, float]:
    """
    Analyze feature importance using Random Forest.
    
    Parameters:
        data (pd.DataFrame): Input data with features
        target_col (str): Name of the target column
        
    Returns:
        Dict[str, float]: Dictionary mapping feature names to importance scores
    """
    # Prepare features and target
    feature_cols = [col for col in data.columns if col != target_col]
    X = data[feature_cols]
    y = data[target_col]
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = dict(zip(feature_cols, rf.feature_importances_))
    
    # Sort by importance
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    return importances

def scale_features(data: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Scale features using the configured scaling method.
    
    Parameters:
        data (pd.DataFrame): Input data
        feature_cols (List[str]): List of feature columns to scale
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (Scaled data, scaler parameters)
    """
    if not FEATURE_ENGINEERING_CONFIG["feature_scaling"]["enabled"]:
        return data, {}
    
    # Create a copy to avoid modifying the original data
    scaled_data = data.copy()
    
    # Select scaling method
    method = FEATURE_ENGINEERING_CONFIG["feature_scaling"]["method"]
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Scale features
    scaled_data[feature_cols] = scaler.fit_transform(scaled_data[feature_cols])
    
    # Store scaler parameters
    scaler_params = {
        "method": method,
        "mean": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
        "scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
        "min": scaler.min_.tolist() if hasattr(scaler, 'min_') else None,
        "data_min": scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
        "data_max": scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None,
        "data_range": scaler.data_range_.tolist() if hasattr(scaler, 'data_range_') else None
    }
    
    return scaled_data, scaler_params

def create_leakage_feature(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create the leakage feature based on tracer concentration.
    
    Parameters:
        data (pd.DataFrame): Input data with tracer_concentration column
        
    Returns:
        pd.DataFrame: Data with added leakage feature
    """
    data['leakage'] = (data['tracer_concentration'] > 0).astype(int)
    print("Added 'leakage' column (1 if tracer_concentration > 0, 0 otherwise)")
    return data

def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the time column.
    
    Parameters:
        data (pd.DataFrame): Input data with time column
        
    Returns:
        pd.DataFrame: Data with added time features
    """
    data['time'] = pd.to_datetime(data['time'])
    simulation_start = data['time'].min()
    
    if FEATURE_ENGINEERING_CONFIG["time_features"]["include_elapsed"]:
        data['elapsed_minutes'] = (data['time'] - simulation_start).dt.total_seconds() / 60.0
    
    if FEATURE_ENGINEERING_CONFIG["time_features"]["include_hour"]:
        data['hour'] = data['time'].dt.hour
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    
    if FEATURE_ENGINEERING_CONFIG["time_features"]["include_minute"]:
        data['minute'] = data['time'].dt.minute
    
    return data

def create_wind_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create wind-related features from wind components.
    
    Parameters:
        data (pd.DataFrame): Input data with wind component columns
        
    Returns:
        pd.DataFrame: Data with added wind features
    """
    if 'u_west_to_east_wind' in data.columns and 'v_south_to_north_wind' in data.columns:
        if FEATURE_ENGINEERING_CONFIG["wind_features"]["include_speed"]:
            data['wind_speed'] = np.sqrt(data['u_west_to_east_wind']**2 + data['v_south_to_north_wind']**2)
        
        if FEATURE_ENGINEERING_CONFIG["wind_features"]["include_direction"]:
            data['wind_direction'] = (np.degrees(np.arctan2(data['v_south_to_north_wind'], data['u_west_to_east_wind'])) + 360) % 360
        
        if FEATURE_ENGINEERING_CONFIG["wind_features"]["include_components"]:
            data['wind_northerly'] = data['v_south_to_north_wind'] 
            data['wind_easterly'] = data['u_west_to_east_wind']
    else:
        print("Warning: 'u_west_to_east_wind' and/or 'v_south_to_north_wind' columns not found in data. Skipping wind feature extraction.")
    
    return data

def create_spatial_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create spatial features from grid coordinates.
    
    Parameters:
        data (pd.DataFrame): Input data with grid coordinates
        
    Returns:
        pd.DataFrame: Data with added spatial features
    """
    if FEATURE_ENGINEERING_CONFIG["spatial_features"]["include_grid_distance"]:
        if 'i_value' in data.columns and 'j_value' in data.columns:
            data['grid_distance'] = np.sqrt(data['i_value']**2 + data['j_value']**2)
    return data

def create_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between meteorological variables.
    
    Parameters:
        data (pd.DataFrame): Input data with meteorological variables
        
    Returns:
        pd.DataFrame: Data with added interaction features
    """
    if FEATURE_ENGINEERING_CONFIG["interaction_features"]["include_wind_chill"]:
        if all(col in data.columns for col in ['temperature', 'relative_humidity', 'wind_speed']):
            data['temp_wind_interaction'] = data['temperature'] * data['wind_speed']
            data['temp_humidity_interaction'] = data['temperature'] * data['relative_humidity']
    
    if FEATURE_ENGINEERING_CONFIG["interaction_features"]["include_stability"]:
        if all(col in data.columns for col in ['temperature', 'vertical_velocity']):
            data['atm_stability'] = data['temperature'] * data['vertical_velocity']
    
    return data

def extract_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract all features from the input data.
    
    Parameters:
        data (pd.DataFrame): Input data
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (Data with extracted features, feature metadata)
    """
    # Create a copy to avoid modifying the original data
    featured_data = data.copy()
    
    # Create all features
    featured_data = create_leakage_feature(featured_data)
    featured_data = create_time_features(featured_data)
    featured_data = create_wind_features(featured_data)
    featured_data = create_spatial_features(featured_data)
    featured_data = create_interaction_features(featured_data)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(featured_data)
    
    # Scale features if enabled
    feature_cols = [col for col in featured_data.columns if col != 'tracer_concentration']
    scaled_data, scaler_params = scale_features(featured_data, feature_cols)
    
    # Prepare feature metadata
    feature_metadata = {
        "feature_importance": feature_importance,
        "scaler_params": scaler_params,
        "feature_columns": feature_cols
    }
    
    return scaled_data, feature_metadata

if __name__ == "__main__":
    # Load processed data
    file_path = os.path.join(PROCESSED_DATA_PATH, "processed_methane_data.csv")
    data = pd.read_csv(file_path)
    
    # Extract features
    print("Extracting features...")
    featured_data, feature_metadata = extract_features(data)
    
    # Save the featured data
    output_path = os.path.join(PROCESSED_DATA_PATH, "featured_data.csv")
    featured_data.to_csv(output_path, index=False)
    print(f"Featured data saved to {output_path}")
    
    # Print feature importance
    print("\nFeature importance ranking:")
    for feature, importance in feature_metadata["feature_importance"].items():
        print(f"  {feature}: {importance:.4f}")
    
    # Print basic statistics
    print(f"\nFeatured data statistics:")
    print(f"  Total samples: {len(featured_data)}")
    print(f"  Total features: {len(featured_data.columns)}")
    print(f"  Samples with tracer_concentration > 0: {sum(featured_data['tracer_concentration'] > 0)} ({sum(featured_data['tracer_concentration'] > 0)/len(featured_data):.2%})")
