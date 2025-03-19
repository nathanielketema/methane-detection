import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import PROCESSED_DATA_PATH

def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function will:
      - Convert the "time" column into a datetime object
      - Create time-based features: elapsed minutes since simulation start, hour, and minute
      - Compute wind speed and wind direction
      - Create distance-based features from grid points
      - Create interaction features for important meteorological variables
    """ 

    data['time'] = pd.to_datetime(data['time'])
    simulation_start = data['time'].min()
    data['elapsed_minutes'] = (data['time'] - simulation_start).dt.total_seconds() / 60.0
    data['hour'] = data['time'].dt.hour
    data['minute'] = data['time'].dt.minute
    
    # Add cyclical time features using sine and cosine transformations
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    
    # Process wind features using the normalized column names
    if 'u_west_to_east_wind' in data.columns and 'v_south_to_north_wind' in data.columns:
        data['wind_speed'] = np.sqrt(data['u_west_to_east_wind']**2 + data['v_south_to_north_wind']**2)
        data['wind_direction'] = (np.degrees(np.arctan2(data['v_south_to_north_wind'], data['u_west_to_east_wind'])) + 360) % 360
        
        # Add wind components in cardinal directions
        data['wind_northerly'] = data['v_south_to_north_wind'] 
        data['wind_easterly'] = data['u_west_to_east_wind']
    else:
        print("Warning: 'u_west_to_east_wind' and/or 'v_south_to_north_wind' columns not found in data. Skipping wind feature extraction.")
    
    # Create spatial grid distance features if coordinates are available
    if 'i_value' in data.columns and 'j_value' in data.columns:
        # Calculate distance from origin point (0,0) in grid coordinates
        data['grid_distance'] = np.sqrt(data['i_value']**2 + data['j_value']**2)
    
    # Create interaction features between key meteorological variables
    if all(col in data.columns for col in ['temperature', 'relative_humidity', 'wind_speed']):
        # Wind chill effect (temperature and wind)
        data['temp_wind_interaction'] = data['temperature'] * data['wind_speed']
        
        # Heat index effect (temperature and humidity)
        data['temp_humidity_interaction'] = data['temperature'] * data['relative_humidity']
    
    # Add stability indicators using available atmospheric data
    if all(col in data.columns for col in ['temperature', 'vertical_velocity']):
        # Simple atmospheric stability indicator
        data['atm_stability'] = data['temperature'] * data['vertical_velocity']
    
    return data

def prepare_train_data(data: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Prepares the data for model training by:
    1. Separating features and target variables
    2. Splitting into training and testing sets
    3. Scaling the features
    
    Returns dictionaries containing data for both classification and regression tasks.
    """
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Ensure required target columns exist
    if 'tracer_concentration' not in df.columns:
        raise ValueError("tracer_concentration column not found in data")
    
    if 'leakage' not in df.columns:
        # Create leakage target if not already present
        df['leakage'] = (df['tracer_concentration'] > 0).astype(int)
        print("Created leakage target variable")
    
    # Identify which columns should not be used as features
    non_feature_cols = ['time', 'tracer_concentration', 'leakage']
    
    # For regression, we only want samples where leakage=1
    regression_df = df[df['leakage'] == 1].copy()
    
    # Create feature matrices and target vectors
    X_all = df.drop(columns=non_feature_cols, errors='ignore')
    y_class = df['leakage']
    X_reg = regression_df.drop(columns=non_feature_cols, errors='ignore')
    y_reg = regression_df['tracer_concentration']
    
    # Split data for classification
    X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
        X_all, y_class, test_size=test_size, random_state=random_state, stratify=y_class
    )
    
    # Split data for regression (only using samples where leakage=1)
    if len(X_reg) > 0:
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
            X_reg, y_reg, test_size=test_size, random_state=random_state
        )
    else:
        print("Warning: No samples with leakage=1 for regression training")
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = None, None, None, None
    
    # Scale the features
    scaler_class = StandardScaler()
    X_class_train_scaled = scaler_class.fit_transform(X_class_train)
    X_class_test_scaled = scaler_class.transform(X_class_test)
    
    classification_data = {
        'X_train': X_class_train_scaled,
        'X_test': X_class_test_scaled,
        'y_train': y_class_train,
        'y_test': y_class_test,
        'feature_names': X_class_train.columns.tolist(),
        'scaler': scaler_class
    }
    
    # Scale features for regression if regression data exists
    regression_data = None
    if X_reg_train is not None:
        scaler_reg = StandardScaler()
        X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
        X_reg_test_scaled = scaler_reg.transform(X_reg_test)
        
        regression_data = {
            'X_train': X_reg_train_scaled,
            'X_test': X_reg_test_scaled,
            'y_train': y_reg_train,
            'y_test': y_reg_test,
            'feature_names': X_reg_train.columns.tolist(),
            'scaler': scaler_reg
        }
    
    return {
        'classification': classification_data,
        'regression': regression_data
    }

def save_prepared_data(prepared_data, output_dir=None):
    """
    Saves the prepared data and scalers for both classification and regression tasks.
    
    Args:
        prepared_data: Dictionary containing the prepared data
        output_dir: Directory to save the data (defaults to PROCESSED_DATA_PATH if None)
    """
    
    if output_dir is None:
        output_dir = PROCESSED_DATA_PATH
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save classification data
    class_data = prepared_data['classification']
    joblib.dump(class_data['scaler'], os.path.join(output_dir, 'classification_scaler.pkl'))
    
    np.save(os.path.join(output_dir, 'X_class_train.npy'), class_data['X_train'])
    np.save(os.path.join(output_dir, 'X_class_test.npy'), class_data['X_test'])
    np.save(os.path.join(output_dir, 'y_class_train.npy'), class_data['y_train'])
    np.save(os.path.join(output_dir, 'y_class_test.npy'), class_data['y_test'])
    
    # Save feature names
    with open(os.path.join(output_dir, 'classification_feature_names.txt'), 'w') as f:
        f.write('\n'.join(class_data['feature_names']))
    
    # Save regression data if available
    if prepared_data['regression'] is not None:
        reg_data = prepared_data['regression']
        joblib.dump(reg_data['scaler'], os.path.join(output_dir, 'regression_scaler.pkl'))
        
        np.save(os.path.join(output_dir, 'X_reg_train.npy'), reg_data['X_train'])
        np.save(os.path.join(output_dir, 'X_reg_test.npy'), reg_data['X_test'])
        np.save(os.path.join(output_dir, 'y_reg_train.npy'), reg_data['y_train'])
        np.save(os.path.join(output_dir, 'y_reg_test.npy'), reg_data['y_test'])
        
        # Save feature names
        with open(os.path.join(output_dir, 'regression_feature_names.txt'), 'w') as f:
            f.write('\n'.join(reg_data['feature_names']))
    
    print(f"Prepared data saved to {output_dir}")

if __name__ == "__main__":
    # Load processed data
    file_path = os.path.join(PROCESSED_DATA_PATH, "processed_methane_data.csv")
    data = pd.read_csv(file_path)
    
    # Extract features
    print("Extracting features...")
    featured_data = extract_features(data)
    
    # Prepare data for training
    print("Preparing training data...")
    prepared_data = prepare_train_data(featured_data)
    
    # Save prepared data
    print("Saving prepared data...")
    save_prepared_data(prepared_data)
    
    # Print statistics
    cls_data = prepared_data['classification']
    print(f"\nClassification data statistics:")
    print(f"  Training samples: {cls_data['X_train'].shape[0]}")
    print(f"  Testing samples: {cls_data['X_test'].shape[0]}")
    print(f"  Features: {cls_data['X_train'].shape[1]}")
    print(f"  Positive class (leakage=1) in training: {sum(cls_data['y_train'])} ({sum(cls_data['y_train'])/len(cls_data['y_train']):.2%})")
    
    if prepared_data['regression'] is not None:
        reg_data = prepared_data['regression']
        print(f"\nRegression data statistics:")
        print(f"  Training samples: {reg_data['X_train'].shape[0]}")
        print(f"  Testing samples: {reg_data['X_test'].shape[0]}")
        print(f"  Features: {reg_data['X_train'].shape[1]}")
        print(f"  Target min: {min(reg_data['y_train']):.6f}, max: {max(reg_data['y_train']):.6f}")
    else:
        print("\nNo regression data available (no samples with leakage=1)")
