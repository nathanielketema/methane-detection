import pandas as pd
import numpy as np
import os
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

if __name__ == "__main__":
    # Load processed data
    file_path = os.path.join(PROCESSED_DATA_PATH, "processed_methane_data.csv")
    data = pd.read_csv(file_path)
    
    # Extract features
    print("Extracting features...")
    featured_data = extract_features(data)
    
    # Save the featured data
    output_path = os.path.join(PROCESSED_DATA_PATH, "featured_data.csv")
    featured_data.to_csv(output_path, index=False)
    print(f"Featured data saved to {output_path}")
    
    # Print basic statistics
    print(f"\nFeatured data statistics:")
    print(f"  Total samples: {len(featured_data)}")
    print(f"  Total features: {len(featured_data.columns)}")
    print(f"  Samples with tracer_concentration > 0: {sum(featured_data['tracer_concentration'] > 0)} ({sum(featured_data['tracer_concentration'] > 0)/len(featured_data):.2%})")
