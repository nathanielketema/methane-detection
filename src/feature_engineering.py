import pandas as pd
import numpy as np
import os
from src.config import RAW_DATA_PATH, FILE_NAME

def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and engineers features from the high-resolution methane dataset.
    
    Expected columns include:
      - "Time (UTC)" 
      - "i value", "j value"
      - "latitude", "longitude"
      - "u" (west to east wind, m/s) and "v" (south to north wind, m/s)
      - "temperature (C)"
      - "relative humidity (%)"
      - "vertical velocity (cm/s)"
      - "pressure (mb)"
      - "water vapor  (g/kg)"
      - "turbulent kinetic energy (m^2/s^2)"
      - "precipitation rate (mm/hr)"
      - "sensible heat flux (W/m^2)"
      - "Latent heat flux (W/m^2)"
      - "tracer concentration" (target variable)
    
    This function will:
      - Convert the "Time (UTC)" column into a datetime object.
      - Create time-based features: elapsed minutes since simulation start, hour, and minute.
      - Compute wind speed and wind direction.
    """
    
    # Normalize column names: remove whitespace and convert to lower case.
    data.columns = [col.strip().lower() for col in data.columns]
    print("Columns in data:", data.columns.tolist())
    
    # Process time-based features using the normalized 'time' column.
    if 'time' in data.columns:
        data['time'] = pd.to_datetime(data['time'])
        simulation_start = data['time'].min()
        data['elapsed_minutes'] = (data['time'] - simulation_start).dt.total_seconds() / 60.0
        data['hour'] = data['time'].dt.hour
        data['minute'] = data['time'].dt.minute
    else:
        print("Warning: 'time' column not found in data. Skipping time-based feature extraction.")
        data['elapsed_minutes'] = 0
        data['hour'] = 0
        data['minute'] = 0

    # Process wind features using the normalized column names.
    if 'u_west_to_east_wind' in data.columns and 'v_south_to_north_wind' in data.columns:
        data['wind_speed'] = np.sqrt(data['u_west_to_east_wind']**2 + data['v_south_to_north_wind']**2)
        data['wind_direction'] = (np.degrees(np.arctan2(data['v_south_to_north_wind'], data['u_west_to_east_wind'])) + 360) % 360
    else:
        print("Warning: 'u_west_to_east_wind' and/or 'v_south_to_north_wind' columns not found in data. Skipping wind feature extraction.")
    
    return data

if __name__ == "__main__":
    import os
    file_path = os.path.join("data", "processed", "processed_methane_data.csv")
    data = pd.read_csv(file_path)
    features = extract_features(data)
    print("Extracted features preview:")
    print(features.head())
