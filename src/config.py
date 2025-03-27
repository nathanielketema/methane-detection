import os
from typing import Dict, List, Any

# Data paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
FILE_NAME = "methane_data.csv"

# Data processing parameters
DATA_PROCESSING_CONFIG: Dict[str, Any] = {
    "required_columns": [
        "time",
        "latitude",
        "longitude",
        "tracer_concentration",
        "u_west_to_east_wind",
        "v_south_to_north_wind",
        "temperature",
        "relative_humidity",
        "vertical_velocity"
    ],
    "column_ranges": {
        "temperature": (-50, 50),  # Celsius
        "wind_speed": (0, 100),    # m/s
        "relative_humidity": (0, 100),  # percentage
        "tracer_concentration": (0, float('inf'))  # ppm
    }
}

# Feature engineering parameters
FEATURE_ENGINEERING_CONFIG: Dict[str, Any] = {
    "time_features": {
        "include_hour": True,
        "include_minute": True,
        "include_elapsed": True
    },
    "wind_features": {
        "include_speed": True,
        "include_direction": True,
        "include_components": True
    },
    "spatial_features": {
        "include_grid_distance": True
    },
    "interaction_features": {
        "include_wind_chill": True,
        "include_heat_index": True,
        "include_stability": True
    },
    "feature_scaling": {
        "enabled": True,
        "method": "standard"  # options: "standard", "minmax", "robust"
    }
}

# Data preparation parameters
DATA_PREPARATION_CONFIG: Dict[str, Any] = {
    "sample_size": 10000,
    "n_navigation_points": 5,
    "random_state": 42,
    "feature_columns": [
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
}

# Model parameters
MODEL_CONFIG: Dict[str, Any] = {
    "quantum_gaussian_regression": {
        "alpha": 1e-10,
        "n_qubits": None,  # Will be determined from data
        "shots": 1024,
        "normalize_y": False,
        "model_dir": "models",
        "version": "1.0.0"
    },
    "cross_validation": {
        "enabled": True,
        "n_splits": 5,
        "scoring": "neg_mean_squared_error"
    }
}

# Create necessary directories
def create_directories():
    """Create all necessary directories for the project."""
    directories = [
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        "models",
        "models/training_data",
        "outputs",
        "evaluation_results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

# Create directories when module is imported
create_directories()
