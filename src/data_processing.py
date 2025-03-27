import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, FILE_NAME, DATA_PROCESSING_CONFIG

def validate_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the input data against required columns and value ranges.
    
    Parameters:
        data (pd.DataFrame): Input data to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
    """
    errors = []
    
    # Check required columns
    missing_columns = [col for col in DATA_PROCESSING_CONFIG["required_columns"] 
                      if col not in data.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check data types
    for col in data.columns:
        if col in DATA_PROCESSING_CONFIG["column_ranges"]:
            if not pd.api.types.is_numeric_dtype(data[col]):
                errors.append(f"Column {col} must be numeric")
            else:
                # Check value ranges
                min_val, max_val = DATA_PROCESSING_CONFIG["column_ranges"][col]
                if data[col].min() < min_val or data[col].max() > max_val:
                    errors.append(f"Column {col} contains values outside valid range [{min_val}, {max_val}]")
    
    # Check for missing values
    missing_values = data[DATA_PROCESSING_CONFIG["required_columns"]].isnull().sum()
    if missing_values.any():
        errors.append(f"Missing values found in columns: {missing_values[missing_values > 0].to_dict()}")
    
    return len(errors) == 0, errors

def load_raw_data(file_name=None):
    """
    Load raw data from the configured raw data path.
    
    Parameters:
        file_name (str, optional): Name of the file to load. If None, uses the default from config.
        
    Returns:
        pd.DataFrame: The loaded raw data
    """
    if file_name is None:
        file_name = FILE_NAME
        
    full_path = os.path.join(RAW_DATA_PATH, file_name)
    
    try:
        data = pd.read_csv(full_path)
        print(f"Loaded raw data from {full_path}")
        
        # Validate the loaded data
        is_valid, errors = validate_data(data)
        if not is_valid:
            print("Data validation errors:")
            for error in errors:
                print(f"  - {error}")
            raise ValueError("Data validation failed")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e
    
    return data

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning operations on the data.
    
    Parameters:
        data (pd.DataFrame): Raw data to clean
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    # Create a copy to avoid modifying the original data
    cleaned_data = data.copy()
    
    # Drop rows with missing values
    cleaned_data = cleaned_data.dropna()
    
    # Drop constant columns
    cleaned_data = cleaned_data.drop(columns=['precipitation_rate'])
    print("Removed precipitation_rate column (constant zero value)")
    
    # Normalize column names
    cleaned_data.columns = [col.strip().lower() for col in cleaned_data.columns]
    
    # Validate the cleaned data
    is_valid, errors = validate_data(cleaned_data)
    if not is_valid:
        print("Data validation errors after cleaning:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Data validation failed after cleaning")
    
    print("Data cleaned and validated")
    return cleaned_data

def process_data(save_output=True):
    """
    Process the raw data by loading and cleaning.
    
    Parameters:
        save_output (bool): Whether to save the processed data to disk
        
    Returns:
        pd.DataFrame: The processed data
    """
    raw_data = load_raw_data()
    processed_data = clean_data(raw_data)
    
    if save_output:
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        output_file = os.path.join(PROCESSED_DATA_PATH, "processed_methane_data.csv")
        processed_data.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
    
    return processed_data

if __name__ == "__main__":
    print("Data processing module - processing raw methane leak detection data")
    processed_data = process_data()
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Columns: {processed_data.columns.tolist()}")
