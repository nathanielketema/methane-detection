import os
import pandas as pd

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
FILE_NAME = "mehane_data.csv"
PROCESSED_FILE = "processed_methane_data.csv"

def load_raw_data():
    full_path = os.path.join(RAW_DATA_PATH, FILE_NAME)
    
    try:
        data = pd.read_csv(full_path)
        print(f"Loaded raw data from {full_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e
    
    return data

def clean_data(data):
    # todo
    cleaned_data = data.dropna()
    
    print("Data cleaned: dropped missing values")
    return cleaned_data

def process_data():
    raw_data = load_raw_data()
    processed_data = clean_data(raw_data)
    
    # Ensure the processed data directory exists
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    output_file = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILE)
    processed_data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
    return processed_data

if __name__ == "__main__":
    process_data()
