import os
import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, FILE_NAME

def load_raw_data():
    full_path = os.path.join(RAW_DATA_PATH, FILE_NAME)
    
    try:
        data = pd.read_csv(full_path)
        print(f"Loaded raw data from {full_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e
    
    return data

    
def balance_data(data):
    """
    Balance the dataset to have equal number of samples with 
    tracer_concentration > 0 and tracer_concentration = 0.
    """

    positive_tracer = data[data['tracer_concentration'] > 0]
    zero_tracer = data[data['tracer_concentration'] == 0]
    
    print(f"Original data distribution:")
    print(f"  - Samples with tracer_concentration > 0: {len(positive_tracer)}")
    print(f"  - Samples with tracer_concentration = 0: {len(zero_tracer)}")
    
    target_size = min(len(positive_tracer), len(zero_tracer))
    
    if len(positive_tracer) > target_size:
        positive_tracer = positive_tracer.sample(n=target_size, random_state=42)
    
    if len(zero_tracer) > target_size:
        zero_tracer = zero_tracer.sample(n=target_size, random_state=42)
    
    balanced_data = pd.concat([positive_tracer, zero_tracer])
    
    # Shuffle the data
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced data distribution:")
    print(f"  - Samples with tracer_concentration > 0: {len(balanced_data[balanced_data['tracer_concentration'] > 0])}")
    print(f"  - Samples with tracer_concentration = 0: {len(balanced_data[balanced_data['tracer_concentration'] == 0])}")
    
    return balanced_data


def clean_data(data):
    # Drop rows with missing values
    cleaned_data = data.dropna()
    
    cleaned_data = cleaned_data.drop(columns=['precipitation_rate'])
    print("Removed precipitation_rate column (constant zero value)")
    
    # Add a boolean column 'leakage' based on tracer_concentration
    cleaned_data['leakage'] = (cleaned_data['tracer_concentration'] > 0).astype(int)
    print("Added 'leakage' column (1 if tracer_concentration > 0, 0 otherwise)")
    
    cleaned_data = balance_data(cleaned_data)
    
    print("Data cleaned and balanced")
    return cleaned_data


def process_data():
    raw_data = load_raw_data()
    processed_data = clean_data(raw_data)
    
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    output_file = os.path.join(PROCESSED_DATA_PATH, "processed_methane_data.csv")
    processed_data.to_csv(output_file, index=False)
    print(f"Processed and balanced data saved to {output_file}")
    
    return processed_data

if __name__ == "__main__":
    process_data()
