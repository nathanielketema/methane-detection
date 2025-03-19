import pandas as pd
from src.config import RAW_DATA_PATH

from src.data_processing import process_data
from src.feature_engineering import extract_features
from src.data_preparation import prepare_training_data
from src.models.methane_leak_classifier import MethaneLeakClassifier
from src.models.quantum_gaussian_regression import QuantumGaussianRegression
from src.drone_navigation import simulate_drone_navigation
from src.visualization import plot_methane_distribution, animate_methane_distribution

def main():
    print("Starting Methane Leak Detection Project...")

    # 1. Process the raw data from data/raw directory
    print("Processing raw data from", RAW_DATA_PATH)
    processed_data = process_data()

    # 2. Feature Engineering
    print("Extracting features from processed data...")
    features = extract_features(processed_data)
    
    # 3. Prepare training data
    print("Preparing training data...")
    prepared_data = prepare_training_data(features, processed_data)
    
    # 4. Train the binary classification model to detect leak presence/absence
    print("Training the methane leak classifier...")
    classifier = MethaneLeakClassifier()
    classifier.fit(prepared_data['X'], prepared_data['y_binary'], feature_names=prepared_data['feature_columns'])
    
    # 5. Use the classifier to predict if there's a leak
    print("Detecting potential methane leaks...")
    # For demonstration, we'll use the navigation data for prediction
    nav_features = prepared_data['nav_df'][prepared_data['feature_columns']].values
    leak_predictions = classifier.predict(nav_features)
    
    leak_detected = any(leak_predictions == 1)
    print(f"Leak detection result: {'Leak detected!' if leak_detected else 'No leak detected.'}")
    
    if leak_detected:
        # 6. If leak is detected, train the regression model to predict concentration
        print("Leak detected! Training the regression model to predict methane concentration...")
        regression_model = QuantumGaussianRegression()
        regression_model.fit(prepared_data['X'], prepared_data['y'])
        
        # 7. Simulate drone navigation decision
        print("Simulating drone navigation...")
        current_position = (prepared_data['nav_df']['latitude'].iloc[0], prepared_data['nav_df']['longitude'].iloc[0])
        next_waypoint, prediction, uncertainty = simulate_drone_navigation(
            regression_model, prepared_data['nav_df'], current_position
        )
        
        # 8. Create a dynamic visualization of methane distribution
        print("Generating dynamic visualization of methane distribution...")
        if 'time' in processed_data.columns:
            # Convert time to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(processed_data['time']):
                processed_data['Time (UTC)'] = pd.to_datetime(processed_data['time'])
            else:
                processed_data['Time (UTC)'] = processed_data['time']
                
            # Create a dynamic animation of methane concentration over time
            animate_methane_distribution(
                processed_data, 
                interval=1000, 
                save_path="methane_concentration_animation.mp4"
            )
        else:
            # If no time column is available, create a static plot
            plot_methane_distribution(
                processed_data, 
                tracer_column='tracer_concentration',
                save_path="methane_concentration_map.png"
            )
    else:
        print("No methane leak detected. Continuing monitoring...")
        plot_methane_distribution(
            processed_data, 
            tracer_column='tracer_concentration',
            save_path="methane_concentration_map.png"
        )
    
    print("Project execution complete.")

if __name__ == "__main__":
    main()
