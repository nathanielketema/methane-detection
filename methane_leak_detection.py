import pandas as pd
from src.config import RAW_DATA_PATH

from src.data_processing import process_data
from src.feature_engineering import extract_features
from src.data_preparation import prepare_training_data
from src.models.methane_leak_classifier import MethaneLeakClassifier
from src.models.quantum_gaussian_regression import QuantumGaussianRegression
from src.drone_navigation import simulate_drone_navigation
from src.visualization import plot_methane_distribution, animate_methane_distribution, create_dynamic_methane_map
import argparse
import os
import numpy as np
import traceback

def main(force_retrain=False):
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
    print("Detecting potential methane leaks using all available data...")

    leak_predictions = classifier.predict(prepared_data['X'])
    
    leak_detected = any(leak_predictions == 1)
    print(f"Leak detection result: {'Leak detected!' if leak_detected else 'No leak detected.'}")
    print(f"Number of potential leak points detected: {sum(leak_predictions == 1)} out of {len(leak_predictions)}")
    
    if leak_detected:
        # 6. If leak is detected, train the regression model to predict methane concentration
        print("Leak detected! Training the regression model to predict methane concentration...")
        
        # Initialize regression model
        regression_model = QuantumGaussianRegression()
        
        try:
            regression_model.fit(prepared_data['X'], prepared_data['y'], force_retrain=force_retrain)
            
            if regression_model.model is None:
                raise ValueError("Model training resulted in a null model")
                
            print("Regression model training successful!")
            
        except Exception as e:
            print("Error during model training:")
            traceback.print_exc()
            raise RuntimeError("Failed to train regression model")
        
        # 7. Predict methane concentration for all points
        try:
            methane_concentrations, concentration_uncertainties = regression_model.predict(prepared_data['X'])
            print("Successfully predicted methane concentrations using trained model.")
        except Exception as pred_error:
            raise RuntimeError(f"Failed to generate predictions: {pred_error}")
        
        # Create a DataFrame with original indices for mapping predictions back to processed data
        prediction_indices = features.index[:len(methane_concentrations)]
        
        # Add predictions to the original data for visualization
        processed_data['predicted_concentration'] = pd.Series(methane_concentrations, index=prediction_indices)
        processed_data['prediction_uncertainty'] = pd.Series(concentration_uncertainties, index=prediction_indices)
        
        # 8. Simulate drone navigation
        print("Simulating drone navigation...")
        current_position = (processed_data['latitude'].iloc[0], processed_data['longitude'].iloc[0])
        
        try:
            next_waypoint, prediction, uncertainty = simulate_drone_navigation(
                regression_model, prepared_data['nav_df'], current_position
            )
            print(f"Next waypoint: {next_waypoint}, Expected concentration: {prediction}")
        except Exception as nav_error:
            raise RuntimeError(f"Failed to simulate drone navigation: {nav_error}")
        
        # 9. Create visualizations
        print("Generating visualization of methane distribution...")
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
                save_path="outputs/methane_concentration_animation.mp4"
            )
            
            # Create interactive visualizations
            print("Creating dynamic methane mapping visualizations...")
            print("Generating Folium interactive map...")
            folium_map = create_dynamic_methane_map(
                processed_data,
                save_path="outputs/methane_map_folium.html",
                map_type="folium",
                critical_threshold=0.8,
                uncertainty_column='prediction_uncertainty'
            )
        else:
            # If no time column is available, create a static plot
            plot_methane_distribution(
                processed_data, 
                tracer_column='tracer_concentration',
                save_path="outputs/methane_concentration_map.png"
            )
            
            # Create a static interactive map
            print("Creating static interactive methane map...")
            folium_map = create_dynamic_methane_map(
                processed_data,
                save_path="outputs/methane_map_static.html",
                map_type="folium",
                critical_threshold=0.8,
                uncertainty_column='prediction_uncertainty'
            )
    else:
        print("No methane leak detected. Continuing monitoring...")
        plot_methane_distribution(
            processed_data, 
            tracer_column='tracer_concentration',
            save_path="outputs/methane_concentration_map.png"
        )
    
    print("Project execution complete.")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Methane Leak Detection Project')
    parser.add_argument('--force-retrain', action='store_true', 
                        help='Force retraining of models even if saved models exist')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/training_data", exist_ok=True)
    
    # Run the main function with parsed arguments
    main(force_retrain=args.force_retrain)
