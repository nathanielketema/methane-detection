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
import time

def main(force_retrain=False, max_retries=3):
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
        
        # Initialize variables for regression model and predictions
        regression_model = None
        methane_concentrations = None
        concentration_uncertainties = None
        
        # Try to fit the model with retries
        for attempt in range(1, max_retries + 1):
            try:
                # If this is a retry, force a retrain
                retry_force_retrain = force_retrain or (attempt > 1)
                
                if attempt > 1:
                    print(f"Attempt {attempt}/{max_retries} to train regression model...")
                    # Add a short delay between retries to avoid race conditions
                    time.sleep(2)
                
                # Create a new model instance for each attempt
                regression_model = QuantumGaussianRegression()
                regression_model.fit(prepared_data['X'], prepared_data['y'], force_retrain=retry_force_retrain)
                
                # Check if the model was properly fitted
                if regression_model.model is not None:
                    print("Regression model training successful!")
                    break
                else:
                    print(f"Warning: Model training attempt {attempt} resulted in a null model.")
                    if attempt < max_retries:
                        continue
                    else:
                        print("Maximum retry attempts reached. Will try to fall back to a simple model.")
            except Exception as e:
                print(f"Error during model training (attempt {attempt}/{max_retries}): {e}")
                traceback.print_exc()
                if attempt < max_retries:
                    continue
                else:
                    print("Maximum retry attempts reached. Will try to fall back to a simple model.")
        
        # If regression model training failed after all retries, create a fallback model
        if regression_model is None or regression_model.model is None:
            print("Critical warning: Unable to train quantum regression model after multiple attempts.")
            print("Creating a fallback model based on average concentration...")
            
            # Create fallback predictions based on simple mean of training data
            avg_concentration = np.mean(prepared_data['y'])
            std_concentration = np.std(prepared_data['y'])
            methane_concentrations = np.full(len(prepared_data['X']), avg_concentration)
            concentration_uncertainties = np.full(len(prepared_data['X']), std_concentration)
            
            print(f"Using fallback constant prediction: {avg_concentration:.4f} ± {std_concentration:.4f}")
        else:
            # 7. Predict methane concentration for all points
            try:
                methane_concentrations, concentration_uncertainties = regression_model.predict(prepared_data['X'])
                print("Successfully predicted methane concentrations using trained model.")
            except Exception as pred_error:
                print(f"Error during prediction: {pred_error}")
                # Final fallback if prediction fails
                avg_concentration = np.mean(prepared_data['y'])
                std_concentration = np.std(prepared_data['y'])
                methane_concentrations = np.full(len(prepared_data['X']), avg_concentration)
                concentration_uncertainties = np.full(len(prepared_data['X']), std_concentration)
                print(f"Using fallback constant prediction: {avg_concentration:.4f} ± {std_concentration:.4f}")
        
        # Create a DataFrame with original indices for mapping predictions back to processed data
        prediction_indices = features.index[:len(methane_concentrations)]
        
        # Add predictions to the original data for visualization
        processed_data['predicted_concentration'] = pd.Series(methane_concentrations, index=prediction_indices)
        processed_data['prediction_uncertainty'] = pd.Series(concentration_uncertainties, index=prediction_indices)
        
        # 8. For drone navigation simulation, we'll use the model if available
        print("Simulating drone navigation...")
        current_position = (processed_data['latitude'].iloc[0], processed_data['longitude'].iloc[0])
        
        try:
            if regression_model is not None and regression_model.model is not None:
                next_waypoint, prediction, uncertainty = simulate_drone_navigation(
                    regression_model, prepared_data['nav_df'], current_position
                )
            else:
                print("Warning: Using simplified navigation as regression model is unavailable.")
                # Use closest point with highest concentration as next waypoint
                idx = processed_data['tracer_concentration'].idxmax()
                next_waypoint = (processed_data['latitude'].loc[idx], processed_data['longitude'].loc[idx])
                prediction = processed_data['tracer_concentration'].loc[idx]
                uncertainty = 0.0
                print(f"Next waypoint: {next_waypoint}, Expected concentration: {prediction}")
        except Exception as nav_error:
            print(f"Error during navigation simulation: {nav_error}")
            print("Continuing with visualization...")
        
        # 9. Create both static and dynamic visualizations of methane distribution
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
            
            # Step 9: Dynamic Methane Mapping - Create interactive visualizations
            print("Creating dynamic methane mapping visualizations...")
            
            # Create Folium-based interactive map
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
            
            # Also create a static interactive map
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
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retries for model training')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/training_data", exist_ok=True)
    
    # Run the main function with parsed arguments
    main(force_retrain=args.force_retrain, max_retries=args.max_retries)
