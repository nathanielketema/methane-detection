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
        regression_model = QuantumGaussianRegression()
        regression_model.fit(prepared_data['X'], prepared_data['y'], force_retrain=force_retrain)
        
        # Check if the model was properly fitted
        if regression_model.model is None:
            print("Error: Regression model training failed. Forcing a retrain...")
            regression_model.fit(prepared_data['X'], prepared_data['y'], force_retrain=True)
            
            # Double-check if model is still None after forced retraining
            if regression_model.model is None:
                print("Critical error: Regression model training still failed after forced retrain.")
                print("Project execution incomplete. Please check the code and training data.")
                return

        # 7. Predict methane concentration for all points
        methane_concentrations, concentration_uncertainties = regression_model.predict(prepared_data['X'])
        
        # Create a DataFrame with original indices for mapping predictions back to processed data
        prediction_indices = features.index[:len(methane_concentrations)]
        
        # Add predictions to the original data for visualization
        processed_data['predicted_concentration'] = pd.Series(methane_concentrations, index=prediction_indices)
        processed_data['prediction_uncertainty'] = pd.Series(concentration_uncertainties, index=prediction_indices)
        
        # 8. For drone navigation simulation, we'll still use a small subset of data
        print("Simulating drone navigation...")
        current_position = (processed_data['latitude'].iloc[0], processed_data['longitude'].iloc[0])
        next_waypoint, prediction, uncertainty = simulate_drone_navigation(
            regression_model, prepared_data['nav_df'], current_position
        )
        
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
            
            # Create Plotly-based interactive map
            print("Generating Plotly interactive map...")
            plotly_map = create_dynamic_methane_map(
                processed_data,
                save_path="outputs/methane_map_plotly.html",
                map_type="plotly",
                critical_threshold=0.8,
                uncertainty_column='prediction_uncertainty'
            )
            
            # Generate interactive dashboard code
            print("Generating interactive dashboard code...")
            dashboard_instructions = create_dynamic_methane_map(
                processed_data,
                save_path="outputs/methane_dashboard.py",
                map_type="plotly-dashboard",
                critical_threshold=0.8,
                uncertainty_column='prediction_uncertainty'
            )
            print(dashboard_instructions)
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Run the main function with parsed arguments
    main(force_retrain=args.force_retrain)
