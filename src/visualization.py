import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
from datetime import datetime, timedelta
import json
import os

def plot_methane_distribution(data: pd.DataFrame, time_filter=None, show=True, save_path=None, tracer_column='tracer_concentration'):
    """
    Plots a scatter map of methane tracer concentration over geographical coordinates.
    
    Parameters:
      data (pd.DataFrame): DataFrame containing at least 'latitude', 'longitude', 'tracer concentration'.
      time_filter (optional): Either a specific timestamp or a tuple (start, end) to filter data on the 'Time (UTC)' column.
      show (bool): Whether to display the plot.
      save_path (str, optional): If provided, the figure will be saved to this file path.
    
    Returns:
      fig, ax: The matplotlib figure and axes objects.
    """
    # If a time_filter is provided and 'Time (UTC)' exists, filter the data accordingly.
    if time_filter and 'Time (UTC)' in data.columns:
        if isinstance(time_filter, tuple) and len(time_filter) == 2:
            data = data[(data['Time (UTC)'] >= time_filter[0]) & (data['Time (UTC)'] <= time_filter[1])]
        else:
            data = data[data['Time (UTC)'] == time_filter]
    
    lat = data['latitude']
    lon = data['longitude']
    tracer = data[tracer_column]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(lon, lat, c=tracer, cmap='viridis', s=50, edgecolor='k')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Tracer Concentration')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Methane Distribution Map')
    
    # Highlight the point with the highest tracer concentration as a potential leak source.
    if not data.empty:
        idx_max = tracer.idxmax()
        max_lat = lat.loc[idx_max]
        max_lon = lon.loc[idx_max]
        ax.plot(max_lon, max_lat, marker='*', markersize=15, color='red', label='Potential Leak Source')
        ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    return fig, ax

def animate_methane_distribution(data: pd.DataFrame, interval=1000, save_path=None):
    """
    Creates an animation showing the evolution of methane tracer concentration over time.
    
    Parameters:
      data (pd.DataFrame): DataFrame containing at least 'Time (UTC)', 'latitude', 'longitude', 'tracer concentration'.
      interval (int): Delay between frames in milliseconds.
      save_path (str, optional): If provided, the animation will be saved to this file (e.g., 'animation.mp4').
                                 Note: For MP4 format, FFmpeg must be installed and available in the system PATH.
    
    Returns:
      ani: The matplotlib.animation.FuncAnimation object.
    """
    # Ensure 'Time (UTC)' is a datetime object.
    if data['Time (UTC)'].dtype == 'O':
        data['Time (UTC)'] = pd.to_datetime(data['Time (UTC)'])
    
    times = np.sort(data['Time (UTC)'].unique())
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)  # Increased resolution for better quality
    
    def update(frame):
        ax.clear()
        current_time = times[frame]
        subset = data[data['Time (UTC)'] == current_time]
        lat = subset['latitude']
        lon = subset['longitude']
        tracer = subset['tracer_concentration']
        scatter = ax.scatter(lon, lat, c=tracer, cmap='viridis', s=50, edgecolor='k')
        ax.set_title(f'Methane Distribution at {current_time}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Tracer Concentration')
        
        # Highlight the point with the maximum tracer concentration.
        if not subset.empty and len(tracer) > 0:
            try:
                idx_max = tracer.idxmax()
                max_lat = lat.loc[idx_max]
                max_lon = lon.loc[idx_max]
                ax.plot(max_lon, max_lat, marker='*', markersize=15, color='red', label='Potential Leak Source')
                ax.legend()
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not highlight maximum point due to error: {e}")
    
    ani = animation.FuncAnimation(fig, update, frames=len(times), interval=interval, repeat=True)
    
    if save_path:
        # Check file extension and use appropriate writer
        file_extension = os.path.splitext(save_path)[1].lower()
        
        if file_extension == '.mp4':
            try:
                # Force matplotlib to use ffmpeg for MP4 output
                # Set up the FFmpeg writer with higher quality settings
                writer = animation.FFMpegWriter(
                    fps=10,  # Higher fps for smoother animation
                    metadata=dict(title='Methane Distribution Animation'),
                    bitrate=5000,  # Higher bitrate for better quality
                    codec='h264',  # Use H.264 codec for compatibility 
                    extra_args=['-pix_fmt', 'yuv420p']  # Ensure compatibility with video players
                )
                
                # Create the output directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                
                print(f"Saving animation as MP4 using FFmpeg writer to {save_path}")
                ani.save(save_path, writer=writer)
                print(f"Animation saved successfully to {save_path}")
            except Exception as e:
                print(f"Error saving MP4 file: {e}")
                # Fall back to GIF format if MP4 fails
                gif_path = save_path.replace('.mp4', '.gif')
                print(f"Attempting to save as GIF instead at: {gif_path}")
                ani.save(gif_path, writer='pillow')
        elif file_extension == '.gif':
            # Use Pillow for GIF format
            print(f"Saving animation as GIF to {save_path}")
            ani.save(save_path, writer='pillow')
        else:
            print(f"Unsupported file format: {file_extension}. Supported formats are .mp4 and .gif.")
    
    return ani

def create_dynamic_methane_map(data: pd.DataFrame, 
                              save_path=None, 
                              map_type='folium',
                              critical_threshold=0.8,
                              uncertainty_column='prediction_uncertainty'):
    """
    Creates an interactive dynamic spatial-temporal visualization of methane data.
    
    Parameters:
      data (pd.DataFrame): DataFrame containing at least 'latitude', 'longitude', 'tracer_concentration', 
                         'Time (UTC)' columns and optionally prediction_uncertainty.
      save_path (str, optional): If provided, the interactive map will be saved to this file.
      map_type (str): Type of map to create ('folium').
      critical_threshold (float): Threshold value (as percentile) to highlight critical methane levels.
      uncertainty_column (str): Column name containing uncertainty values.
    
    Returns:
      map_obj: The created map object (folium.Map)
    """
    # Ensure data has the necessary columns
    required_cols = ['latitude', 'longitude', 'tracer_concentration']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data must contain these columns: {required_cols}")
    
    # Calculate critical threshold based on percentile
    critical_value = data['tracer_concentration'].quantile(critical_threshold)
    
    # Add a column to flag critical points
    data['is_critical'] = data['tracer_concentration'] >= critical_value
    
    # Flag high uncertainty areas if the column exists
    if uncertainty_column in data.columns:
        high_uncertainty_threshold = data[uncertainty_column].quantile(0.8)
        data['high_uncertainty'] = data[uncertainty_column] >= high_uncertainty_threshold
    else:
        data['high_uncertainty'] = False
    
    # Create visualization based on selected map type
    if map_type == 'folium':
        return _create_folium_map(data, save_path)
    else:
        raise ValueError("map_type must be 'folium'")

def _create_folium_map(data, save_path=None):
    """
    Creates a folium-based interactive map with timeseries support if time data is available.
    """
    # Calculate the center point for the map
    center_lat = data['latitude'].mean()
    center_lon = data['longitude'].mean()
    
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Check if time column exists for time-based visualization
    has_time = 'Time (UTC)' in data.columns
    
    if has_time:
        # Convert times to strings for GeoJSON if needed
        if isinstance(data['Time (UTC)'].iloc[0], (datetime, pd.Timestamp)):
            data['time_str'] = data['Time (UTC)'].dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            data['time_str'] = data['Time (UTC)'].astype(str)
            
        # Create features list for TimestampedGeoJson
        features = []
        for _, row in data.iterrows():
            popup_text = f"""
            Concentration: {row['tracer_concentration']:.4f}<br>
            Critical: {'Yes' if row['is_critical'] else 'No'}<br>
            High Uncertainty: {'Yes' if row['high_uncertainty'] else 'No'}
            """
            
            # Determine point color based on criticality
            if row['is_critical'] and row['high_uncertainty']:
                color = "#FF00FF"  # Purple for critical + high uncertainty
            elif row['is_critical']:
                color = "#FF0000"  # Red for critical
            elif row['high_uncertainty']:
                color = "#FFA500"  # Orange for high uncertainty
            else:
                color = "#0000FF"  # Blue for normal
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row['longitude'], row['latitude']]
                },
                'properties': {
                    'time': row['time_str'],
                    'popup': popup_text,
                    'icon': 'circle',
                    'iconstyle': {
                        'fillColor': color,
                        'fillOpacity': 0.8,
                        'stroke': 'true',
                        'radius': 7 + (3 if row['is_critical'] else 0)
                    }
                }
            }
            features.append(feature)
        
        # Add the TimestampedGeoJson
        TimestampedGeoJson(
            {'type': 'FeatureCollection', 'features': features},
            period='PT1H',  # 1 hour period
            duration='PT5M',  # 5 minute duration
            add_last_point=True,
            auto_play=True,
            loop=False
        ).add_to(m)
    else:
        # Create a static heatmap if no time data
        heat_data = [[row['latitude'], row['longitude'], row['tracer_concentration']] 
                     for _, row in data.iterrows()]
        HeatMap(heat_data).add_to(m)
        
        # Also add markers for critical points and high uncertainty
        for _, row in data[data['is_critical'] | data['high_uncertainty']].iterrows():
            popup_text = f"""
            Concentration: {row['tracer_concentration']:.4f}<br>
            Critical: {'Yes' if row['is_critical'] else 'No'}<br>
            High Uncertainty: {'Yes' if row['high_uncertainty'] else 'No'}
            """
            
            # Determine marker color
            if row['is_critical'] and row['high_uncertainty']:
                color = 'purple'
            elif row['is_critical']:
                color = 'red'
            else:
                color = 'orange'
                
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=7,
                popup=popup_text,
                color=color,
                fill=True,
                fill_color=color
            ).add_to(m)
    
    # Add legend
    legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
        padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <p><b>Methane Levels Legend</b></p>
        <p><i class="fa fa-circle" style="color:red;"></i> Critical Concentration</p>
        <p><i class="fa fa-circle" style="color:orange;"></i> High Uncertainty</p>
        <p><i class="fa fa-circle" style="color:purple;"></i> Critical + High Uncertainty</p>
        <p><i class="fa fa-circle" style="color:blue;"></i> Normal Levels</p>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    if save_path:
        m.save(save_path)
    
    return m

if __name__ == "__main__":
    print("Visualization module - this module provides functions for visualizing methane concentration data")
    print("Import and use this module in other scripts rather than running it directly.")
