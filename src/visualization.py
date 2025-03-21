import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
import plotly.express as px
import plotly.graph_objects as go
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
      map_type (str): Type of map to create ('folium', 'plotly', or 'plotly-dashboard').
      critical_threshold (float): Threshold value (as percentile) to highlight critical methane levels.
      uncertainty_column (str): Column name containing uncertainty values.
    
    Returns:
      map_obj: The created map object (folium.Map or plotly Figure)
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
    elif map_type == 'plotly':
        return _create_plotly_map(data, save_path)
    elif map_type == 'plotly-dashboard':
        return _create_plotly_dashboard(data, save_path)
    else:
        raise ValueError("map_type must be one of: 'folium', 'plotly', 'plotly-dashboard'")

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

def _create_plotly_map(data, save_path=None):
    """
    Creates a plotly-based static or animated map based on the data.
    """
    has_time = 'Time (UTC)' in data.columns
    
    # Create a copy of the dataset for visualization
    vis_data = data.copy()
    
    # Create a color map for the critical and uncertainty markers
    vis_data['marker_color'] = 'blue'  # Default
    vis_data.loc[vis_data['high_uncertainty'], 'marker_color'] = 'orange'
    vis_data.loc[vis_data['is_critical'], 'marker_color'] = 'red'
    vis_data.loc[(vis_data['is_critical'] & vis_data['high_uncertainty']), 'marker_color'] = 'purple'
    
    # Adjust marker size based on concentration
    min_size = 5
    max_size = 15
    vis_data['marker_size'] = min_size + (max_size - min_size) * (
        (vis_data['tracer_concentration'] - vis_data['tracer_concentration'].min()) / 
        (vis_data['tracer_concentration'].max() - vis_data['tracer_concentration'].min())
    )
    
    if has_time:
        # For animated visualization with time data
        fig = px.scatter_mapbox(
            vis_data, 
            lat="latitude", 
            lon="longitude", 
            color="tracer_concentration",
            size="marker_size",
            color_continuous_scale="Viridis",
            animation_frame="Time (UTC)" if has_time else None,
            mapbox_style="open-street-map",
            hover_data=["tracer_concentration", "is_critical", "high_uncertainty"],
            zoom=10,
            title="Dynamic Methane Concentration Map"
        )
    else:
        # Static map without animation
        fig = px.scatter_mapbox(
            vis_data, 
            lat="latitude", 
            lon="longitude", 
            color="tracer_concentration",
            size="marker_size",
            color_continuous_scale="Viridis",
            mapbox_style="open-street-map",
            hover_data=["tracer_concentration", "is_critical", "high_uncertainty"],
            zoom=10,
            title="Methane Concentration Map"
        )
    
    # Highlight critical points and uncertainty with custom markers
    for label, color in [
        ("Critical + High Uncertainty", "purple"),
        ("Critical", "red"),
        ("High Uncertainty", "orange")
    ]:
        subset = vis_data[vis_data['marker_color'] == color]
        if not subset.empty:
            fig.add_trace(go.Scattermapbox(
                lat=subset['latitude'],
                lon=subset['longitude'],
                mode='markers',
                marker=dict(
                    size=subset['marker_size'] + 2,
                    color=color,
                    opacity=0.7
                ),
                name=label,
                hoverinfo='none'
            ))
    
    # Improve layout
    fig.update_layout(
        mapbox=dict(center=dict(lat=vis_data['latitude'].mean(), lon=vis_data['longitude'].mean())),
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig

def _create_plotly_dashboard(data, save_path=None):
    """
    Creates a more advanced plotly dashboard that can be deployed as a web app.
    Requires: dash and dash-leaflet
    
    This function returns instructions on how to run the dashboard
    instead of the dashboard object itself.
    """
    # Generate a Python file for the dashboard
    dashboard_code = f'''
# Save this to a file named methane_dashboard.py and run with "python methane_dashboard.py"
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import json
import os

# Load the data
data_json = {data.to_json(orient='records', date_format='iso')}
df = pd.read_json(json.dumps(data_json))

# Convert time column back to datetime if it exists
if 'Time (UTC)' in df.columns:
    df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'])
    all_timestamps = sorted(df['Time (UTC)'].unique())
    time_marks = {{int(i): ts.strftime('%H:%M:%S') 
                  for i, ts in enumerate(all_timestamps)}}
else:
    all_timestamps = [None]
    time_marks = {{0: 'N/A'}}

# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dynamic Methane Leak Detection Dashboard"),
    
    html.Div([
        html.Div([
            html.H3("Time Control"),
            dcc.Slider(
                id='time-slider',
                min=0,
                max=len(all_timestamps) - 1,
                value=0,
                marks=time_marks,
                step=None,
            ),
        ], style={{'width': '100%', 'padding': '10px'}}),
        
        html.Div([
            html.H3("Methane Concentration Map"),
            dcc.Graph(id='methane-map'),
        ], style={{'width': '70%', 'display': 'inline-block', 'padding': '10px'}}),
        
        html.Div([
            html.H3("Statistics"),
            html.Div(id='stats-display'),
            html.H4("Critical Points"),
            html.Div(id='critical-points-table'),
        ], style={{'width': '25%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}}),
    ]),
])

@app.callback(
    [Output('methane-map', 'figure'),
     Output('stats-display', 'children'),
     Output('critical-points-table', 'children')],
    [Input('time-slider', 'value')]
)
def update_map(time_idx):
    if 'Time (UTC)' in df.columns:
        selected_time = all_timestamps[time_idx]
        filtered_df = df[df['Time (UTC)'] == selected_time]
    else:
        filtered_df = df
    
    # Create the map
    fig = px.scatter_mapbox(
        filtered_df, 
        lat="latitude", 
        lon="longitude", 
        color="tracer_concentration",
        size=filtered_df['tracer_concentration'] * 50,  # Scale appropriately
        color_continuous_scale="Viridis",
        hover_data=["tracer_concentration", "is_critical", "high_uncertainty"],
        zoom=10,
        mapbox_style="open-street-map"
    )
    
    # Add markers for critical and high uncertainty points
    for label, condition, color in [
        ("Critical + High Uncertainty", 
         (filtered_df['is_critical'] & filtered_df['high_uncertainty']), "purple"),
        ("Critical", 
         (filtered_df['is_critical'] & ~filtered_df['high_uncertainty']), "red"),
        ("High Uncertainty", 
         (~filtered_df['is_critical'] & filtered_df['high_uncertainty']), "orange"),
    ]:
        subset = filtered_df[condition]
        if not subset.empty:
            fig.add_trace({
                'type': 'scattermapbox',
                'lat': subset['latitude'],
                'lon': subset['longitude'],
                'mode': 'markers',
                'marker': {
                    'size': subset['tracer_concentration'] * 50 + 5,
                    'color': color,
                    'opacity': 0.7
                },
                'name': label,
                'showlegend': True
            })
    
    # Update layout
    fig.update_layout(
        mapbox=dict(center=dict(lat=filtered_df['latitude'].mean(), 
                                 lon=filtered_df['longitude'].mean())),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    # Calculate statistics
    total_points = len(filtered_df)
    critical_points = sum(filtered_df['is_critical'])
    high_uncertainty_points = sum(filtered_df['high_uncertainty'])
    
    # Create statistics display
    stats = html.Div([
        html.P(f"Total Data Points: {total_points}"),
        html.P(f"Critical Points: {critical_points} ({critical_points/total_points*100:.1f}%)"),
        html.P(f"High Uncertainty: {high_uncertainty_points} ({high_uncertainty_points/total_points*100:.1f}%)"),
        html.P(f"Max Concentration: {filtered_df['tracer_concentration'].max():.4f}"),
        html.P(f"Mean Concentration: {filtered_df['tracer_concentration'].mean():.4f}")
    ])
    
    # Create critical points table
    critical_df = filtered_df[filtered_df['is_critical']]
    if len(critical_df) > 0:
        critical_table = html.Table([
            html.Thead(html.Tr([html.Th('Lat'), html.Th('Lon'), html.Th('Conc.'), html.Th('Uncert')])),
            html.Tbody([
                html.Tr([
                    html.Td(f"{row['latitude']:.4f}"),
                    html.Td(f"{row['longitude']:.4f}"),
                    html.Td(f"{row['tracer_concentration']:.4f}"),
                    html.Td(f"{row['high_uncertainty']}")
                ]) for _, row in critical_df.head(10).iterrows()
            ])
        ])
    else:
        critical_table = html.P("No critical points found")
    
    return fig, stats, critical_table

if __name__ == '__main__':
    print("Starting Methane Leak Detection Dashboard")
    print("Access the dashboard at http://127.0.0.1:8050")
    app.run_server(debug=True)
'''
    
    # Save the dashboard code to a file if requested
    if save_path:
        dashboard_file = os.path.join(os.path.dirname(save_path), 'methane_dashboard.py')
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_code)
        
        return f"Dashboard code saved to {dashboard_file}. Run with 'python {dashboard_file}'"
    
    return dashboard_code

if __name__ == "__main__":
    print("Visualization module - this module provides functions for visualizing methane concentration data")
    print("Import and use this module in other scripts rather than running it directly.")
