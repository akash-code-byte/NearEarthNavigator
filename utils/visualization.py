import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_asteroids(df):
    """
    Create a scatter plot of asteroids showing miss distance vs. relative velocity
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with asteroid data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create scatter plot
    fig = px.scatter(
        df,
        x='miss_distance',
        y='relative_velocity',
        size='estimated_diameter_max',
        color='is_potentially_hazardous',
        hover_name='name',
        hover_data={
            'close_approach_date_display': True,
            'estimated_diameter_max': ':.3f',
            'miss_distance': ':.0f',
            'relative_velocity': ':.2f',
            'is_potentially_hazardous': False
        },
        size_max=50,
        color_discrete_map={True: '#F44336', False: '#2196F3'},
        labels={
            'miss_distance': 'Miss Distance (km)',
            'relative_velocity': 'Relative Velocity (km/s)',
            'estimated_diameter_max': 'Diameter (km)',
            'is_potentially_hazardous': 'Potentially Hazardous'
        },
        title='Asteroid Distribution: Miss Distance vs. Relative Velocity'
    )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis=dict(type='log'),  # Log scale for miss distance
        xaxis_title='Miss Distance (km) - Log Scale',
        yaxis_title='Relative Velocity (km/s)',
        legend_title='Potentially Hazardous',
        template='plotly_white',
        height=600
    )
    
    # Add Earth reference as a vertical line (average Earth diameter: ~12,742 km)
    earth_diameter = 12742
    fig.add_vline(x=earth_diameter, line_width=1, line_dash="dash", line_color="#4CAF50",
                 annotation_text="Earth diameter", annotation_position="top right")
    
    # Add moon orbit reference as a vertical line (average moon distance: ~384,400 km)
    moon_distance = 384400
    fig.add_vline(x=moon_distance, line_width=1, line_dash="dash", line_color="#9C27B0",
                 annotation_text="Moon orbit", annotation_position="top right")
    
    return fig

def visualize_top_3_hazardous_asteroids(df):
    """
    Create a visual representation of the top 3 potentially hazardous asteroids
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with asteroid data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Filter potentially hazardous asteroids
    hazardous_df = df[df['is_potentially_hazardous'] == True].copy()
    
    # If no hazardous asteroids, return an empty figure with a message
    if len(hazardous_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No potentially hazardous asteroids found in the dataset",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=400)
        return fig
    
    # Sort by a danger score (using energy_proxy and proximity_risk)
    hazardous_df['danger_score'] = hazardous_df['energy_proxy'] * hazardous_df['proximity_risk']
    top_hazardous = hazardous_df.sort_values('danger_score', ascending=False).head(3)
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"{name}" for name in top_hazardous['name']],
        specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]]
    )
    
    # Colors for the indicators
    colors = ['#F44336', '#FF9800', '#FFEB3B']
    
    # Add indicators for each asteroid
    for i, (_, row) in enumerate(top_hazardous.iterrows()):
        # Calculate the risk level (0-100%)
        risk_level = min(100, row['danger_score'] / hazardous_df['danger_score'].max() * 100)
        
        # Create asteroid representation
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=risk_level,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={
                    'text': f"<b>{row['name']}</b><br><span style='font-size:0.8em;'>Approach: {row['close_approach_date_display']}</span>",
                    'font': {'size': 12}
                },
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': colors[i]},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 33], 'color': 'rgba(0, 200, 0, 0.1)'},
                        {'range': [33, 66], 'color': 'rgba(255, 180, 0, 0.1)'},
                        {'range': [66, 100], 'color': 'rgba(255, 0, 0, 0.1)'},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                },
                number={
                    'suffix': '%',
                    'font': {'size': 20}
                },
                delta={
                    'reference': 50,
                    'increasing': {'color': "red"},
                    'decreasing': {'color': "green"}
                }
            ),
            row=1, col=i+1
        )
        
        # Add additional info
        fig.add_annotation(
            text=f"Diameter: {row['estimated_diameter_max']*1000:.1f} m<br>"
                 f"Velocity: {row['relative_velocity']:.1f} km/s<br>"
                 f"Miss Distance: {row['miss_distance']/1000000:.2f} million km",
            xref=f"x{i+1 if i > 0 else ''}", yref=f"y{i+1 if i > 0 else ''}",
            x=0.5, y=0.15,
            showarrow=False,
            font=dict(size=10),
            align="center",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        margin=dict(t=100, b=100, l=40, r=40),
        title={
            'text': 'Top 3 Potentially Hazardous Asteroids',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    return fig

def plot_feature_importance(feature_importance):
    """
    Create a bar chart showing feature importance
    
    Parameters:
    -----------
    feature_importance : dict
        Dictionary mapping feature names to importance values
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Convert dictionary to dataframe
    df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        df,
        x='Feature',
        y='Importance',
        color='Importance',
        color_continuous_scale='viridis',
        title='Feature Importance'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Feature',
        yaxis_title='Importance',
        xaxis={'categoryorder': 'total descending'},
        height=500
    )
    
    return fig

def plot_correlation_heatmap(df):
    """
    Create a correlation heatmap for numerical columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with asteroid data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Feature Correlation Matrix'
    )
    
    # Add correlation values as text
    for i, row in enumerate(corr_matrix.values):
        for j, val in enumerate(row):
            fig.add_annotation(
                x=j, y=i,
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(
                    color='white' if abs(val) > 0.5 else 'black',
                    size=8
                )
            )
    
    # Update layout
    fig.update_layout(
        height=700,
        width=700
    )
    
    return fig

def plot_feature_distributions(df):
    """
    Create a grid of histograms showing distributions of key features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with asteroid data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Select relevant features
    features = [
        'absolute_magnitude', 'estimated_diameter_max', 'relative_velocity', 
        'miss_distance', 'diameter_velocity_ratio', 'energy_proxy'
    ]
    
    # Create subplot grid
    fig = make_subplots(
        rows=2, 
        cols=3,
        subplot_titles=[f.replace('_', ' ').title() for f in features]
    )
    
    # Add histograms for each feature
    row, col = 1, 1
    for i, feature in enumerate(features):
        # Calculate row and column position
        row = i // 3 + 1
        col = i % 3 + 1
        
        # Create traces for hazardous and non-hazardous asteroids
        for is_hazardous, color, name in [(True, '#F44336', 'Hazardous'), (False, '#2196F3', 'Non-Hazardous')]:
            # Get subset of data
            subset = df[df['is_potentially_hazardous'] == is_hazardous]
            
            # Skip if no data
            if len(subset) == 0:
                continue
            
            # Transform data if needed for better visualization
            if feature in ['miss_distance', 'energy_proxy']:
                values = np.log10(subset[feature] + 1)  # log scale for wide-range values
                feature_name = f"Log10({feature})"
            else:
                values = subset[feature]
                feature_name = feature
            
            # Add histogram trace
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=name,
                    marker_color=color,
                    opacity=0.6,
                    showlegend=True if i == 0 else False  # Only show legend for first plot
                ),
                row=row, col=col
            )
    
    # Update layout for better readability
    fig.update_layout(
        height=600,
        barmode='overlay',
        title={
            'text': 'Feature Distributions by Hazard Classification',
            'y': 0.98
        }
    )
    
    # Update x-axis titles
    for i, feature in enumerate(features):
        row = i // 3 + 1
        col = i % 3 + 1
        
        if feature in ['miss_distance', 'energy_proxy']:
            feature_name = f"Log10({feature})"
        else:
            feature_name = feature
            
        fig.update_xaxes(title_text=feature_name.replace('_', ' ').title(), row=row, col=col)
    
    return fig

def plot_roc_curve(fpr, tpr, auc):
    """
    Plot ROC curve for model evaluation
    
    Parameters:
    -----------
    fpr : array
        False positive rates
    tpr : array
        True positive rates
    auc : float
        Area under the ROC curve
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc:.4f})',
            line=dict(color='#F44336', width=2)
        )
    )
    
    # Add diagonal reference line (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='#888888', width=2, dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=500,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig

def plot_precision_recall_curve(precision, recall, avg_precision):
    """
    Plot precision-recall curve for model evaluation
    
    Parameters:
    -----------
    precision : array
        Precision values
    recall : array
        Recall values
    avg_precision : float
        Average precision score
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add precision-recall curve
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR Curve (AP = {avg_precision:.4f})',
            line=dict(color='#2196F3', width=2)
        )
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=500,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig

def create_3d_asteroid_visualization(df, map_type='3d'):
    if map_type == 'global':
        # Generate random impact locations
        num_impacts = min(len(df), 20)  # Limit for visualization
        np.random.seed(42)  # For reproducibility
        
        # Create impact coordinates
        lats = np.random.uniform(-80, 80, num_impacts)
        lons = np.random.uniform(-180, 180, num_impacts)
        
        # Create impact energy and size values
        energies = df['energy_proxy'].head(num_impacts).values
        sizes = df['estimated_diameter_max'].head(num_impacts).values * 1000  # Convert to meters
        
        # Create hover text
        hover_texts = []
        for i in range(num_impacts):
            text = (f"Impact Energy: {energies[i]:.2f} MT<br>"
                   f"Size: {sizes[i]:.1f} m<br>"
                   f"Location: {lats[i]:.2f}°N, {lons[i]:.2f}°E")
            hover_texts.append(text)
        
        # Create the map
        fig = go.Figure(data=go.Scattergeo(
            lon=lons,
            lat=lats,
            mode='markers',
            marker=dict(
                size=sizes/10,
                color=energies,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Impact Energy (MT)'),
                sizemode='area',
            ),
            text=hover_texts,
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='Predicted NEO Impact Locations',
            geo=dict(
                showland=True,
                showcountries=True,
                showocean=True,
                countrywidth=0.5,
                landcolor='rgb(243, 243, 243)',
                oceancolor='rgb(204, 229, 255)',
                projection_type='equirectangular'
            ),
            height=600
        )
        
        return fig
    
    # Original 3D visualization code
    """
    Create a 3D visualization of asteroids
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with asteroid data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create a 3D scatter plot
    fig = go.Figure()
    
    # Generate 3D points from asteroid data
    # We'll create a simplified model where:
    # - x is based on miss_distance
    # - y is based on relative_velocity
    # - z is based on estimated_diameter
    
    # Normalize values for better visualization
    max_miss = df['miss_distance'].max()
    max_vel = df['relative_velocity'].max()
    
    # Create a scale factor to adjust distance values
    # We want to show the Earth at the center and asteroids around it
    scale_factor = 50
    
    # Earth coordinates (origin)
    earth_x, earth_y, earth_z = 0, 0, 0
    
    # Add Earth
    fig.add_trace(
        go.Scatter3d(
            x=[earth_x],
            y=[earth_y],
            z=[earth_z],
            mode='markers',
            marker=dict(
                size=15,
                color='blue',
                symbol='circle'
            ),
            name='Earth',
            text=['Earth'],
            hoverinfo='text'
        )
    )
    
    # Generate angles for asteroid positions
    thetas = np.linspace(0, 2*np.pi, len(df))
    phis = np.linspace(0, np.pi, len(df))
    
    # Calculate 3D positions for asteroids
    x = []
    y = []
    z = []
    sizes = []
    colors = []
    texts = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        # Calculate radius based on miss distance
        radius = np.log10(row['miss_distance'] + 1) / np.log10(max_miss + 1) * scale_factor
        
        # Add some randomness to distribution
        theta = thetas[i] + np.random.normal(0, 0.2)
        phi = phis[i] + np.random.normal(0, 0.2)
        
        # Convert to cartesian coordinates
        xi = radius * np.sin(phi) * np.cos(theta)
        yi = radius * np.sin(phi) * np.sin(theta)
        zi = radius * np.cos(phi)
        
        # Add to lists
        x.append(xi)
        y.append(yi)
        z.append(zi)
        
        # Size based on diameter
        size = 5 + (row['estimated_diameter_max'] / df['estimated_diameter_max'].max() * 20)
        sizes.append(size)
        
        # Color based on hazard status
        color = '#F44336' if row['is_potentially_hazardous'] else '#2196F3'
        colors.append(color)
        
        # Hover text
        text = (
            f"Name: {row['name']}<br>"
            f"Date: {row['close_approach_date_display']}<br>"
            f"Diameter: {row['estimated_diameter_max']*1000:.1f} m<br>"
            f"Velocity: {row['relative_velocity']:.1f} km/s<br>"
            f"Miss Distance: {row['miss_distance']/1000000:.2f} million km"
        )
        texts.append(text)
    
    # Add asteroids
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.8,
                symbol='circle'
            ),
            name='Asteroids',
            text=texts,
            hoverinfo='text'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='3D Visualization of Asteroid Positions',
        scene=dict(
            xaxis_title='X Distance',
            yaxis_title='Y Distance',
            zaxis_title='Z Distance',
            aspectmode='cube'
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def create_asteroid_trajectory_animation(asteroid_data):
    """
    Create an animation of asteroid trajectory
    
    Parameters:
    -----------
    asteroid_data : pandas.DataFrame
        DataFrame with data for a single asteroid
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with animation
    """
    # Get asteroid data
    asteroid = asteroid_data.iloc[0]
    
    # Create figure
    fig = go.Figure()
    
    # Number of frames for animation
    n_frames = 100
    
    # Earth radius in km
    earth_radius = 6371
    
    # Create trajectory points
    # Simplified model: straight line from a distant point to minimum distance then away
    
    # Miss distance
    miss_distance = asteroid['miss_distance']
    
    # Start from 3x the miss distance away
    start_distance = 3 * miss_distance
    
    # Generate trajectory points
    trajectory_distances = np.linspace(start_distance, miss_distance, n_frames//2)
    trajectory_distances = np.concatenate([trajectory_distances, np.linspace(miss_distance, start_distance, n_frames//2)])
    
    # Add Earth
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            marker=dict(
                size=30,
                color='blue',
                symbol='circle'
            ),
            name='Earth'
        )
    )
    
    # Add asteroid trajectory
    asteroid_x = []
    asteroid_y = []
    
    # Generate trajectory with some curvature
    angle = np.pi / 6  # Angle of approach
    for d in trajectory_distances:
        # Add some curvature to trajectory
        if d > miss_distance:
            curve_factor = 0.2 * (d - miss_distance) / (start_distance - miss_distance)
            y_offset = curve_factor * miss_distance * np.sin(angle)
        else:
            y_offset = 0
            
        asteroid_x.append(d * np.cos(angle))
        asteroid_y.append(d * np.sin(angle) + y_offset)
    
    # Show the whole trajectory as a line
    fig.add_trace(
        go.Scatter(
            x=asteroid_x,
            y=asteroid_y,
            mode='lines',
            line=dict(
                color='gray',
                width=1,
                dash='dash'
            ),
            name='Trajectory'
        )
    )
    
    # Add asteroid at starting position
    fig.add_trace(
        go.Scatter(
            x=[asteroid_x[0]],
            y=[asteroid_y[0]],
            mode='markers',
            marker=dict(
                size=15,
                color='red' if asteroid['is_potentially_hazardous'] else 'orange',
                symbol='diamond'
            ),
            name=asteroid['name']
        )
    )
    
    # Create frames for animation
    frames = []
    for i in range(n_frames):
        frames.append(
            go.Frame(
                data=[
                    # Earth remains static
                    go.Scatter(
                        x=[0],
                        y=[0],
                        mode='markers',
                        marker=dict(
                            size=30,
                            color='blue',
                            symbol='circle'
                        ),
                        name='Earth'
                    ),
                    # Trajectory line remains static
                    go.Scatter(
                        x=asteroid_x,
                        y=asteroid_y,
                        mode='lines',
                        line=dict(
                            color='gray',
                            width=1,
                            dash='dash'
                        ),
                        name='Trajectory'
                    ),
                    # Asteroid moves
                    go.Scatter(
                        x=[asteroid_x[i]],
                        y=[asteroid_y[i]],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='red' if asteroid['is_potentially_hazardous'] else 'orange',
                            symbol='diamond'
                        ),
                        name=asteroid['name']
                    )
                ],
                name=f'frame{i}'
            )
        )
    
    # Add frames to figure
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        title=f"Trajectory Animation for {asteroid['name']}",
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 12},
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f'frame{i}'],
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300}
                            }
                        ],
                        "label": str(i),
                        "method": "animate"
                    }
                    for i in range(0, n_frames, 10)
                ]
            }
        ],
        height=600,
        xaxis=dict(
            title='X Distance (km)',
            range=[min(asteroid_x) * 1.1, max(asteroid_x) * 1.1],
            autorange=False
        ),
        yaxis=dict(
            title='Y Distance (km)',
            range=[min(asteroid_y) * 1.1, max(asteroid_y) * 1.1],
            autorange=False,
            scaleanchor="x",  # Force equal scaling
            scaleratio=1      # Force equal scaling
        )
    )
    
    # Add annotations
    fig.add_annotation(
        x=0, y=0,
        text="Earth",
        showarrow=True,
        arrowhead=1,
        ax=20, ay=20
    )
    
    fig.add_annotation(
        x=miss_distance * np.cos(angle),
        y=miss_distance * np.sin(angle),
        text=f"Closest Approach: {miss_distance/1000000:.2f} million km",
        showarrow=True,
        arrowhead=1,
        ax=20, ay=-30
    )
    
    return fig

def create_interactive_asteroid_paths(asteroid_data):
    """
    Create an interactive visualization of asteroid paths relative to Earth
    
    Parameters:
    -----------
    asteroid_data : pandas.DataFrame
        DataFrame with data for one or more asteroids
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Earth radius in millions of km
    earth_radius = 6371 / 1000000  # Convert from km to million km
    
    # Drawing the Earth
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            marker=dict(
                size=20,
                color='blue',
                symbol='circle'
            ),
            name='Earth'
        )
    )
    
    # Draw Earth orbit (approximation)
    theta = np.linspace(0, 2*np.pi, 100)
    earth_orbit_x = np.cos(theta) * 1  # 1 AU
    earth_orbit_y = np.sin(theta) * 1  # 1 AU
    
    fig.add_trace(
        go.Scatter(
            x=earth_orbit_x,
            y=earth_orbit_y,
            mode='lines',
            line=dict(color='lightblue', width=1, dash='dot'),
            name='Earth Orbit (1 AU)'
        )
    )
    
    # Process each asteroid
    for _, asteroid in asteroid_data.iterrows():
        # Calculate asteroid path
        # We'll use a simplified model where asteroids approach on random trajectories
        
        # Convert miss distance to millions of km
        miss_distance = asteroid['miss_distance'] / 1000000
        
        # Generate a random approach angle
        approach_angle = np.random.uniform(0, 2*np.pi)
        
        # Start and end points (extend 3x beyond miss distance)
        max_distance = 3 * miss_distance
        
        # Generate path points
        path_distances = np.linspace(max_distance, miss_distance, 50)
        path_distances = np.concatenate([path_distances, np.linspace(miss_distance, max_distance, 50)])
        
        # Calculate x and y coordinates
        path_x = np.cos(approach_angle) * path_distances
        path_y = np.sin(approach_angle) * path_distances
        
        # Add slight curve to path
        curve_factor = np.random.uniform(0.1, 0.3)
        perpendicular_angle = approach_angle + np.pi/2
        curve_x = np.sin(np.linspace(0, np.pi, 100)) * curve_factor * miss_distance * np.cos(perpendicular_angle)
        curve_y = np.sin(np.linspace(0, np.pi, 100)) * curve_factor * miss_distance * np.sin(perpendicular_angle)
        
        path_x += curve_x
        path_y += curve_y
        
        # Add asteroid path
        fig.add_trace(
            go.Scatter(
                x=path_x,
                y=path_y,
                mode='lines',
                line=dict(
                    color='red' if asteroid['is_potentially_hazardous'] else 'orange',
                    width=2
                ),
                name=asteroid['name'],
                hoverinfo='name+text',
                hovertext=[
                    f"Name: {asteroid['name']}<br>"
                    f"Date: {asteroid['close_approach_date_display']}<br>"
                    f"Diameter: {asteroid['estimated_diameter_max']*1000:.1f} m<br>"
                    f"Velocity: {asteroid['relative_velocity']:.1f} km/s<br>"
                    f"Miss Distance: {miss_distance:.2f} million km"
                ] * len(path_x)
            )
        )
        
        # Add asteroid position
        fig.add_trace(
            go.Scatter(
                x=[path_x[0]],
                y=[path_y[0]],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red' if asteroid['is_potentially_hazardous'] else 'orange',
                    symbol='diamond'
                ),
                name=f"{asteroid['name']} (position)",
                showlegend=False,
                hoverinfo='name+text',
                hovertext=[
                    f"Name: {asteroid['name']}<br>"
                    f"Date: {asteroid['close_approach_date_display']}<br>"
                    f"Diameter: {asteroid['estimated_diameter_max']*1000:.1f} m<br>"
                    f"Velocity: {asteroid['relative_velocity']:.1f} km/s<br>"
                    f"Miss Distance: {miss_distance:.2f} million km"
                ]
            )
        )
        
        # Add closest approach point
        closest_idx = len(path_distances) // 2  # Middle point is closest approach
        fig.add_trace(
            go.Scatter(
                x=[path_x[closest_idx]],
                y=[path_y[closest_idx]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='white',
                    symbol='circle',
                    line=dict(color='red' if asteroid['is_potentially_hazardous'] else 'orange', width=2)
                ),
                name=f"{asteroid['name']} (closest approach)",
                showlegend=False,
                hoverinfo='name+text',
                hovertext=[
                    f"Closest Approach:<br>"
                    f"Distance: {miss_distance:.2f} million km<br>"
                    f"Date: {asteroid['close_approach_date_display']}"
                ]
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Asteroid Trajectories Relative to Earth",
        xaxis=dict(
            title="Distance (million km)",
            scaleanchor="y",  # Force equal scaling
            scaleratio=1      # Force equal scaling
        ),
        yaxis=dict(
            title="Distance (million km)"
        ),
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Add annotations for scale
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=-earth_radius,
        y0=-earth_radius,
        x1=earth_radius,
        y1=earth_radius,
        line_color="blue",
        fillcolor="blue",
        opacity=0.5
    )
    
    fig.add_annotation(
        x=0, y=0,
        text="Earth",
        showarrow=True,
        arrowhead=1,
        ax=20, ay=20
    )
    
    # Add Moon orbit for scale
    moon_orbit = 384400 / 1000000  # ~0.384 million km
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=-moon_orbit,
        y0=-moon_orbit,
        x1=moon_orbit,
        y1=moon_orbit,
        line=dict(color="lightgray", width=1, dash="dot"),
        fillcolor="rgba(0,0,0,0)"
    )
    
    fig.add_annotation(
        x=moon_orbit, y=0,
        text="Moon Orbit",
        showarrow=True,
        arrowhead=1,
        ax=20, ay=-20
    )
    
    return fig
