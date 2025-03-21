import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import fetch_neo_data, preprocess_data, feature_engineering
from utils.visualization import (
    plot_asteroids, visualize_top_3_hazardous_asteroids,
    plot_feature_importance, plot_correlation_heatmap,
    plot_feature_distributions, plot_roc_curve, plot_precision_recall_curve,
    create_3d_asteroid_visualization, create_asteroid_trajectory_animation,
    create_interactive_asteroid_paths
)
from utils.model_training import train_and_evaluate_models, load_cached_model
from utils.export import export_data, export_visualization

# Set page config
st.set_page_config(
    page_title="Comprehensive Space Threat Assessment and Prediction System",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NASA API Key
NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")

# Theme configuration (Dark/Light mode)
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"  # Default to dark theme

# Function to apply theme styling
def apply_theme_styling():
    if st.session_state.theme == "dark":
        # Apply dark theme styling
        st.markdown("""
        <style>
        .main {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stButton>button {
            background-color: #2196F3; /* Blue buttons */
            color: white;
            border: 1px solid #1E88E5;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .stButton>button:hover {
            background-color: #1976D2;
            border-color: #1565C0;
            box-shadow: 0 3px 7px rgba(0,0,0,0.3);
        }
        .stButton>button[kind="primary"] {
            background-color: #4CAF50; /* Green for primary buttons */
            border-color: #43A047;
            color: white;
        }
        .stButton>button[kind="primary"]:hover {
            background-color: #388E3C;
            box-shadow: 0 3px 7px rgba(0,0,0,0.3);
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #262730;
            color: #FAFAFA;
            border-radius: 4px;
        }
        .stSelectbox>div>div {
            background-color: #262730;
            color: #FAFAFA;
            border-radius: 4px;
        }
        .st-br {
            border-color: #4B4F5A;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0 0;
            padding: 0.5rem 1rem;
            background-color: #262730;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4CAF50;
            color: white;
        }
        .info-box {
            background-color: #262730;
            border-left: 4px solid #4CAF50;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .alert-box {
            background-color: #262730;
            border-left: 4px solid #F44336;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Apply light theme styling
        st.markdown("""
        <style>
        .main {
            background-color: #FFFFFF;
            color: #31333F;
        }
        .stButton>button {
            background-color: #F0F2F6;
            color: #31333F;
            border: 1px solid #D2D6DE;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #E2E8F0;
            border-color: #C0C7D1;
        }
        .stButton>button[kind="primary"] {
            background-color: #4CAF50;
            color: white;
            border-color: #3E8E41;
        }
        .stButton>button[kind="primary"]:hover {
            background-color: #3E8E41;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #F0F2F6;
            color: #31333F;
            border-radius: 4px;
        }
        .stSelectbox>div>div {
            background-color: #F0F2F6;
            color: #31333F;
            border-radius: 4px;
        }
        .st-br {
            border-color: #D2D6DE;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0 0;
            padding: 0.5rem 1rem;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4CAF50;
            color: white;
        }
        .info-box {
            background-color: #F0F2F6;
            border-left: 4px solid #4CAF50;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .alert-box {
            background-color: #F0F2F6;
            border-left: 4px solid #F44336;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

# Apply theme styling
apply_theme_styling()

# Initialize session state for advanced settings
if 'show_advanced' not in st.session_state:
    st.session_state.show_advanced = False

# Initialize session state for model cache
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
    st.session_state.model_metrics = None
    st.session_state.feature_importance = None

# Initialize session state for visualizations
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = {}

# Sidebar for inputs
with st.sidebar:
    st.title("☄️ Space Threat Assessment")
    
    # Date range picker
    st.subheader("Date Range")
    today = datetime.today().date()
    default_start_date = today - timedelta(days=7)
    
    start_date = st.date_input("Start Date", value=default_start_date)
    end_date = st.date_input("End Date", value=today)
    
    # Convert to string format for API
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # Calculate date range in days
    date_range = (end_date - start_date).days
    if date_range > 7:
        st.warning("⚠️ NASA API limits requests to 7 days at a time. The query will be limited to the first 7 days.")
        end_date_str = (start_date + timedelta(days=7)).strftime("%Y-%m-%d")
    
    # Model selection
    st.subheader("Model Selection")
    model_type = st.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "XGBoost", "k-Nearest Neighbors", "Neural Network"]
    )
    
    # Advanced Settings toggle
    if st.button("Show Advanced Settings" if not st.session_state.show_advanced else "Hide Advanced Settings"):
        st.session_state.show_advanced = not st.session_state.show_advanced
    
    if st.session_state.show_advanced:
        st.subheader("Advanced Settings")
        
        # Visualization settings
        st.write("Visualization Settings")
        visualization_type = st.selectbox(
            "Visualization Type",
            ["2D Scatter", "3D Scatter", "Asteroid Trajectories"]
        )
        
        # Data preprocessing settings
        st.write("Data Preprocessing")
        scaling_method = st.selectbox(
            "Feature Scaling Method",
            ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
        )
        
        remove_outliers = st.checkbox("Remove Outliers", value=False)
        if remove_outliers:
            outlier_threshold = st.slider("Outlier Threshold (Z-Score)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        
        # Model hyperparameter settings
        st.write("Model Hyperparameters")
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", min_value=50, max_value=500, value=200, step=10)
            max_depth = st.slider("Max Depth", min_value=3, max_value=20, value=10, step=1)
        elif model_type == "XGBoost":
            learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
            max_depth = st.slider("Max Depth", min_value=3, max_value=15, value=6, step=1)
        elif model_type == "Neural Network":
            hidden_layers = st.slider("Hidden Layers", min_value=1, max_value=5, value=2, step=1)
            neurons_per_layer = st.slider("Neurons per Layer", min_value=5, max_value=100, value=32, step=5)
        
        # Hyperparameter optimization
        use_hyperopt = st.checkbox("Use Hyperparameter Optimization", value=False)
        if use_hyperopt:
            n_trials = st.slider("Number of Optimization Trials", min_value=10, max_value=100, value=30, step=5)
    
    # Button to fetch new data with styled container
    st.write("---")
    
    # Create a styled container for the fetch button
    st.markdown("""
    <style>
    .fetch-button-container {
        background-color: rgba(76, 175, 80, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #4CAF50;
        text-align: center;
    }
    .fetch-button-title {
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 1.1em;
    }
    </style>
    <div class="fetch-button-container">
        <div class="fetch-button-title">🔍 Retrieve Latest NEO Data</div>
        <p style="font-size:0.9em; margin-bottom:10px;">
        Click below to fetch the most current asteroid data from NASA's Near-Earth Object database
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    fetch_button = st.button("Fetch NEO Data", type="primary", key="fancy_fetch_button")
    
    # Export options with improved styling
    st.markdown("""
    <style>
    .export-container {
        background-color: rgba(25, 118, 210, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
        border-left: 4px solid #1976D2;
    }
    </style>
    <div class="export-container">
        <div style="font-weight: bold; margin-bottom: 10px; font-size: 1.1em;">📊 Export Options</div>
    </div>
    """, unsafe_allow_html=True)
    
    export_format = st.selectbox(
        "Export Format",
        ["CSV", "JSON", "Excel", "HTML", "PDF"]
    )
    export_content = st.selectbox(
        "Export Content",
        ["Data", "Visualization", "Model Results", "All"]
    )
    export_button = st.button("Export Data", key="fancy_export_button")

# Main content area
col1, col2 = st.columns([10, 2])
with col1:
    st.markdown("<h1 style='font-size:2.0em; color:#2E86C1;'>Comprehensive Space Threat Assessment and Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.2em; margin-bottom:20px;'>An advanced asteroid analysis platform for tracking, visualizing, and predicting hazards from Near Earth Objects</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.9em; color:#888; margin-bottom:20px;'>
    Advanced monitoring and prediction platform for near-Earth objects, providing threat analysis, 
    impact simulations, and risk assessment for potential hazardous asteroids approaching Earth.
    </div>
    """, unsafe_allow_html=True)
with col2:
    # Theme toggle button
    if st.button("🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        apply_theme_styling()
        st.rerun()

# Add space threat assessment info box with custom styling
st.markdown("""
<style>
.info-box {
    background-color: rgba(25, 118, 210, 0.05);
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
    border-left: 6px solid #1976D2;
}
</style>
<div class="info-box">
    <h3 style="color: #1976D2; margin-top:0;">About Space Threat Assessment</h3>
    <p>
    This application analyzes Near-Earth Objects (NEOs) to identify potentially hazardous asteroids
    that could pose a threat to Earth. By leveraging NASA's NEO database and advanced machine
    learning algorithms, the system provides comprehensive risk assessment and predictive analytics.
    </p>
</div>
""", unsafe_allow_html=True)

# Fetch data on button click or use cached data
if fetch_button:
    with st.spinner("Fetching asteroid data from NASA API..."):
        neo_data = fetch_neo_data(NASA_API_KEY, start_date_str, end_date_str)
        
        if neo_data:
            df = preprocess_data(neo_data)
            df = feature_engineering(df)
            
            # Advanced data preprocessing if enabled
            if st.session_state.show_advanced:
                from utils.data_processing import apply_scaling, remove_outliers_zscore
                
                if scaling_method != "None":
                    df = apply_scaling(df, method=scaling_method)
                
                if remove_outliers:
                    df = remove_outliers_zscore(df, threshold=outlier_threshold)
                    
            # Store in session state
            st.session_state['neo_df'] = df
            st.success(f"✅ Successfully fetched data for {len(df)} asteroids!")
        else:
            st.error("Failed to fetch data from NASA API. Please try again.")

# Main tabs
if 'neo_df' in st.session_state:
    df = st.session_state['neo_df']
    
    tabs = st.tabs(["Overview", "Visualizations", "Threat Assessment", "Model Insights", "Data Explorer"])
    
    # Overview Tab
    with tabs[0]:
        st.subheader("Near-Earth Object Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_objects = len(df)
            st.metric("Total NEOs", total_objects)
        with col2:
            potentially_hazardous = df['is_potentially_hazardous'].sum()
            st.metric("Potentially Hazardous", potentially_hazardous)
        with col3:
            avg_miss_distance = df['miss_distance'].mean() / 1000000  # km
            st.metric("Avg. Miss Distance", f"{avg_miss_distance:.2f} million km")
        with col4:
            max_diameter = df['estimated_diameter_max'].max() * 1000  # meters
            st.metric("Max Diameter", f"{max_diameter:.1f} m")
        
        # Distribution plots of key features
        st.subheader("NEO Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x="absolute_magnitude", 
                              title="Distribution of Absolute Magnitude",
                              labels={"absolute_magnitude": "Absolute Magnitude (H)"},
                              color_discrete_sequence=['#4CAF50'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x="estimated_diameter_max", 
                              title="Distribution of Estimated Diameter",
                              labels={"estimated_diameter_max": "Maximum Estimated Diameter (km)"},
                              color_discrete_sequence=['#2196F3'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Overview map of asteroids
        st.subheader("Miss Distance vs. Relative Velocity")
        fig = plot_asteroids(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hazardous asteroids
        st.subheader("Top 3 Potentially Hazardous Asteroids")
        fig = visualize_top_3_hazardous_asteroids(df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Visualizations Tab
    with tabs[1]:
        st.subheader("Advanced Visualizations")
        
        viz_type = st.radio(
            "Select Visualization Type",
            ["Asteroid Distribution", "3D Space Visualization", "Trajectory Analysis", "Time Series Analysis"]
        )
        
        if viz_type == "Asteroid Distribution":
            # Distribution visualization
            st.subheader("NEO Distribution in Space")
            fig = plot_feature_distributions(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Feature Correlation")
            fig = plot_correlation_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "3D Space Visualization":
            # 3D visualization of asteroids
            st.subheader("3D Asteroid Positions")
            fig = create_3d_asteroid_visualization(df)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Trajectory Analysis":
            # Asteroid trajectories
            st.subheader("Asteroid Trajectories")
            
            # Select specific asteroids for trajectory
            top_hazardous = df.sort_values('diameter_velocity_ratio', ascending=False).head(10)
            selected_asteroid = st.selectbox(
                "Select Asteroid for Trajectory Analysis",
                top_hazardous['name'].tolist()
            )
            
            selected_data = df[df['name'] == selected_asteroid]
            
            if not selected_data.empty:
                fig = create_interactive_asteroid_paths(selected_data)
                st.plotly_chart(fig, use_container_width=True)
            
                # Additional trajectory animation
                st.subheader("Animated Trajectory")
                fig = create_asteroid_trajectory_animation(selected_data)
                st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Time Series Analysis":
            # Time series of NEO approaches
            st.subheader("NEO Approaches Over Time")
            
            # Group by date
            df['approach_date'] = pd.to_datetime(df['close_approach_date'])
            daily_counts = df.groupby('approach_date').size().reset_index(name='count')
            daily_hazardous = df[df['is_potentially_hazardous']].groupby('approach_date').size().reset_index(name='hazardous_count')
            
            # Merge the two dataframes
            merged_df = pd.merge(daily_counts, daily_hazardous, on='approach_date', how='left')
            merged_df['hazardous_count'] = merged_df['hazardous_count'].fillna(0)
            
            # Create a time series plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged_df['approach_date'], 
                y=merged_df['count'],
                mode='lines+markers',
                name='All NEOs',
                line=dict(color='#2196F3', width=2),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=merged_df['approach_date'], 
                y=merged_df['hazardous_count'],
                mode='lines+markers',
                name='Hazardous NEOs',
                line=dict(color='#F44336', width=2),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title='NEO Approaches by Date',
                xaxis_title='Date',
                yaxis_title='Number of NEOs',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Threat Assessment Tab
    with tabs[2]:
        st.subheader("Threat Assessment Analysis")
        
        # Model training and prediction
        with st.spinner("Training model and generating predictions..."):
            # Get model hyperparameters from advanced settings
            model_params = {}
            if st.session_state.show_advanced:
                if model_type == "Random Forest":
                    model_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth
                    }
                elif model_type == "XGBoost":
                    model_params = {
                        'learning_rate': learning_rate,
                        'max_depth': max_depth
                    }
                elif model_type == "Neural Network":
                    model_params = {
                        'hidden_layers': hidden_layers,
                        'neurons_per_layer': neurons_per_layer
                    }
                
                # Add hyperparameter optimization if enabled
                if use_hyperopt:
                    model_params['optimize'] = True
                    model_params['n_trials'] = n_trials
            
            # Train model or load cached model if already trained with same parameters
            if (st.session_state.trained_model is None or 
                st.session_state.model_type != model_type or 
                st.session_state.model_params != model_params):
                
                model, metrics, feature_importance = train_and_evaluate_models(
                    df, model_type, model_params
                )
                
                # Cache the trained model and results
                st.session_state.trained_model = model
                st.session_state.model_metrics = metrics
                st.session_state.feature_importance = feature_importance
                st.session_state.model_type = model_type
                st.session_state.model_params = model_params
            else:
                # Use cached model
                model = st.session_state.trained_model
                metrics = st.session_state.model_metrics
                feature_importance = st.session_state.feature_importance
        
        # Display model performance metrics
        st.subheader(f"Model Performance ({model_type})")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.4f}")
        
        # ROC curve and Precision-Recall curve
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ROC Curve")
            fig = plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Precision-Recall Curve")
            fig = plot_precision_recall_curve(metrics['precision_curve'], metrics['recall_curve'], metrics['avg_precision'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance")
        fig = plot_feature_importance(feature_importance)
        st.plotly_chart(fig, use_container_width=True)
        
        # High-risk asteroids
        st.subheader("High-Risk Asteroids")
        
        # Filter to show only the top risk asteroids
        if 'hazard_probability' in df.columns:
            high_risk_df = df.sort_values('hazard_probability', ascending=False).head(10)
            high_risk_df = high_risk_df[['name', 'close_approach_date_display', 'hazard_probability', 
                                         'estimated_diameter_max', 'miss_distance', 'relative_velocity']]
            
            # Format columns for display
            high_risk_df['hazard_probability'] = high_risk_df['hazard_probability'].apply(lambda x: f"{x:.4f}")
            high_risk_df['estimated_diameter_max'] = high_risk_df['estimated_diameter_max'].apply(lambda x: f"{x*1000:.1f} m")
            high_risk_df['miss_distance'] = high_risk_df['miss_distance'].apply(lambda x: f"{x/1000000:.2f} million km")
            high_risk_df['relative_velocity'] = high_risk_df['relative_velocity'].apply(lambda x: f"{x:.2f} km/s")
            
            # Rename columns for display
            high_risk_df.columns = ['Asteroid Name', 'Approach Date', 'Hazard Probability', 
                                   'Diameter (max)', 'Miss Distance', 'Relative Velocity']
            
            st.dataframe(high_risk_df, use_container_width=True)
        else:
            st.warning("Hazard probabilities not available. Please retrain the model.")
    
    # Model Insights Tab
    with tabs[3]:
        st.subheader("Model Insights and Analysis")
        
        # Display model details
        st.info(f"Current Model: {model_type}")
        
        if st.session_state.model_params:
            st.write("Model Parameters:")
            st.json(st.session_state.model_params)
        
        # Feature analysis
        st.subheader("Feature Analysis")
        
        # Allow selecting specific features for analysis
        feature_list = [col for col in df.columns if col not in ['id', 'name', 'close_approach_date', 'orbiting_body', 'has_missing_data', 'is_potentially_hazardous']]
        selected_features = st.multiselect(
            "Select Features for Analysis",
            feature_list,
            default=feature_list[:3]
        )
        
        if len(selected_features) >= 2:
            # Create scatter plot matrix
            fig = px.scatter_matrix(
                df, 
                dimensions=selected_features,
                color="is_potentially_hazardous", 
                color_discrete_sequence=['#2196F3', '#F44336'],
                opacity=0.7
            )
            fig.update_layout(title="Feature Relationships")
            st.plotly_chart(fig, use_container_width=True)
        elif len(selected_features) == 1:
            # Create histogram for single feature
            fig = px.histogram(
                df, 
                x=selected_features[0],
                color="is_potentially_hazardous",
                color_discrete_sequence=['#2196F3', '#F44336'],
                barmode="overlay",
                opacity=0.7
            )
            fig.update_layout(title=f"Distribution of {selected_features[0]}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Please select at least one feature for analysis.")
        
        # Feature correlation with hazard probability
        if 'hazard_probability' in df.columns:
            st.subheader("Feature Correlation with Hazard Probability")
            
            # Calculate correlations with hazard_probability
            correlation_df = pd.DataFrame(df[feature_list].corrwith(df['hazard_probability']))
            correlation_df.columns = ['correlation']
            correlation_df = correlation_df.sort_values('correlation', ascending=False)
            
            fig = px.bar(
                correlation_df,
                x=correlation_df.index,
                y='correlation',
                title="Correlation with Hazard Probability",
                color='correlation',
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(xaxis_title="Feature", yaxis_title="Correlation")
            st.plotly_chart(fig, use_container_width=True)
    
    # Data Explorer Tab
    with tabs[4]:
        st.subheader("Raw Data Explorer")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_hazardous = st.checkbox("Show only hazardous asteroids")
        
        with col2:
            min_diameter = float(df['estimated_diameter_max'].min())
            max_diameter = float(df['estimated_diameter_max'].max())
            diameter_range = st.slider(
                "Diameter Range (km)",
                min_value=min_diameter,
                max_value=max_diameter,
                value=(min_diameter, max_diameter)
            )
        
        # Apply filters
        filtered_df = df.copy()
        if filter_hazardous:
            filtered_df = filtered_df[filtered_df['is_potentially_hazardous'] == True]
        
        filtered_df = filtered_df[
            (filtered_df['estimated_diameter_max'] >= diameter_range[0]) &
            (filtered_df['estimated_diameter_max'] <= diameter_range[1])
        ]
        
        # Display filtered data
        st.dataframe(filtered_df, use_container_width=True)
        
        # Data statistics
        st.subheader("Data Statistics")
        
        # Select columns for statistics
        stat_columns = st.multiselect(
            "Select columns for statistics",
            df.select_dtypes(include=['number']).columns.tolist(),
            default=['absolute_magnitude', 'estimated_diameter_max', 'miss_distance', 'relative_velocity']
        )
        
        if stat_columns:
            st.dataframe(filtered_df[stat_columns].describe(), use_container_width=True)
        
        # Download data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_neo_data.csv",
            mime="text/csv",
        )

# Handle export button
if export_button and 'neo_df' in st.session_state:
    try:
        with st.spinner("Exporting data..."):
            export_path = export_data(
                st.session_state['neo_df'],
                format=export_format.lower(),
                content=export_content.lower()
            )
            
            if export_format.lower() == 'pdf' and export_content.lower() in ['visualization', 'all']:
                # For PDF exports with visualizations, we need to export the visualizations first
                if 'trained_model' in st.session_state and st.session_state.trained_model is not None:
                    metrics = st.session_state.model_metrics
                    feature_importance = st.session_state.feature_importance
                    
                    export_visualization(
                        st.session_state['neo_df'],
                        metrics,
                        feature_importance,
                        format='pdf'
                    )
            
            st.success(f"Successfully exported data as {export_format}!")
            
            # Generate download link if applicable
            if export_format.lower() in ['csv', 'json', 'excel', 'html']:
                with open(export_path, 'rb') as f:
                    data = f.read()
                
                file_extension = {
                    'csv': 'csv',
                    'json': 'json',
                    'excel': 'xlsx',
                    'html': 'html'
                }[export_format.lower()]
                
                st.download_button(
                    label=f"Download {export_format} File",
                    data=data,
                    file_name=f"neo_data.{file_extension}",
                    mime=f"application/{file_extension}"
                )
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")

# Show application information at the bottom
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8em;">
    <p>Comprehensive Space Threat Assessment and Prediction System | Using NASA NEO API</p>
    <p>Data refreshed: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
