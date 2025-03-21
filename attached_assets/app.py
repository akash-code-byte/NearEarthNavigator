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
NASA_API_KEY = os.environ.get("NASA_API_KEY", "yl3iawXawys50GTGtKdzQ9TmbKlJpmoptjC8Shqb")

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
    # Set theme to light by default
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
        apply_theme_styling()

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

# Add space threat assessment info box with custom styling
st.markdown("""
<style>
.info-box {
    background-color: rgba(25, 118, 210, 0.05);
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
    border-left: 6px solid #1976D2;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.info-box h4 {
    margin-top: 15px;
    margin-bottom: 10px;
    color: #333;
    font-weight: 600;
}
.info-box ul {
    margin-bottom: 15px;
}
</style>
<div class="info-box">
    <h4>Current Statistics (as of March 2025):</h4>
    <ul>
        <li>Over 31,000 NEOs cataloged by NASA</li>
        <li>Approximately 2,300 potentially hazardous asteroids identified</li>
        <li>Continuous monitoring through global observation networks</li>
    </ul>
    
    <h4>Threat Classification System:</h4>
    <ul>
        <li><strong style="color:#4CAF50;">Low Risk</strong> - Objects with minimal chance of Earth impact</li>
        <li><strong style="color:#FFC107;">Moderate Risk</strong> - Objects requiring continued observation</li>
        <li><strong style="color:#FF5722;">High Risk</strong> - Objects with significant impact potential</li>
        <li><strong style="color:#F44336;">Critical Risk</strong> - Objects requiring immediate attention and contingency planning</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Process and display data if available
if 'neo_df' in st.session_state:
    df = st.session_state['neo_df']
    
    # Display current threat assessment summary with custom styling
    st.markdown("""
    <style>
    .metrics-container {
        background-color: rgba(25, 118, 210, 0.05);
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .metrics-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: #1976D2;
        font-size: 1.1em;
        border-bottom: 1px solid rgba(25, 118, 210, 0.2);
        padding-bottom: 5px;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
    }
    </style>
    <div class="metrics-container">
        <div class="metrics-title">Current Threat Assessment Summary</div>
    </div>
    """, unsafe_allow_html=True)
    
    threat_summary_col1, threat_summary_col2, threat_summary_col3, threat_summary_col4 = st.columns(4)
    
    with threat_summary_col1:
        st.metric("Objects Analyzed", len(df), delta=None)
    
    with threat_summary_col2:
        hazardous_count = df["is_potentially_hazardous"].sum() if "is_potentially_hazardous" in df.columns else 0
        st.metric("Hazardous Objects", int(hazardous_count), delta=None)
    
    with threat_summary_col3:
        closest_distance = df["miss_distance"].min() if "miss_distance" in df.columns and len(df) > 0 else "N/A"
        if isinstance(closest_distance, (int, float)):
            formatted_distance = f"{closest_distance:,.0f} km"
        else:
            formatted_distance = "N/A"
        st.metric("Closest Approach", formatted_distance, delta=None)
    
    with threat_summary_col4:
        largest_diameter = df["estimated_diameter_max"].max() if "estimated_diameter_max" in df.columns and len(df) > 0 else "N/A"
        if isinstance(largest_diameter, (int, float)):
            formatted_diameter = f"{largest_diameter:.2f} km"
        else:
            formatted_diameter = "N/A"
        st.metric("Largest Object", formatted_diameter, delta=None)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview", 
        "Threat Visualizations", 
        "Predictive Models", 
        "Impact Simulations",
        "Risk Assessment"
    ])
    
    with tab1:
        st.header("Asteroid Dataset")
        st.write(f"Total asteroids: {len(df)}")
        
        # Display statistics
        st.subheader("Dataset Statistics")
        st.write(df.describe())
        
        # Display the dataframe
        st.subheader("Raw Data")
        st.dataframe(df)

    with tab2:
        st.header("Threat Visualization & Analysis")
        
        # Visualization type selector
        vis_options = st.radio(
            "Select Visualization Type",
            ["Basic Scatter Plot", "3D Visualization", "Asteroid Trajectories", "Distribution Plots"]
        )
        
        if vis_options == "Basic Scatter Plot":
            st.subheader("Asteroid Distribution")
            fig = plot_asteroids(df, "is_potentially_hazardous", "Asteroid Distribution", show_hazard_colors=True)
            st.plotly_chart(fig, use_container_width=True, key="scatter_plot")
            
            st.subheader("Correlation Heatmap")
            corr_fig = plot_correlation_heatmap(df)
            st.plotly_chart(corr_fig, use_container_width=True, key="correlation_heatmap")
            
        elif vis_options == "3D Visualization":
            st.subheader("3D Asteroid Visualization")
            fig_3d = create_3d_asteroid_visualization(df)
            st.plotly_chart(fig_3d, use_container_width=True, key="3d_visualization")
            
        elif vis_options == "Asteroid Trajectories":
            st.subheader("Interactive Asteroid Trajectories")
            trajectory_fig = create_interactive_asteroid_paths(df)
            st.plotly_chart(trajectory_fig, use_container_width=True, key="trajectory_chart")
            
            st.markdown("""
            This visualization shows the paths of asteroids relative to Earth. Each asteroid is positioned at its closest approach distance.
            - **Red dots** represent potentially hazardous asteroids
            - **Green dots** represent non-hazardous asteroids
            - The concentric circles represent reference distances from Earth
            - Hover over points for detailed information about each asteroid
            """)
            
            # Provide a separate button to try the animation if desired
            if st.button("Try Animation Version (may not work in all environments)", key="animation_btn"):
                st.subheader("Asteroid Trajectory Animation")
                animation_fig = create_asteroid_trajectory_animation(df)
                st.plotly_chart(animation_fig, use_container_width=True, key="animation_chart")
            
        elif vis_options == "Distribution Plots":
            st.subheader("Feature Distributions")
            dist_fig = plot_feature_distributions(df)
            st.plotly_chart(dist_fig, use_container_width=True, key="distribution_plots")

    with tab3:
        st.header("Predictive Modeling")
        
        # Train models button with custom styling
        st.markdown("""
        <style>
        .train-model-container {
            background-color: rgba(76, 175, 80, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            border-left: 4px solid #4CAF50;
            text-align: center;
        }
        </style>
        <div class="train-model-container">
            <p style="font-size:0.9em; margin-bottom:10px;">
            Train machine learning models to predict asteroid hazard potential based on physical characteristics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Train Predictive Models", type="primary"):
            with st.spinner("Training models... This may take a while."):
                # Configure model training based on advanced settings if enabled
                training_config = {}
                
                if st.session_state.show_advanced:
                    if model_type == "Random Forest":
                        training_config = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth
                        }
                    elif model_type == "XGBoost":
                        training_config = {
                            "learning_rate": learning_rate,
                            "max_depth": max_depth
                        }
                    elif model_type == "Neural Network":
                        training_config = {
                            "hidden_layers": hidden_layers,
                            "neurons_per_layer": neurons_per_layer
                        }
                    
                    training_config["use_hyperopt"] = use_hyperopt
                    if use_hyperopt:
                        training_config["n_trials"] = n_trials
                        training_config["optimizer"] = "optuna"
                
                best_model, model_metrics, feature_importance = train_and_evaluate_models(
                    df, model_type=model_type, config=training_config
                )
                
                if best_model:
                    st.session_state['best_model'] = best_model
                    st.session_state['model_metrics'] = model_metrics
                    st.session_state['feature_importance'] = feature_importance
                    st.success("✅ Models trained successfully!")
        
        # Display model results if available
        if 'best_model' in st.session_state and 'model_metrics' in st.session_state:
            model_metrics = st.session_state['model_metrics']
            
            st.subheader("Model Performance Metrics")
            
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{model_metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{model_metrics['precision']:.4f}")
            col3.metric("Recall", f"{model_metrics['recall']:.4f}")
            col4.metric("F1 Score", f"{model_metrics['f1']:.4f}")
            
            # Display feature importance
            if 'feature_importance' in st.session_state:
                st.subheader("Feature Importance")
                try:
                    importance_fig = plot_feature_importance(
                        st.session_state['best_model'], 
                        st.session_state['feature_importance']
                    )
                    if importance_fig is not None:
                        st.plotly_chart(importance_fig, use_container_width=True, key="feature_importance")
                    else:
                        st.info("Feature importance visualization is not available for this model type.")
                except Exception as e:
                    st.error(f"Error generating feature importance plot: {str(e)}")
            
            # Plot ROC and PR curves
            if 'y_test' in st.session_state and 'y_pred_proba' in st.session_state:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("ROC Curve")
                    try:
                        roc_fig = plot_roc_curve(
                            st.session_state['y_test'], 
                            st.session_state['y_pred_proba']
                        )
                        if roc_fig is not None:
                            st.plotly_chart(roc_fig, use_container_width=True, key="roc_curve")
                        else:
                            st.info("ROC curve could not be generated for this model.")
                    except Exception as e:
                        st.error(f"Error generating ROC curve: {str(e)}")
                
                with col2:
                    st.subheader("Precision-Recall Curve")
                    try:
                        pr_fig = plot_precision_recall_curve(
                            st.session_state['y_test'], 
                            st.session_state['y_pred_proba']
                        )
                        if pr_fig is not None:
                            st.plotly_chart(pr_fig, use_container_width=True, key="pr_curve")
                        else:
                            st.info("Precision-Recall curve could not be generated for this model.")
                    except Exception as e:
                        st.error(f"Error generating Precision-Recall curve: {str(e)}")

    with tab4:
        st.header("Impact Simulation & Risk Assessment")
        
        # Create three simulation options
        simulation_type = st.radio(
            "Select Simulation Type",
            ["Single Asteroid Impact", "Asteroid Collision", "Asteroid Fragments"],
            horizontal=True
        )
        
        if simulation_type == "Single Asteroid Impact":
            st.subheader("Asteroid Impact Simulation")
            
            # Impact simulator inputs
            st.write("Simulate the impact of an asteroid with Earth")
            
        elif simulation_type == "Asteroid Collision":
            st.subheader("Asteroid Collision Simulation")
            
            # Asteroid collision simulator inputs
            st.write("Simulate a collision between two asteroids and its consequences")
            
            # Create two columns for input form
            collision_col1, collision_col2 = st.columns(2)
            
            with collision_col1:
                # Inputs for first asteroid
                st.markdown("#### First Asteroid")
                asteroid1_diameter = st.number_input(
                    "Diameter (meters)", 
                    min_value=1.0, 
                    max_value=5000.0, 
                    value=100.0,
                    step=10.0,
                    key="ast1_diam"
                )
                
                asteroid1_density = st.number_input(
                    "Density (kg/m³)",
                    min_value=1000.0,
                    max_value=8000.0,
                    value=3000.0,
                    step=500.0,
                    key="ast1_dens"
                )
                
                asteroid1_velocity = st.number_input(
                    "Velocity (km/s)",
                    min_value=1.0,
                    max_value=72.0,
                    value=15.0,
                    step=1.0,
                    key="ast1_vel"
                )
            
            with collision_col2:
                # Inputs for second asteroid
                st.markdown("#### Second Asteroid")
                asteroid2_diameter = st.number_input(
                    "Diameter (meters)", 
                    min_value=1.0, 
                    max_value=5000.0, 
                    value=80.0,
                    step=10.0,
                    key="ast2_diam"
                )
                
                asteroid2_density = st.number_input(
                    "Density (kg/m³)",
                    min_value=1000.0,
                    max_value=8000.0,
                    value=2500.0,
                    step=500.0,
                    key="ast2_dens"
                )
                
                asteroid2_velocity = st.number_input(
                    "Velocity (km/s)",
                    min_value=1.0,
                    max_value=72.0,
                    value=18.0,
                    step=1.0,
                    key="ast2_vel"
                )
            
            # Collision angle
            collision_angle = st.slider(
                "Collision Angle (degrees)",
                min_value=0,
                max_value=180,
                value=45,
                help="0° = head-on collision, 90° = perpendicular, 180° = rear-end collision"
            )
            
            # Distance from Earth
            distance_from_earth = st.number_input(
                "Distance from Earth (million km)",
                min_value=0.1,
                max_value=150.0,
                value=10.0,
                step=1.0
            )
            
            # Run collision simulation button with custom styling
            st.markdown("""
            <style>
            .collision-container {
                background-color: rgba(156, 39, 176, 0.1);
                border-radius: 10px;
                padding: 15px;
                margin: 20px 0;
                border-left: 4px solid #9C27B0;
                text-align: center;
            }
            </style>
            <div class="collision-container">
                <p style="font-size:0.9em; margin-bottom:10px;">
                Simulate an asteroid collision in space and calculate the resulting fragments and trajectories
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Run Collision Simulation", type="primary", key="collision_btn"):
                # Calculate asteroid masses
                volume1 = (4/3) * np.pi * ((asteroid1_diameter / 2) ** 3)
                mass1 = volume1 * asteroid1_density
                
                volume2 = (4/3) * np.pi * ((asteroid2_diameter / 2) ** 3)
                mass2 = volume2 * asteroid2_density
                
                # Convert velocities to m/s
                velocity1_ms = asteroid1_velocity * 1000
                velocity2_ms = asteroid2_velocity * 1000
                
                # Calculate momenta
                momentum1 = mass1 * velocity1_ms
                momentum2 = mass2 * velocity2_ms
                
                # Calculate collision energy
                relative_velocity = abs(velocity1_ms - velocity2_ms)
                collision_factor = np.sin(np.radians(collision_angle / 2))  # Factor based on collision angle
                collision_energy = 0.5 * (mass1 + mass2) * (relative_velocity * collision_factor) ** 2
                
                # Convert to kilotons of TNT
                collision_energy_kt = collision_energy / 4.184e12
                
                # Estimate number of fragments based on energy
                fragment_count = min(int((collision_energy_kt) ** 0.5) + 2, 100)
                
                # Display results in expander
                with st.expander("Collision Simulation Results", expanded=True):
                    # Show summary metrics
                    st.markdown(f"### Collision Energy: **{collision_energy_kt:.2f} kilotons** of TNT")
                    
                    # Compare to known events
                    if collision_energy_kt < 1:
                        st.write(f"This is a relatively minor collision, comparable to a small conventional explosion.")
                    elif collision_energy_kt < 100:
                        st.write(f"This collision releases energy similar to a small nuclear weapon.")
                    else:
                        st.write(f"This is a major collision event with significant energy release.")
                    
                    # Create metrics
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.metric("Total Mass", f"{(mass1 + mass2)/1000000:.2f} million kg")
                        st.metric("Estimated Fragments", f"{fragment_count}")
                    
                    with metrics_col2:
                        st.metric("Distance from Earth", f"{distance_from_earth:.1f} million km")
                        st.metric("Earth Impact Risk", f"{'Low' if distance_from_earth > 20 else 'Moderate' if distance_from_earth > 5 else 'High'}")
                    
                    # Visualization of collision
                    st.subheader("Collision Visualization")
                    
                    # Create visualization using plotly
                    fig = go.Figure()
                    
                    # Add pre-collision asteroids
                    fig.add_trace(go.Scatter(
                        x=[-50], y=[0],
                        mode="markers",
                        marker=dict(
                            size=asteroid1_diameter/10,
                            color="brown",
                            symbol="circle"
                        ),
                        name="Asteroid 1"
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[50], y=[0],
                        mode="markers",
                        marker=dict(
                            size=asteroid2_diameter/10,
                            color="gray",
                            symbol="circle"
                        ),
                        name="Asteroid 2"
                    ))
                    
                    # Add collision point
                    fig.add_trace(go.Scatter(
                        x=[0], y=[0],
                        mode="markers",
                        marker=dict(
                            size=30,
                            color="orange",
                            opacity=0.7,
                            symbol="star"
                        ),
                        name="Collision Point"
                    ))
                    
                    # Add fragments
                    np.random.seed(42)  # For reproducible results
                    fragment_sizes = np.random.exponential(scale=5, size=fragment_count)
                    fragment_sizes = np.clip(fragment_sizes, 2, 20)
                    
                    angles = np.random.uniform(0, 2*np.pi, fragment_count)
                    distances = np.random.uniform(10, 100, fragment_count)
                    
                    x_coords = np.cos(angles) * distances
                    y_coords = np.sin(angles) * distances
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords, y=y_coords,
                        mode="markers",
                        marker=dict(
                            size=fragment_sizes,
                            color="red",
                            opacity=0.6,
                            symbol="circle"
                        ),
                        name="Fragments"
                    ))
                    
                    # Add Earth (for scale)
                    earth_distance = distance_from_earth * 10  # Scale for visualization
                    fig.add_trace(go.Scatter(
                        x=[-earth_distance], y=[0],
                        mode="markers+text",
                        marker=dict(
                            size=12,
                            color="blue",
                            symbol="circle"
                        ),
                        text=["Earth"],
                        textposition="top center",
                        name="Earth"
                    ))
                    
                    # Add trajectory lines
                    for i in range(min(20, fragment_count)):  # Show trajectories for up to 20 fragments
                        x_end = x_coords[i] * 2
                        y_end = y_coords[i] * 2
                        
                        # Check if trajectory points toward Earth
                        points_to_earth = (x_end < 0 and abs(y_end / x_end) < 0.2)
                        
                        fig.add_trace(go.Scatter(
                            x=[0, x_end], y=[0, y_end],
                            mode="lines",
                            line=dict(
                                color="red" if points_to_earth else "gray",
                                width=2 if points_to_earth else 1,
                                dash="solid" if points_to_earth else "dot"
                            ),
                            showlegend=False
                        ))
                    
                    # Improve layout
                    fig.update_layout(
                        title="Asteroid Collision and Fragment Trajectories",
                        xaxis_title="X Distance (arbitrary units)",
                        yaxis_title="Y Distance (arbitrary units)",
                        template="plotly_white",
                        height=500,
                        showlegend=True,
                        legend=dict(x=0, y=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add analysis
                    st.subheader("Collision Analysis")
                    
                    earth_risk = "low" if distance_from_earth > 20 else "moderate" if distance_from_earth > 5 else "high"
                    
                    if earth_risk == "low":
                        st.write("This collision poses minimal risk to Earth due to the large distance. Any fragments are unlikely to reach Earth's vicinity.")
                    elif earth_risk == "moderate":
                        st.write("This collision occurs at a moderate distance from Earth. Some fragments may enter Earth's vicinity within months, requiring monitoring.")
                    else:
                        st.write("This collision occurs dangerously close to Earth. Multiple fragments are likely to reach Earth's vicinity within weeks, posing significant hazards.")
                    
                    # Emergency response
                    if earth_risk != "low":
                        st.subheader("Response Recommendations")
                        st.markdown(f"""
                        - **Monitoring**: Immediate tracking of all fragments larger than {max(asteroid1_diameter, asteroid2_diameter)/20:.0f} meters
                        - **Impact Risk Assessment**: Calculate potential impact trajectories within {7 if earth_risk == "high" else 30} days
                        - **Deflection Options**: Evaluate potential deflection missions for any fragments on Earth-impact trajectories
                        - **Civil Defense**: {'Immediate preparation of emergency protocols' if earth_risk == "high" else 'Review of emergency protocols'}
                        """)
        
        elif simulation_type == "Asteroid Fragments":
            st.subheader("Asteroid Fragments Earth Impact")
            
            # Asteroid fragments simulator inputs
            st.write("Simulate multiple asteroid fragments hitting Earth simultaneously")
            
            # Parameters for fragment simulation
            fragments_col1, fragments_col2 = st.columns(2)
            
            with fragments_col1:
                parent_asteroid_size = st.number_input(
                    "Parent Asteroid Size (meters)",
                    min_value=50.0,
                    max_value=5000.0,
                    value=200.0,
                    step=50.0
                )
                
                fragment_count = st.slider(
                    "Number of Fragments",
                    min_value=2,
                    max_value=50,
                    value=8
                )
                
                fragment_density = st.number_input(
                    "Fragment Density (kg/m³)",
                    min_value=1000.0,
                    max_value=8000.0,
                    value=3000.0,
                    step=500.0
                )
                
            with fragments_col2:
                entry_angle = st.slider(
                    "Entry Angle (degrees from horizontal)",
                    min_value=5,
                    max_value=90,
                    value=45
                )
                
                spread_angle = st.slider(
                    "Fragment Spread (degrees)",
                    min_value=1,
                    max_value=60,
                    value=15,
                    help="Angular spread of fragments in the sky"
                )
                
                impact_velocity = st.number_input(
                    "Impact Velocity (km/s)",
                    min_value=11.0,
                    max_value=72.0,
                    value=20.0,
                    step=1.0
                )
            
            # Impact region selector
            impact_region = st.selectbox(
                "Primary Impact Region",
                ["Random Global", "Ocean", "Land (Urban)", "Land (Rural)", "Polar Region"]
            )
            
            # Run fragments simulation button with custom styling
            st.markdown("""
            <style>
            .fragments-container {
                background-color: rgba(0, 188, 212, 0.1);
                border-radius: 10px;
                padding: 15px;
                margin: 20px 0;
                border-left: 4px solid #00BCD4;
                text-align: center;
            }
            </style>
            <div class="fragments-container">
                <p style="font-size:0.9em; margin-bottom:10px;">
                Simulate multiple asteroid fragments hitting Earth and calculate the combined damage effects
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Run Fragments Simulation", type="primary", key="fragments_btn"):
                # Calculate total volume of parent asteroid
                parent_volume = (4/3) * np.pi * ((parent_asteroid_size / 2) ** 3)
                
                # Calculate fragment sizes (following a power law distribution)
                np.random.seed(42)  # For reproducible results
                
                # Generate fragment sizes using a power law distribution
                alpha = 1.8  # Power law exponent
                r_min = 0.05 * (parent_asteroid_size / 2)  # Minimum radius as fraction of parent
                r_max = 0.4 * (parent_asteroid_size / 2)   # Maximum radius as fraction of parent
                
                # Generate random values following power law
                u = np.random.uniform(0, 1, fragment_count)
                fragment_radii = ((r_max**(1-alpha) - r_min**(1-alpha)) * u + r_min**(1-alpha))**(1/(1-alpha))
                
                # Convert to diameters
                fragment_diameters = 2 * fragment_radii
                
                # Calculate fragment masses
                fragment_volumes = [(4/3) * np.pi * (radius**3) for radius in fragment_radii]
                fragment_masses = [volume * fragment_density for volume in fragment_volumes]
                
                # Calculate total fragment volume to verify conservation
                total_fragment_volume = sum(fragment_volumes)
                volume_ratio = total_fragment_volume / parent_volume
                
                # Adjust if total volume differs significantly from parent volume
                if abs(1 - volume_ratio) > 0.1:
                    adjustment_factor = (parent_volume / total_fragment_volume) ** (1/3)
                    fragment_radii = [radius * adjustment_factor for radius in fragment_radii]
                    fragment_diameters = [2 * radius for radius in fragment_radii]
                    fragment_volumes = [(4/3) * np.pi * (radius**3) for radius in fragment_radii]
                    fragment_masses = [volume * fragment_density for volume in fragment_volumes]
                
                # Generate impact locations
                # For simplicity, we'll distribute the impacts in a circular pattern
                central_angle = np.random.uniform(0, 360)  # Random central direction
                impact_angles = np.linspace(central_angle - spread_angle/2, central_angle + spread_angle/2, fragment_count)
                
                # Convert impact velocity to m/s
                velocity_ms = impact_velocity * 1000
                
                # Calculate impact energies
                impact_energies = [0.5 * mass * (velocity_ms ** 2) for mass in fragment_masses]
                
                # Convert to kilotons
                impact_energies_kt = [energy / 4.184e12 for energy in impact_energies]
                
                # Adjust for entry angle
                angle_factor = np.sin(np.radians(entry_angle))
                adjusted_energies_kt = [energy * angle_factor for energy in impact_energies_kt]
                
                # Calculate damage radii
                thermal_radii = [0.01 * (energy ** 0.5) for energy in adjusted_energies_kt]
                blast_radii = [0.03 * (energy ** 0.3333) for energy in adjusted_energies_kt]
                
                # Display results in expander
                with st.expander("Fragments Simulation Results", expanded=True):
                    # Show summary metrics
                    total_energy = sum(adjusted_energies_kt)
                    st.markdown(f"### Total Impact Energy: **{total_energy:.2f} kilotons** of TNT")
                    
                    # Compare to known events
                    if total_energy < 100:
                        comparison = f"less than the 1908 Tunguska event (~15,000 kt)"
                    elif total_energy < 15000:
                        comparison = f"similar to the 1908 Tunguska event (~15,000 kt)"
                    elif total_energy < 100000:
                        comparison = f"greater than the 1908 Tunguska event but less than the Chicxulub impact"
                    else:
                        comparison = f"approaching the scale of the Chicxulub impact that led to dinosaur extinction"
                    
                    st.write(f"The combined energy release is {comparison}.")
                    
                    # Create metrics
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.metric("Fragment Count", f"{fragment_count}")
                        st.metric("Largest Fragment", f"{max(fragment_diameters):.1f} m")
                        
                    with metrics_col2:
                        st.metric("Total Affected Area", f"{sum([np.pi * r**2 for r in blast_radii]):.2f} km²")
                        st.metric("Maximum Blast Radius", f"{max(blast_radii):.2f} km")
                    
                    # Fragment data table
                    st.subheader("Fragment Details")
                    
                    fragment_df = pd.DataFrame({
                        "Fragment": [f"F{i+1}" for i in range(fragment_count)],
                        "Diameter (m)": [f"{d:.1f}" for d in fragment_diameters],
                        "Mass (tons)": [f"{m/1000:.1f}" for m in fragment_masses],
                        "Energy (kt)": [f"{e:.2f}" for e in adjusted_energies_kt],
                        "Blast Radius (km)": [f"{r:.2f}" for r in blast_radii]
                    })
                    
                    st.dataframe(fragment_df.style.highlight_max(axis=0, color='red', subset=['Diameter (m)', 'Energy (kt)', 'Blast Radius (km)']))
                    
                    # Visualization of fragment impacts
                    st.subheader("Impact Visualization")
                    
                    # Create visualization using plotly
                    fig = go.Figure()
                    
                    # Draw Earth surface
                    x_range = 200  # Arbitrary units for visualization
                    fig.add_trace(go.Scatter(
                        x=np.linspace(-x_range, x_range, 100),
                        y=[0] * 100,
                        mode="lines",
                        line=dict(color="green", width=2),
                        name="Earth's Surface"
                    ))
                    
                    # Add impact points and blast radii
                    for i in range(fragment_count):
                        # Generate positions distributed across the visualization
                        position = (i - fragment_count/2) * (x_range / (fragment_count * 0.7))
                        
                        # Add impact point
                        fig.add_trace(go.Scatter(
                            x=[position], y=[0],
                            mode="markers",
                            marker=dict(
                                size=fragment_diameters[i] / 4,  # Scaled for visibility
                                color="brown",
                                symbol="circle"
                            ),
                            name=f"Fragment {i+1}",
                            showlegend=(i < 5)  # Show legend for first 5 only to avoid clutter
                        ))
                        
                        # Add blast radius circle (flattened for perspective)
                        theta = np.linspace(0, 2*np.pi, 100)
                        blast_x = position + blast_radii[i] * np.cos(theta) 
                        blast_y = blast_radii[i] * np.abs(np.sin(theta)) / 4  # Flattened circle
                        
                        fig.add_trace(go.Scatter(
                            x=blast_x, y=blast_y,
                            mode="lines",
                            line=dict(color="rgba(255,0,0,0.3)", width=1),
                            fill="toself",
                            fillcolor=f"rgba(255,0,0,{min(adjusted_energies_kt[i] / max(adjusted_energies_kt), 0.5):.2f})",
                            name=f"Blast {i+1}",
                            showlegend=False
                        ))
                        
                        # Add trajectory line
                        entry_x = position - (blast_radii[i] * 3) * np.cos(np.radians(entry_angle))
                        entry_y = (blast_radii[i] * 3) * np.sin(np.radians(entry_angle))
                        
                        fig.add_trace(go.Scatter(
                            x=[entry_x, position], y=[entry_y, 0],
                            mode="lines",
                            line=dict(color="gray", width=1, dash="dot"),
                            showlegend=False
                        ))
                    
                    # Improve layout
                    fig.update_layout(
                        title="Multiple Fragment Impacts",
                        xaxis_title="Distance (km)",
                        yaxis_title="Height (km)",
                        template="plotly_white",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add analysis
                    st.subheader("Multiple Impact Analysis")
                    
                    impact_severity = "low" if total_energy < 100 else "moderate" if total_energy < 10000 else "severe" if total_energy < 100000 else "catastrophic"
                    
                    if impact_severity == "low":
                        st.write("These impacts would cause localized damage with minimal regional effects. Most fragments would create small craters and limited blast damage.")
                    elif impact_severity == "moderate":
                        st.write("These impacts would cause significant regional damage. Multiple cities could be affected if impacts occur near populated areas, with potential for thousands of casualties.")
                    elif impact_severity == "severe":
                        st.write("These impacts represent a severe regional or continental catastrophe. Multiple large cities could be completely destroyed with millions of casualties possible.")
                    else:
                        st.write("These impacts represent a global catastrophe scenario. Worldwide climate effects would follow, including significant cooling and agricultural disruption for years.")
                    
                    # Emergency response
                    st.subheader("Emergency Response Planning")
                    st.markdown(f"""
                    - **Combined Affected Area**: ~{sum([np.pi * r**2 for r in blast_radii]):.2f} km²
                    - **Population Impact**: {"Localized" if impact_severity == "low" else "Regional" if impact_severity == "moderate" else "Continental" if impact_severity == "severe" else "Global"}
                    - **Evacuation Requirements**: {"Limited, tactical evacuation" if impact_severity == "low" else "Major regional evacuation" if impact_severity == "moderate" else "Mass evacuation of multiple regions" if impact_severity == "severe" else "Impossible - focus on sheltering"}
                    - **Long-term Effects**: {"Minimal" if impact_severity == "low" else "Months of recovery needed" if impact_severity == "moderate" else "Years of recovery needed" if impact_severity == "severe" else "Decades of global climate and agricultural disruption"}
                    """)
                    
                    # Add a downloadable report option
                    st.markdown("### Download Simulation Report")
                    st.info("PDF report generation would be available here in the production version.")
        
        if simulation_type == "Single Asteroid Impact":
            # Impact simulator inputs
            st.write("Simulate the impact of an asteroid with Earth")
        
        # Create two columns for input form
        sim_col1, sim_col2 = st.columns(2)
        
        with sim_col1:
            # Input for asteroid diameter
            asteroid_diameter = st.number_input(
                "Asteroid Diameter (meters)", 
                min_value=1.0, 
                max_value=10000.0, 
                value=50.0,
                step=10.0,
                key="single_asteroid_diameter"
            )
            
            # Input for asteroid velocity
            asteroid_velocity = st.number_input(
                "Impact Velocity (km/s)",
                min_value=11.0,  # Earth's escape velocity
                max_value=72.0,  # Max recorded NEO velocity 
                value=20.0,
                step=1.0,
                key="single_asteroid_velocity"
            )
        
        with sim_col2:
            # Input for asteroid density
            asteroid_density = st.number_input(
                "Asteroid Density (kg/m³)",
                min_value=1000.0,
                max_value=8000.0,
                value=3000.0,
                step=500.0,
                key="single_asteroid_density"
            )
            
            # Input for impact angle
            impact_angle = st.slider(
                "Impact Angle (degrees from horizontal)",
                min_value=0,
                max_value=90,
                value=45,
                key="single_impact_angle"
            )
        
        # Impact location (for future map integration)
        impact_location = st.selectbox(
            "Impact Location",
            ["Ocean", "Land (Urban)", "Land (Rural)", "Polar Region", "Custom"],
            key="single_impact_location"
        )
        
        # Run simulation button with custom styling
        st.markdown("""
        <style>
        .simulation-container {
            background-color: rgba(244, 67, 54, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            border-left: 4px solid #F44336;
            text-align: center;
        }
        </style>
        <div class="simulation-container">
            <p style="font-size:0.9em; margin-bottom:10px;">
            Simulate the consequences of an asteroid impact based on size, velocity, and impact parameters
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Run Impact Simulation", type="primary", key="run_single_impact_sim"):
            # Calculate impact energy (kilotons of TNT)
            # Formula based on kinetic energy: E = 0.5 * m * v^2
            # Converting to kilotons of TNT (4.184 terajoules per kiloton)
            
            # Calculate asteroid mass (kg)
            volume = (4/3) * np.pi * ((asteroid_diameter / 2) ** 3)
            mass = volume * asteroid_density
            
            # Convert velocity to m/s
            velocity_m_s = asteroid_velocity * 1000
            
            # Calculate energy in joules
            energy_joules = 0.5 * mass * (velocity_m_s ** 2)
            
            # Convert to kilotons (1 kt = 4.184e12 joules)
            energy_kt = energy_joules / 4.184e12
            
            # Adjust for impact angle (vertical impact transfers more energy)
            angle_factor = np.sin(np.radians(impact_angle))
            adjusted_energy_kt = energy_kt * angle_factor
            
            # Calculate crater diameter (km) - simplified formula based on kinetic energy scaling
            crater_diameter = 0.01 * (adjusted_energy_kt ** 0.3333)
            
            # Calculate fireball radius (km)
            fireball_radius = 0.002 * (adjusted_energy_kt ** 0.4)
            
            # Calculate thermal radiation radius (km) - 3rd degree burns
            thermal_radius = 0.01 * (adjusted_energy_kt ** 0.5)
            
            # Calculate air blast radius (km) - significant damage to buildings
            blast_radius = 0.03 * (adjusted_energy_kt ** 0.3333)
            
            # Calculate earthquake magnitude (Richter scale) - simplified estimation
            earthquake_magnitude = 0.67 * np.log10(adjusted_energy_kt) + 0.9
            
            # Display results in expander section
            with st.expander("Impact Simulation Results", expanded=True):
                st.markdown(f"### Impact Energy: **{adjusted_energy_kt:.2f} kilotons** of TNT")
                
                # Create comparison to known explosions
                if adjusted_energy_kt < 20:
                    st.write(f"Equivalent to: **{adjusted_energy_kt/0.02:.1f}× Hiroshima atomic bomb** (20 kt)")
                elif adjusted_energy_kt < 15000:
                    st.write(f"Equivalent to: **{adjusted_energy_kt/15000:.2f}× Tsar Bomba** (largest nuclear device ever detonated, 15 Mt)")
                else:
                    st.write(f"Equivalent to: **{adjusted_energy_kt/100000:.2f}× Chicxulub impactor** (dinosaur extinction event, ~100 Mt)")
                
                # Create columns for displaying multiple metrics
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric("Crater Diameter", f"{crater_diameter:.2f} km")
                    st.metric("Fireball Radius", f"{fireball_radius:.2f} km")
                    st.metric("Thermal Radiation Radius", f"{thermal_radius:.2f} km")
                
                with res_col2:
                    st.metric("Air Blast Radius (Building Damage)", f"{blast_radius:.2f} km")
                    st.metric("Earthquake Magnitude", f"{earthquake_magnitude:.1f} Richter")
                    st.metric("Affected Area", f"{np.pi * (blast_radius ** 2):.2f} km²")
                
                # Visualization of impact
                st.subheader("Impact Visualization")
                
                # Create visualization using plotly
                fig = go.Figure()
                
                # Add Earth's surface as a line
                fig.add_trace(go.Scatter(
                    x=np.linspace(-blast_radius*1.5, blast_radius*1.5, 100),
                    y=[0] * 100,
                    mode="lines",
                    line=dict(color="green", width=2),
                    name="Earth's Surface"
                ))
                
                # Add crater
                theta = np.linspace(0, np.pi, 100)
                crater_x = crater_diameter/2 * np.cos(theta)
                crater_y = -crater_diameter/2 * np.sin(theta)/3  # Flattened ellipse for crater
                
                fig.add_trace(go.Scatter(
                    x=crater_x,
                    y=crater_y,
                    mode="lines",
                    line=dict(color="brown", width=2),
                    fill="toself",
                    fillcolor="rgba(139, 69, 19, 0.3)",
                    name="Crater"
                ))
                
                # Add concentric circles for different effect radii
                for radius, name, color in [
                    (fireball_radius, "Fireball", "rgba(255, 165, 0, 0.5)"),
                    (thermal_radius, "Thermal Radiation", "rgba(255, 0, 0, 0.3)"),
                    (blast_radius, "Air Blast", "rgba(128, 128, 128, 0.2)")
                ]:
                    theta = np.linspace(0, 2*np.pi, 100)
                    x = radius * np.cos(theta)
                    y = radius * abs(np.sin(theta)) / 4  # Flatten to show perspective
                    
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(color=color.replace("0.5", "1").replace("0.3", "1").replace("0.2", "1"), width=1),
                        fill="toself",
                        fillcolor=color,
                        name=name
                    ))
                
                # Improve layout
                fig.update_layout(
                    title="Cross-section View of Impact Effects",
                    xaxis_title="Distance from Impact (km)",
                    yaxis_title="Depth/Height (km)",
                    legend=dict(x=0, y=1),
                    template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Impact outcome description
                st.subheader("Impact Consequences")
                
                if adjusted_energy_kt < 10:
                    st.write("This impact would likely cause minimal damage, with most of the asteroid burning up in the atmosphere. It might create a small crater and local damage.")
                elif adjusted_energy_kt < 1000:
                    st.write("This impact would cause significant regional damage, with a substantial crater and destruction in the immediate area. Buildings would be damaged or destroyed within the air blast radius.")
                elif adjusted_energy_kt < 100000:
                    st.write("This impact would cause severe regional devastation, with extensive destruction within hundreds of kilometers. It would trigger significant earthquakes, tsunamis (if over water), and potential climate effects.")
                else:
                    st.write("This impact would cause global catastrophe, with immediate devastation across an entire continent. Long-term climate effects including global cooling would threaten agriculture worldwide.")
                
                # Tips for minimizing casualties
                st.subheader("Emergency Response Recommendations")
                st.markdown("""
                - **Evacuation Radius**: At minimum, evacuate all areas within the air blast radius
                - **Emergency Services**: Position beyond the thermal radiation radius
                - **Medical Response**: Prepare for burn injuries, crush injuries, and respiratory issues
                - **Long-term Planning**: Consider environmental and climate impacts for larger impacts
                """)
    
    with tab5:
        st.header("Risk Assessment & Analysis")
        
        if 'best_model' in st.session_state:
            # Add a styled button to make predictions
            st.markdown("""
            <style>
            .hazard-prediction-container {
                background-color: rgba(255, 193, 7, 0.1);
                border-radius: 10px;
                padding: 15px;
                margin: 20px 0;
                border-left: 4px solid #FFC107;
                text-align: center;
            }
            </style>
            <div class="hazard-prediction-container">
                <p style="font-size:0.9em; margin-bottom:10px;">
                Analyze asteroid properties to predict potential hazards using the trained machine learning model
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            make_predictions = st.button("Make Hazard Predictions", type="primary", key="make_hazard_predictions")
            
            # Make predictions when button is clicked or if predictions already made
            if make_predictions or 'predictions_made' in st.session_state:
                with st.spinner("Making predictions..."):
                    # Use the same feature names that were used during model training
                    # Include all engineered features to match training data
                    feature_names = [
                        "absolute_magnitude", "estimated_diameter_min", "estimated_diameter_max", 
                        "relative_velocity", "miss_distance", "orbital_stability_index", "diameter_diff"
                    ]
                    
                    # Add additional engineered features if present
                    additional_features = [
                        "diameter_velocity_ratio", "energy_proxy", "proximity_risk", "size_uncertainty"
                    ]
                    
                    # Add only features that exist in the dataframe
                    for feature in additional_features:
                        if feature in df.columns:
                            feature_names.append(feature)
                        else:
                            st.warning(f"Missing feature: {feature}. This may affect prediction accuracy.")
                    
                    # Verify all required features exist
                    missing_required = [f for f in feature_names if f not in df.columns]
                    if missing_required:
                        st.error(f"Missing required features for prediction: {missing_required}")
                        st.info("Please go back to the Data Processing tab and ensure feature engineering is applied.")
                    else:
                        try:
                            X = df[feature_names]
                            
                            # Make predictions
                            model = st.session_state['best_model']
                            
                            # Debug information
                            st.write(f"Making predictions with model: {type(model).__name__}")
                            st.write(f"Input shape: {X.shape}")
                            
                            # Make predictions and convert to proper type for display and calculation
                            preds = model.predict(X)
                            df['predicted_hazard'] = preds
                            
                            # Show some statistics about predictions
                            positive_count = np.sum(preds == True)
                            st.write(f"Predicted hazardous asteroids: {positive_count} out of {len(preds)}")
                            
                            # Get prediction probabilities if available
                            if hasattr(model, "predict_proba"):
                                try:
                                    proba = model.predict_proba(X)
                                    if proba.shape[1] > 1:  # Binary classification
                                        df['hazard_probability'] = proba[:, 1]
                                    else:
                                        df['hazard_probability'] = proba[:, 0]
                                except (IndexError, ValueError) as e:
                                    st.warning(f"Could not get prediction probabilities: {e}")
                                    df['hazard_probability'] = 0.5
                            else:
                                df['hazard_probability'] = 0.5
                            
                            # Update session state
                            st.session_state['neo_df'] = df.copy()
                            st.session_state['predictions_made'] = True
                            
                            # Confirm predictions complete
                            st.success("✅ Predictions made successfully!")
                        except Exception as e:
                            st.error(f"Error making predictions: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            
            # Display prediction results
            st.subheader("Hazardous Asteroid Predictions")
            
            # Check if predictions have been made
            if 'predicted_hazard' in df.columns:
                # Prediction statistics
                hazard_count = df['predicted_hazard'].sum()
                total_count = len(df)
                hazard_percent = (hazard_count / total_count) * 100 if total_count > 0 else 0
                
                st.write(f"Predicted hazardous asteroids: {hazard_count} out of {total_count} ({hazard_percent:.2f}%)")
                
                # Visualize top hazardous asteroids
                st.subheader("Top Hazardous Asteroids")
                
                # Show only if there are hazardous asteroids
                if hazard_count > 0:
                    try:
                        hazard_fig = visualize_top_3_hazardous_asteroids(df)
                        st.plotly_chart(hazard_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error rendering hazardous asteroid visualization: {str(e)}")
                    
                    # Table of hazardous asteroids
                    st.subheader("Hazardous Asteroid Details")
                    hazardous_df = df[df['predicted_hazard'] == True].sort_values(by='hazard_probability', ascending=False)
                    
                    if not hazardous_df.empty:
                        display_cols = [
                            'name', 'close_approach_date', 'miss_distance', 'relative_velocity',
                            'estimated_diameter_max', 'hazard_probability'
                        ]
                        st.dataframe(hazardous_df[display_cols])
                    else:
                        st.info("No hazardous asteroids detected in this dataset.")
                else:
                    st.info("No hazardous asteroids detected in this dataset.")
            else:
                st.info("Please train a model first and make predictions to see hazardous asteroid analysis.")
        else:
            st.info("Please train a model first to see analysis results.")

# Handle export button click
if export_button and 'neo_df' in st.session_state:
    df = st.session_state['neo_df']
    
    try:
        # Export based on selected format and content
        with st.spinner(f"Exporting {export_content} as {export_format}..."):
            filename = export_data(df, format=export_format, content=export_content)
            
            # If visualization export is selected and there are visualizations in session state
            if export_content in ["Visualization", "All"] and 'best_model' in st.session_state:
                try:
                    # Create visualizations with error handling
                    figs = {}
                    
                    # Try to create each visualization
                    try:
                        scatter_fig = plot_asteroids(df, "is_potentially_hazardous", "Asteroid Distribution", True)
                        if scatter_fig is not None:
                            figs["scatter"] = scatter_fig
                    except Exception as e:
                        st.warning(f"Could not create scatter plot for export: {str(e)}")
                    
                    try:
                        viz_3d = create_3d_asteroid_visualization(df)
                        if viz_3d is not None:
                            figs["3d"] = viz_3d
                    except Exception as e:
                        st.warning(f"Could not create 3D visualization for export: {str(e)}")
                    
                    try:
                        # Use the static version for export as it's more reliable
                        traj_fig = create_interactive_asteroid_paths(df)
                        if traj_fig is not None:
                            figs["trajectory"] = traj_fig
                    except Exception as e:
                        st.warning(f"Could not create asteroid trajectory visualization for export: {str(e)}")
                    
                    # Export visualizations if we have any
                    if figs:
                        viz_filename = export_visualization(format=export_format, figs=figs)
                        st.success(f"✅ Successfully exported data to {filename} and visualizations to {viz_filename}!")
                    else:
                        st.warning("No visualizations could be exported due to errors.")
                        st.success(f"✅ Successfully exported data to {filename}!")
                        
                except Exception as e:
                    st.error(f"Failed to export visualizations: {str(e)}")
                    st.success(f"✅ Successfully exported data to {filename}!")
            else:
                st.success(f"✅ Successfully exported to {filename}!")
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("🚀 **NEO Asteroid Analysis Dashboard** | Data from NASA API")
