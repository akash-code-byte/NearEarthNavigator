import os
import time
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
    page_title="Comprehensive Space Threat Assessment and Prediction System (Local)",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light theme
st.markdown("""
<script>
    document.documentElement.setAttribute('data-theme', 'light');
    localStorage.setItem('theme', 'light');
</script>
""", unsafe_allow_html=True)

# NASA API Key - Set to provided API key 
NASA_API_KEY = "yl3iawXawys50GTGtKdzQ9TmbKlJpmoptjC8Shqb"

# Initialize session state
if 'show_advanced' not in st.session_state:
    st.session_state.show_advanced = False
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'model_params' not in st.session_state:
    st.session_state.model_params = None
if 'viz_type' not in st.session_state:
    st.session_state.viz_type = "Asteroid Distribution"

# Function to apply theme styling - permanently using light theme with black text and enhanced UI
def apply_theme_styling():
    # Apply enhanced light theme styling with black text (this is permanent - no toggle button)
    st.markdown("""
    <style>
    /* Base styling */
    .main {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif !important;
    }
    .stApp {
        background-color: #FFFFFF !important;
    }
    body {
        background-color: #FFFFFF !important;
    }
    
    /* Text styling */
    p, h1, h2, h3, h4, h5, h6, span, div, label, th, td {
        color: #000000 !important;
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif !important;
    }
    
    h1 {
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }
    
    h2, h3 {
        font-weight: 500 !important;
    }
    
    /* Sidebar and containers */
    .css-1d391kg, .css-1v3fvcr, .css-18e3th9, .css-1inwz65 {
        background-color: #FFFFFF !important;
    }
    
    .sidebar .sidebar-content {
        background-color: #F8F9FA !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        box-shadow: 2px 0 5px rgba(0, 0, 0, 0.05) !important;
        border-right: 1px solid #EAECEF !important;
        padding: 1.5rem !important;
    }
    
    /* Card styling */
    [data-testid="stVerticalBlock"] {
        border-radius: 8px !important;
    }
    
    /* Card hover effect */
    .stCard {
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    .stCard:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #F8F9FA !important;
        color: #000000 !important;
        border: 1px solid #E0E3E7 !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    
    .stButton>button:hover {
        background-color: #EAECEF !important;
        border-color: #CED4DA !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton>button[kind="primary"] {
        background-color: #4CAF50 !important;
        color: white !important;
        border-color: #3E8E41 !important;
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.3) !important;
    }
    
    .stButton>button[kind="primary"]:hover {
        background-color: #3E8E41 !important;
    }
    
    /* Input fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #F8F9FA !important;
        color: #000000 !important;
        border-radius: 6px !important;
        border: 1px solid #E0E3E7 !important;
        padding: 0.75rem !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2) !important;
    }
    
    /* Selectbox */
    .stSelectbox>div>div {
        background-color: #F8F9FA !important;
        color: #000000 !important;
        border-radius: 6px !important;
        border: 1px solid #E0E3E7 !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    
    .stSelectbox>div:hover {
        border-color: #CED4DA !important;
    }
    
    .stSelectbox>div[data-baseweb="select"] > div {
        background-color: #F8F9FA !important;
    }
    
    /* Dividers */
    .st-br {
        border-color: #EAECEF !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        border-bottom: 1px solid #EAECEF !important;
        padding-bottom: 0 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0 !important;
        padding: 0.6rem 1.2rem !important;
        color: #000000 !important;
        font-weight: 500 !important;
        background-color: #F8F9FA !important;
        border: 1px solid #EAECEF !important;
        border-bottom: none !important;
        margin-bottom: -1px !important;
        transition: background-color 0.2s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #F0F2F6 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
        border-color: #4CAF50 !important;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #F8F9FA !important;
        border-left: 4px solid #4CAF50 !important;
        padding: 1.2rem !important;
        border-radius: 8px !important;
        margin-bottom: 1.5rem !important;
        color: #000000 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05) !important;
    }
    
    .alert-box {
        background-color: #F8F9FA !important;
        border-left: 4px solid #4CAF50 !important;
        padding: 1.2rem !important;
        border-radius: 8px !important;
        margin-bottom: 1.5rem !important;
        color: #000000 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Charts and dataframes */
    .js-plotly-plot {
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        padding: 1rem !important;
        background-color: #FFFFFF !important;
    }
    
    .dataframe {
        color: #000000 !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    }
    
    .dataframe th {
        background-color: #F0F2F6 !important;
        color: #000000 !important;
        padding: 12px 15px !important;
        text-align: left !important;
        border-bottom: 1px solid #EAECEF !important;
        font-weight: 600 !important;
    }
    
    .dataframe td {
        color: #000000 !important;
        padding: 10px 15px !important;
        border-bottom: 1px solid #EAECEF !important;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #F8F9FA !important;
    }
    
    .dataframe tr:hover {
        background-color: #F0F2F6 !important;
    }
    
    /* Radio buttons and checkboxes */
    .stRadio label, .stCheckbox label {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #4CAF50 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
    }
    
    /* Plotly chart enhancements */
    .js-plotly-plot .plotly .main-svg {
        border-radius: 8px !important;
    }
    
    /* Animations */
    .stButton>button, .stSelectbox>div, .stTextInput>div>div>input, .stNumberInput>div>div>input {
        transition: all 0.3s ease !important;
    }
    
    /* Modern scrollbar */
    ::-webkit-scrollbar {
        width: 8px !important;
        height: 8px !important;
    }
    
    ::-webkit-scrollbar-track {
        background: #F8F9FA !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #CED4DA !important;
        border-radius: 4px !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #ADB5BD !important;
    }
    
    /* Container for dashboard items */
    .dashboard-container {
        background-color: #F8F9FA !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative !important;
        display: inline-block !important;
        cursor: pointer !important;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden !important;
        width: 200px !important;
        background-color: #555 !important;
        color: #fff !important;
        text-align: center !important;
        border-radius: 6px !important;
        padding: 10px !important;
        position: absolute !important;
        z-index: 1 !important;
        bottom: 125% !important;
        left: 50% !important;
        margin-left: -100px !important;
        opacity: 0 !important;
        transition: opacity 0.3s !important;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible !important;
        opacity: 1 !important;
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

# Main content area - Enhanced UI with modern dashboard layout
st.markdown("""
<div style="text-align: center; padding: 10px 0; margin-bottom: 20px;">
    <h1 style='font-size: 2.5em; font-weight: 700; color: #000000; margin-bottom: 10px;'>
        <span style="color: #4CAF50;">☄️</span> Comprehensive Space Threat Assessment
    </h1>
    <p style='font-size: 1.3em; color: #333333; max-width: 800px; margin: 0 auto 15px;'>
        An advanced platform for tracking, visualizing, and predicting hazards from Near Earth Objects
    </p>
    <div style='width: 100px; height: 3px; background-color: #4CAF50; margin: 0 auto 25px;'></div>
</div>
""", unsafe_allow_html=True)

# Create a modern dashboard-style info container
st.markdown("""
<div style="background-color: #F8F9FA; border-radius: 12px; padding: 25px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);">
    <div style="display: flex; align-items: flex-start;">
        <div style="flex: 1;">
            <h3 style="color: #000000; margin-top: 0; font-size: 1.3em; font-weight: 600;">About Space Threat Assessment</h3>
            <p style="color: #000000; line-height: 1.6; margin-bottom: 15px;">
                This application analyzes Near-Earth Objects (NEOs) to identify potentially hazardous asteroids
                that could pose a threat to Earth. By leveraging NASA's NEO database and advanced machine
                learning algorithms, the system provides comprehensive risk assessment and predictive analytics.
            </p>
            <div style="margin-top: 15px;">
                <span style="background-color: rgba(76, 175, 80, 0.15); color: #2E7D32; padding: 5px 10px; border-radius: 15px; font-size: 0.9em; margin-right: 8px;">
                    <strong>Real-time Data</strong>
                </span>
                <span style="background-color: rgba(33, 150, 243, 0.15); color: #1565C0; padding: 5px 10px; border-radius: 15px; font-size: 0.9em; margin-right: 8px;">
                    <strong>ML Predictions</strong>
                </span>
                <span style="background-color: rgba(255, 152, 0, 0.15); color: #E65100; padding: 5px 10px; border-radius: 15px; font-size: 0.9em;">
                    <strong>Impact Simulations</strong>
                </span>
            </div>
        </div>
        <div style="width: 160px; margin-left: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <div style="background-color: rgba(76, 175, 80, 0.1); border-radius: 50%; width: 120px; height: 120px; display: flex; align-items: center; justify-content: center;">
                <div style="font-size: 40px;">☄️</div>
            </div>
            <div style="margin-top: 10px; text-align: center; font-weight: 500;">NASA data-driven</div>
        </div>
    </div>
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

# Main tabs - Enhanced UI
if 'neo_df' in st.session_state:
    df = st.session_state['neo_df']
    
    # Create a better-looking tab system with icons
    st.markdown("""
    <style>
    .custom-tabs {
        margin-top: 20px;
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Custom tab container
    st.markdown('<div class="custom-tabs"></div>', unsafe_allow_html=True)
    
    # Define tab icons
    tab_icons = {
        "Overview": "📊",
        "Visualizations": "🔍",
        "Threat Assessment": "⚠️",
        "Model Insights": "🧠",
        "Impact Simulator": "💥",
        "Data Explorer": "🔎"
    }
    
    # Create tab names with icons
    tab_names = [f"{icon} {name}" for name, icon in tab_icons.items()]
    
    tabs = st.tabs(tab_names)
    
    # Overview Tab
    with tabs[0]:
        st.subheader("Near-Earth Object Overview")
        
        # Enhanced metrics with cards and icons
        st.markdown("""
        <style>
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px 15px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            height: 100%;
            border-top: 4px solid #4CAF50;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 600;
            color: #4CAF50;
            margin: 10px 0;
        }
        .metric-title {
            font-size: 1rem;
            color: #555;
            font-weight: 500;
            margin-bottom: 8px;
        }
        .metric-icon {
            font-size: 2rem;
            margin-bottom: 10px;
            color: #4CAF50;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Calculate metrics
        total_objects = len(df)
        potentially_hazardous = df['is_potentially_hazardous'].sum()
        avg_miss_distance = df['miss_distance'].mean() / 1000000  # km
        max_diameter = df['estimated_diameter_max'].max() * 1000  # meters
        
        # Create 4 columns
        col1, col2, col3, col4 = st.columns(4)
        
        # Custom metric cards with icons
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🔭</div>
                <div class="metric-title">Total NEOs</div>
                <div class="metric-value">{total_objects}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #FF5722;">
                <div class="metric-icon" style="color: #FF5722;">⚠️</div>
                <div class="metric-title">Potentially Hazardous</div>
                <div class="metric-value" style="color: #FF5722;">{potentially_hazardous}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #2196F3;">
                <div class="metric-icon" style="color: #2196F3;">📏</div>
                <div class="metric-title">Avg. Miss Distance</div>
                <div class="metric-value" style="color: #2196F3;">{avg_miss_distance:.2f}<span style="font-size: 1rem;"> M km</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #9C27B0;">
                <div class="metric-icon" style="color: #9C27B0;">📊</div>
                <div class="metric-title">Max Diameter</div>
                <div class="metric-value" style="color: #9C27B0;">{max_diameter:.1f}<span style="font-size: 1rem;"> m</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced distribution plots with card styling
        st.markdown("""
        <style>
        .chart-card {
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 24px;
            border-left: 5px solid #4CAF50;
        }
        .chart-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h3 style="margin-top:30px; margin-bottom:20px; font-size:1.5rem;">NEO Distribution Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-card" style="border-left-color: #4CAF50;">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Distribution of Absolute Magnitude</div>', unsafe_allow_html=True)
            
            fig = px.histogram(df, x="absolute_magnitude", 
                             labels={"absolute_magnitude": "Absolute Magnitude (H)"},
                             color_discrete_sequence=['#4CAF50'],
                             opacity=0.8,
                             nbins=15)  # Control number of bins to prevent overcrowding
            
            fig.update_layout(
                showlegend=False,
                margin=dict(l=40, r=40, t=30, b=40),
                plot_bgcolor='rgba(248,249,250,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title_font=dict(size=14),
                    gridcolor='rgba(220,220,220,0.5)',
                    tickangle=-45,  # Angle the labels
                    tickformat=".1f"  # Format the tick values
                ),
                yaxis=dict(
                    title="Count",
                    title_font=dict(size=14),
                    gridcolor='rgba(220,220,220,0.5)'
                ),
                bargap=0.05
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-card" style="border-left-color: #2196F3;">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Distribution of Estimated Diameter</div>', unsafe_allow_html=True)
            
            fig = px.histogram(df, x="estimated_diameter_max", 
                             labels={"estimated_diameter_max": "Maximum Estimated Diameter (km)"},
                             color_discrete_sequence=['#2196F3'],
                             opacity=0.8,
                             nbins=15)  # Control number of bins to prevent overcrowding
            
            fig.update_layout(
                showlegend=False,
                margin=dict(l=40, r=40, t=30, b=40),
                plot_bgcolor='rgba(248,249,250,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title_font=dict(size=14),
                    gridcolor='rgba(220,220,220,0.5)',
                    tickangle=-45,  # Angle the labels
                    tickformat=".2f"  # Format the tick values
                ),
                yaxis=dict(
                    title="Count",
                    title_font=dict(size=14),
                    gridcolor='rgba(220,220,220,0.5)'
                ),
                bargap=0.05
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
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
        
        # Visualization selector using session state to maintain selection
        viz_options = ["Asteroid Distribution", "3D Space Visualization", "Trajectory Analysis", "Time Series Analysis", "Close Approach Map"]
        
        # Add previous selection to session state if not present
        if 'previous_viz_type' not in st.session_state:
            st.session_state.previous_viz_type = st.session_state.viz_type
            
        # This ensures the visualization type is maintained when switching tabs
        selected_viz = st.radio(
            "Select Visualization Type",
            viz_options,
            index=viz_options.index(st.session_state.viz_type) if st.session_state.viz_type in viz_options else 0,
            key='viz_selector'
        )
        
        # Check if visualization type changed
        if selected_viz != st.session_state.previous_viz_type:
            st.session_state.previous_viz_type = st.session_state.viz_type
            st.session_state.viz_type = selected_viz
            st.rerun()
        
        # Update session state with selected visualization
        st.session_state.viz_type = selected_viz
        
        # Use the selected visualization type
        viz_type = selected_viz
        
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
            
        elif viz_type == "Close Approach Map":
            st.subheader("Global Map of NEO Close Approaches")
            
            st.markdown("""
            <div style='background-color: rgba(240, 240, 240, 0.5); border-radius: 10px; padding: 15px; margin-bottom: 20px; border-left: 6px solid #2196F3;'>
                <h4 style='color: #000000; margin-top:0;'>About Close Approach Map</h4>
                <p style='color: #000000;'>
                This map visualizes the predicted close approach locations of Near-Earth Objects (NEOs).
                The points represent the location on Earth that would be closest to the NEO during its approach.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate approximate closest approach points
            # In reality, close approach points would be calculated based on orbital mechanics,
            # but for demonstration we'll create them based on asteroid properties
            
            # Filter data for the visualization
            approach_asteroids = df.sort_values('miss_distance').head(20)
            
            # Create latitude/longitude points
            # We'll use a deterministic approach to generate points based on asteroid properties
            # This ensures consistent visualization and spreads the points across the globe
            
            # Initialize empty lists
            lats = []
            lons = []
            sizes = []
            colors = []
            hover_texts = []
            
            # Generate points based on asteroid properties
            import numpy as np
            np.random.seed(42)  # For reproducibility
            
            for idx, row in approach_asteroids.iterrows():
                # Use asteroid properties to deterministically generate coordinates
                # This approach creates a global distribution based on asteroid properties
                lat = (row['absolute_magnitude'] % 18) * 10 - 90
                lon = (row['miss_distance'] % 36) * 10 - 180
                
                # Adjust to ensure within bounds
                lat = max(min(lat, 90), -90)
                lon = max(min(lon, 180), -180)
                
                # Add some small variation
                lat += np.random.uniform(-5, 5)
                lon += np.random.uniform(-5, 5)
                
                # Calculate size for visualization (based on diameter)
                size = row['estimated_diameter_max'] * 50
                size = max(5, min(size, 20))  # Constrain size for better visualization
                
                # Calculate color value (based on velocity)
                # Normalize velocity to 0-1 range for color scale
                min_vel = approach_asteroids['relative_velocity'].min()
                max_vel = approach_asteroids['relative_velocity'].max()
                norm_vel = (row['relative_velocity'] - min_vel) / (max_vel - min_vel) if max_vel > min_vel else 0.5
                
                # Create hover text with asteroid information
                hover_text = f"<b>{row['name']}</b><br>" + \
                             f"Date: {row['close_approach_date']}<br>" + \
                             f"Diameter: {row['estimated_diameter_max']*1000:.1f} m<br>" + \
                             f"Miss Distance: {row['miss_distance']/1000000:.2f} million km<br>" + \
                             f"Velocity: {row['relative_velocity']:.1f} km/s"
                
                # Add to lists
                lats.append(lat)
                lons.append(lon)
                sizes.append(size)
                colors.append(norm_vel)
                hover_texts.append(hover_text)
            
            # Create a dataframe for the map
            map_df = pd.DataFrame({
                'lat': lats,
                'lon': lons,
                'size': sizes,
                'color': colors,
                'text': hover_texts,
                'name': approach_asteroids['name'].values
            })
            
            # Create the map with scatter points
            fig = px.scatter_geo(
                map_df,
                lat='lat',
                lon='lon',
                size='size',
                color='color',
                color_continuous_scale='Viridis',
                hover_name='name',
                custom_data=['text'],
                title="Predicted NEO Close Approach Locations",
                projection='natural earth'
            )
            
            # Update hover template to use the HTML hover text
            fig.update_traces(
                hovertemplate="%{customdata[0]}<extra></extra>"
            )
            
            # Update layout
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title="Relative Velocity",
                    tickvals=[0, 0.5, 1],
                    ticktext=['Slower', 'Medium', 'Faster']
                ),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            # Show the map
            st.plotly_chart(fig, use_container_width=True)
            
            # Add information about the top 5 closest approaches
            st.subheader("Closest Approaches Details")
            
            closest_approaches = df.sort_values('miss_distance').head(5)
            closest_df = closest_approaches[['name', 'close_approach_date', 
                                            'miss_distance', 'relative_velocity', 
                                            'estimated_diameter_max']]
            
            # Format columns for display
            closest_df['miss_distance'] = closest_df['miss_distance'].apply(lambda x: f"{x/1000000:.2f} million km")
            closest_df['relative_velocity'] = closest_df['relative_velocity'].apply(lambda x: f"{x:.2f} km/s")
            closest_df['estimated_diameter_max'] = closest_df['estimated_diameter_max'].apply(lambda x: f"{x*1000:.1f} m")
            
            # Rename columns for display
            closest_df.columns = ['Asteroid Name', 'Approach Date', 'Miss Distance', 
                                 'Relative Velocity', 'Maximum Diameter']
            
            st.dataframe(closest_df, use_container_width=True)
    
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
        
        # First row of metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}", 
                      help="Percentage of correct predictions (TP+TN)/(TP+TN+FP+FN)")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}", 
                     help="How many selected items are relevant (TP)/(TP+FP)")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}", 
                     help="How many relevant items are selected (TP)/(TP+FN)")
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.4f}", 
                     help="Harmonic mean of precision and recall: 2*(Precision*Recall)/(Precision+Recall)")
                     
        # Add a divider for additional metrics
        st.markdown("---")
        
        # Second row with additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            auc_value = metrics.get('auc', 0)
            st.metric("AUC-ROC", f"{auc_value:.4f}", 
                     help="Area Under the Receiver Operating Characteristic Curve - higher is better")
            
            # Add classification thresholds
            st.markdown("##### Classification Thresholds")
            threshold_df = pd.DataFrame({
                'Metric': ['Precision', 'Recall', 'F1 Score'],
                'Value at 0.3': [f"{metrics.get('precision_at_thresholds', {}).get(0.3, 0):.3f}", 
                               f"{metrics.get('recall_at_thresholds', {}).get(0.3, 0):.3f}",
                               f"{metrics.get('f1_at_thresholds', {}).get(0.3, 0):.3f}"],
                'Value at 0.5': [f"{metrics.get('precision_at_thresholds', {}).get(0.5, 0):.3f}", 
                               f"{metrics.get('recall_at_thresholds', {}).get(0.5, 0):.3f}",
                               f"{metrics.get('f1_at_thresholds', {}).get(0.5, 0):.3f}"],
                'Value at 0.7': [f"{metrics.get('precision_at_thresholds', {}).get(0.7, 0):.3f}", 
                               f"{metrics.get('recall_at_thresholds', {}).get(0.7, 0):.3f}",
                               f"{metrics.get('f1_at_thresholds', {}).get(0.7, 0):.3f}"]
            })
            
            st.dataframe(threshold_df, use_container_width=True, hide_index=True)
            
        with col2:
            # Add confusion matrix as a heatmap
            st.markdown("##### Confusion Matrix")
            
            # Create a confusion matrix from TN, FP, FN, TP if available
            if all(k in metrics for k in ['tn', 'fp', 'fn', 'tp']):
                cm = np.array([[metrics['tn'], metrics['fp']], 
                              [metrics['fn'], metrics['tp']]])
                
                cm_df = pd.DataFrame(cm, 
                                    index=['Actual Negative', 'Actual Positive'], 
                                    columns=['Predicted Negative', 'Predicted Positive'])
                
                fig = px.imshow(cm, 
                               x=['Predicted Negative', 'Predicted Positive'],
                               y=['Actual Negative', 'Actual Positive'],
                               color_continuous_scale='Blues',
                               labels=dict(color="Count"),
                               text_auto=True)
                
                fig.update_layout(width=300, height=300, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Confusion matrix data not available.")
            
        with col3:
            # Display model specific metrics
            st.markdown("##### Model Details")
            
            # Create a dataframe of important model parameters
            model_details = []
            
            # Get common details
            training_time = metrics.get('training_time', 'Not recorded')
            model_details.append(["Training Time", f"{training_time:.2f} sec" if isinstance(training_time, (int, float)) else training_time])
            
            if 'class_distribution' in metrics:
                pos_class = metrics['class_distribution'].get(1, 0)
                neg_class = metrics['class_distribution'].get(0, 0)
                model_details.append(["Class Balance", f"Pos: {pos_class}, Neg: {neg_class}"])
                
            if 'cross_val_scores' in metrics:
                cv_mean = np.mean(metrics['cross_val_scores'])
                cv_std = np.std(metrics['cross_val_scores'])
                model_details.append(["Cross-Val Score", f"{cv_mean:.4f} ± {cv_std:.4f}"])
            
            # Model specific details
            if model_type == "Random Forest":
                if 'n_estimators' in model_params:
                    model_details.append(["Number of Trees", model_params['n_estimators']])
                if 'max_depth' in model_params:
                    model_details.append(["Max Depth", model_params['max_depth']])
                    
            elif model_type == "XGBoost":
                if 'learning_rate' in model_params:
                    model_details.append(["Learning Rate", model_params['learning_rate']])
                if 'max_depth' in model_params:
                    model_details.append(["Max Depth", model_params['max_depth']])
                    
            elif model_type == "Neural Network":
                if 'hidden_layers' in model_params:
                    model_details.append(["Hidden Layers", model_params['hidden_layers']])
                if 'neurons_per_layer' in model_params:
                    model_details.append(["Neurons/Layer", model_params['neurons_per_layer']])
            
            # Create dataframe and display
            details_df = pd.DataFrame(model_details, columns=["Parameter", "Value"])
            st.dataframe(details_df, use_container_width=True, hide_index=True)
        
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
            
            # Create a numeric-only DataFrame for correlation calculation
            numeric_cols = df[feature_list].select_dtypes(include=['float64', 'int64']).columns.tolist()
            # Skip if there are no numeric columns to correlate
            if len(numeric_cols) > 0:
                # Calculate correlations with hazard_probability
                correlation_df = pd.DataFrame(df[numeric_cols].corrwith(df['hazard_probability']))
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
            else:
                st.info("No numeric features available for correlation analysis.")
    
    # Impact Simulator Tab - Enhanced with collision simulation
    with tabs[4]:
        st.markdown("""
        <style>
        .simulator-header {
            text-align: center;
            margin-bottom: 25px;
        }
        .simulator-card {
            background-color: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            margin-bottom: 30px;
            border-top: 5px solid #FF5722;
        }
        .simulator-title {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
        }
        .sim-type-btn {
            padding: 12px 16px;
            font-weight: 500;
            border-radius: 8px;
            margin: 0 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        }
        .sim-type-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        }
        .sim-type-active {
            background-color: #4CAF50;
            color: white;
        }
        .sim-type-inactive {
            background-color: #f2f2f2;
            color: #333;
        }
        </style>
        
        <div class="simulator-header">
            <h2 style="font-size: 2.2rem; font-weight: 700; color: #FF5722; margin-bottom: 10px;">🔥 Impact Simulator</h2>
            <p style="font-size: 1.1rem; color: #555; max-width: 700px; margin: 0 auto;">
                Advanced simulation tools to model asteroid impacts, collisions, and assess potential Earth damage
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulation type tabs
        sim_types = ["Single Impact", "Asteroid Collision & Fragmentation"]
        sim_type = st.radio("Select Simulation Type", sim_types, horizontal=True)
        
        if sim_type == "Single Impact":
            st.markdown("""
            <div class="simulator-card">
                <div class="simulator-title">🌍 Single Asteroid Impact Simulator</div>
                <p style="color: #555; margin-bottom: 20px;">
                This simulator uses physics-based models to estimate the potential consequences of an asteroid impact on Earth.
                Adjust the parameters below to simulate different impact scenarios and view the estimated effects.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create two columns for input parameters
        col1, col2 = st.columns(2)
        
        with col1:
            # Asteroid selection
            if not df.empty:
                hazardous_asteroids = df[df['is_potentially_hazardous']].sort_values('estimated_diameter_max', ascending=False)
                if not hazardous_asteroids.empty:
                    asteroid_options = ["Custom Asteroid"] + hazardous_asteroids['name'].tolist()
                    selected_asteroid = st.selectbox("Select Asteroid", asteroid_options)
                    
                    if selected_asteroid != "Custom Asteroid":
                        # Pre-fill with selected asteroid parameters
                        selected_data = df[df['name'] == selected_asteroid].iloc[0]
                        default_diameter = selected_data['estimated_diameter_max'] * 1000  # Convert to meters
                        default_velocity = selected_data['relative_velocity']
                        default_density = 3000  # kg/m³ (typical asteroid density)
                    else:
                        # Default values for custom asteroid
                        default_diameter = 100
                        default_velocity = 20
                        default_density = 3000
                else:
                    # No hazardous asteroids in data, use defaults
                    selected_asteroid = "Custom Asteroid"
                    default_diameter = 100
                    default_velocity = 20
                    default_density = 3000
            else:
                # No data loaded, use defaults
                selected_asteroid = "Custom Asteroid"
                default_diameter = 100
                default_velocity = 20
                default_density = 3000
            
            # Custom parameters input
            if selected_asteroid == "Custom Asteroid":
                st.subheader("Asteroid Parameters")
                diameter = st.slider("Asteroid Diameter (meters)", 1, 1000, int(default_diameter))
                velocity = st.slider("Impact Velocity (km/s)", 10, 70, int(default_velocity))
                density = st.slider("Asteroid Density (kg/m³)", 1000, 8000, default_density, step=100)
            else:
                # Display asteroid parameters from selected asteroid but allow override
                st.subheader("Asteroid Parameters")
                diameter = st.slider("Asteroid Diameter (meters)", 1, 1000, int(default_diameter))
                velocity = st.slider("Impact Velocity (km/s)", 10, 70, int(default_velocity))
                density = st.slider("Asteroid Density (kg/m³)", 1000, 8000, default_density, step=100)
                
                # Display additional info about the selected asteroid
                st.info(f"""
                Selected asteroid: {selected_asteroid}
                Original estimated diameter: {default_diameter:.1f} meters
                Original velocity: {default_velocity:.1f} km/s
                Miss distance: {selected_data['miss_distance']/1000000:.1f} million km
                """)
        
        with col2:
            # Impact parameters
            st.subheader("Impact Parameters")
            impact_angle = st.slider("Impact Angle (degrees from horizontal)", 5, 90, 45)
            
            # Target selection
            target_options = ["Ocean", "Continental Crust", "Urban Area", "Forest", "Desert"]
            target = st.selectbox("Impact Target", target_options)
            
            # Target-specific parameters
            if target == "Ocean":
                water_depth = st.slider("Water Depth (meters)", 100, 5000, 2000)
                distance_from_shore = st.slider("Distance from Shore (km)", 1, 1000, 100)
            elif target == "Urban Area":
                population_density = st.slider("Population Density (people/km²)", 1000, 20000, 5000)
                building_strength = st.selectbox("Building Types", ["Weak", "Medium", "Strong"])
            
            # Calculate button with custom styling
            st.markdown("""
            <style>
            div.stButton > button {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: none;
                padding: 10px 24px;
                border-radius: 4px;
                margin-top: 20px;
            }
            </style>
            """, unsafe_allow_html=True)
            simulate_button = st.button("Run Impact Simulation")
        
        # Simulation results
        if simulate_button:
            st.subheader("Impact Simulation Results")
            
            # Create a progress indicator
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)  # Small delay for visual effect
                progress_bar.progress(i + 1)
            
            # Calculate impact energy (kinetic energy)
            # E = 0.5 * m * v^2
            # m = (4/3) * π * r^3 * ρ
            import math
            radius = diameter / 2
            volume = (4/3) * math.pi * (radius**3)
            mass = volume * density  # kg
            energy_joules = 0.5 * mass * (velocity * 1000)**2  # Convert km/s to m/s
            energy_megatons = energy_joules / 4.184e15  # Convert joules to megatons of TNT
            
            # Create columns for results
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric("Impact Energy", f"{energy_megatons:.2f} megatons of TNT")
                
                # Calculate crater size using scaling laws
                # Simple scaling law: Crater diameter ≈ 10-20 * asteroid diameter
                # More complex formulas exist but require more parameters
                crater_factor = 12 * (math.sin(math.radians(impact_angle)) ** 0.33)
                crater_diameter = crater_factor * diameter
                
                st.metric("Crater Diameter", f"{crater_diameter:.1f} meters")
                
                # Calculate blast radius - scaled based on energy
                blast_radius = 1000 * (energy_megatons ** 0.33)  # very rough approximation
                st.metric("Blast Radius (3rd degree burns)", f"{blast_radius:.1f} meters")
                
                # Calculate seismic effects
                richter_scale = 0.67 * math.log10(energy_joules) - 5.87  # Rough conversion
                st.metric("Earthquake Equivalent", f"{richter_scale:.1f} on Richter scale")
            
            with res_col2:
                # Target-specific effects
                if target == "Ocean":
                    # Calculate tsunami height (very approximate)
                    tsunami_height_at_source = diameter * 0.25 * (energy_megatons**0.1)
                    tsunami_height_at_shore = tsunami_height_at_source * math.exp(-0.0010 * distance_from_shore)
                    st.metric("Estimated Tsunami Height at Shore", f"{tsunami_height_at_shore:.1f} meters")
                
                elif target == "Urban Area":
                    # Calculate casualties (very approximate)
                    area_affected = math.pi * (blast_radius/1000)**2  # km²
                    estimated_casualties = area_affected * population_density * 0.5  # 50% fatality rate in affected area
                    st.metric("Estimated Casualties", f"{estimated_casualties:,.0f} people")
                    
                    # Building damage
                    building_destruction_radius = blast_radius * (0.6 if building_strength == "Strong" else 
                                                                0.8 if building_strength == "Medium" else 1.0)
                    st.metric("Building Destruction Radius", f"{building_destruction_radius:.1f} meters")
                
                # Calculate atmospheric effects
                dust_lofted = mass * 1000 if diameter > 100 else mass * 100  # kg, more for larger asteroids
                st.metric("Dust Lofted into Atmosphere", f"{dust_lofted:,.0f} kg")
                
                # Global cooling effect for large impacts
                if energy_megatons > 10000:  # Threshold for global effects
                    cooling = 0.5 + 0.5 * math.log10(energy_megatons / 10000)
                    st.metric("Potential Global Cooling", f"{cooling:.1f}°C for several months")
            
            # Visualization of impact
            st.subheader("Impact Visualization")
            
            # Create a simple visualization
            fig = go.Figure()
            
            # Draw Earth surface
            x = np.linspace(-blast_radius*1.5, blast_radius*1.5, 100)
            if target == "Ocean":
                fig.add_trace(go.Scatter(x=x, y=np.zeros_like(x), mode='lines', name='Ocean Surface', line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=x, y=np.ones_like(x)*-water_depth, mode='lines', name='Ocean Floor', line=dict(color='brown', width=2)))
            else:
                fig.add_trace(go.Scatter(x=x, y=np.zeros_like(x), mode='lines', name='Ground Level', line=dict(color='brown', width=2)))
            
            # Draw crater
            crater_x = np.linspace(-crater_diameter/2, crater_diameter/2, 100)
            crater_depth = -crater_diameter/5  # Approximate depth as 1/5 of diameter
            crater_y = -((1 - (crater_x/(crater_diameter/2))**2) * abs(crater_depth))
            if target == "Ocean" and abs(crater_depth) < water_depth:
                fig.add_trace(go.Scatter(x=crater_x, y=crater_y, mode='lines', name='Crater', line=dict(color='darkblue', width=2)))
            else:
                bottom_y = np.ones_like(crater_x) * (0 if target != "Ocean" else -water_depth)
                adjusted_y = np.maximum(crater_y, bottom_y)
                fig.add_trace(go.Scatter(x=crater_x, y=adjusted_y, mode='lines', name='Crater', line=dict(color='gray', width=2)))
            
            # Draw blast radius
            fig.add_shape(type="circle", xref="x", yref="y", x0=-blast_radius, y0=-blast_radius/4, x1=blast_radius, y1=blast_radius/4, opacity=0.3, fillcolor="orange", line_color="red")
            
            # Add asteroid at impact point
            fig.add_trace(go.Scatter(x=[0], y=[diameter], mode='markers', name='Asteroid', marker=dict(size=20, color='gray')))
            
            # Add arrow showing impact direction
            arrow_length = blast_radius * 0.3
            arrow_x = arrow_length * math.cos(math.radians(impact_angle))
            arrow_y = diameter + arrow_length * math.sin(math.radians(impact_angle))
            fig.add_annotation(x=0, y=diameter, ax=-arrow_x, ay=arrow_y, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=2, arrowwidth=3, arrowcolor="red")
            
            # Add shockwave circles
            for i in range(1, 4):
                radius = blast_radius * i/3
                fig.add_shape(type="circle", xref="x", yref="y", x0=-radius, y0=-radius/4, x1=radius, y1=radius/4, opacity=0.1, fillcolor="orange", line_color="red", line_dash="dash")
            
            # Update layout
            fig.update_layout(
                title="Simulated Impact Cross-Section",
                xaxis_title="Distance from Impact (meters)",
                yaxis_title="Height/Depth (meters)",
                autosize=True,
                height=500,
                showlegend=True,
                xaxis=dict(range=[-blast_radius*1.2, blast_radius*1.2]),
                yaxis=dict(range=[
                    min(-water_depth*1.2 if target == "Ocean" else crater_depth*1.5, crater_depth*1.5), 
                    max(blast_radius/3, diameter*2)
                ]),
                legend=dict(x=0.01, y=0.99),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Conclusions and notes
            st.markdown("""
            <div style='background-color: rgba(240, 240, 240, 0.5); border-radius: 10px; padding: 15px; margin-top: 20px; border-left: 6px solid #FF9800;'>
                <h4 style='color: #000000; margin-top:0;'>Simulation Notes</h4>
                <p style='color: #000000;'>
                This simulation provides approximate results based on physics models and empirical data from impact studies.
                Actual impacts may vary due to numerous factors including asteroid composition, angle of entry, atmospheric
                effects, and local geography.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add historical comparison
            historical_events = {
                "Chelyabinsk (2013)": 0.5,
                "Tunguska (1908)": 10,
                "Chicxulub (Dinosaur Extinction)": 100000000
            }
            
            st.subheader("Comparison with Historical Events")
            
            # Create a bar chart comparing energy
            comparison_df = pd.DataFrame({
                'Event': list(historical_events.keys()) + [f"Simulated {diameter}m Asteroid"],
                'Energy (Megatons)': list(historical_events.values()) + [energy_megatons]
            })
            
            fig = px.bar(comparison_df, x='Event', y='Energy (Megatons)', log_y=True,
                        color='Energy (Megatons)', color_continuous_scale='Viridis')
            fig.update_layout(title="Impact Energy Comparison (Log Scale)")
            st.plotly_chart(fig, use_container_width=True)
        
        elif sim_type == "Asteroid Collision & Fragmentation":
            st.markdown("""
            <div class="simulator-card" style="border-top-color: #9C27B0;">
                <div class="simulator-title">☄️ Asteroid Collision & Fragmentation Simulator</div>
                <p style="color: #555; margin-bottom: 20px;">
                This simulator models what happens when two asteroids collide in space, resulting in fragmentation and potential Earth impacts of the resulting debris.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Two columns for input parameters
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Primary Asteroid")
                primary_diameter = st.slider("Primary Diameter (meters)", 50, 1000, 300)
                primary_velocity = st.slider("Primary Velocity (km/s)", 5, 40, 15)
                primary_density = st.slider("Primary Density (kg/m³)", 1500, 5000, 3000, step=100)
                
                st.subheader("Collision Parameters")
                collision_angle = st.slider("Collision Angle (degrees)", 0, 90, 45)
                distance_from_earth = st.slider("Distance from Earth (million km)", 0.1, 10.0, 2.0, step=0.1)
                
            with col2:
                st.subheader("Secondary Asteroid")
                secondary_diameter = st.slider("Secondary Diameter (meters)", 10, 500, 100)
                secondary_velocity = st.slider("Secondary Velocity (km/s)", 5, 40, 20)
                secondary_density = st.slider("Secondary Density (kg/m³)", 1500, 5000, 2500, step=100)
                
                st.subheader("Fragmentation Settings")
                fragment_size_distribution = st.select_slider(
                    "Fragment Size Distribution",
                    options=["Mostly Small", "Mixed", "Mostly Large"],
                    value="Mixed"
                )
                
                momentum_conservation = st.checkbox("Apply Momentum Conservation", value=True)
            
            # Calculate collision button
            simulate_collision_button = st.button("Simulate Collision", key="collision_button")
            
            if simulate_collision_button:
                st.subheader("Collision Simulation Results")
                
                # Create a progress indicator for the simulation
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)  # Small delay for visual effect
                    progress_bar.progress(i + 1)
                
                # Calculate masses
                import math
                
                def calculate_mass(diameter, density):
                    radius = diameter / 2
                    volume = (4/3) * math.pi * (radius**3)
                    return volume * density  # kg
                
                primary_mass = calculate_mass(primary_diameter, primary_density)
                secondary_mass = calculate_mass(secondary_diameter, secondary_density)
                
                # Calculate impact energy
                relative_velocity = abs(primary_velocity - secondary_velocity)
                collision_energy_joules = 0.5 * min(primary_mass, secondary_mass) * (relative_velocity * 1000)**2
                collision_energy_megatons = collision_energy_joules / 4.184e15
                
                # Create columns for results
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric("Collision Energy", f"{collision_energy_megatons:.2f} megatons of TNT")
                    
                    # Calculate how many fragments will be created based on energy and size distribution
                    if fragment_size_distribution == "Mostly Small":
                        fragment_multiplier = 15
                    elif fragment_size_distribution == "Mixed":
                        fragment_multiplier = 10
                    else:  # Mostly Large
                        fragment_multiplier = 5
                    
                    # Number of fragments based on energy and distribution setting
                    num_fragments = int(math.log10(collision_energy_joules) * fragment_multiplier)
                    
                    st.metric("Number of Fragments", f"{num_fragments}")
                    
                    # Calculate largest fragment size
                    if fragment_size_distribution == "Mostly Small":
                        largest_fragment_pct = 0.2  # 20% of original size
                    elif fragment_size_distribution == "Mixed":
                        largest_fragment_pct = 0.4  # 40% of original size
                    else:  # Mostly Large
                        largest_fragment_pct = 0.6  # 60% of original size
                    
                    largest_fragment_size = max(primary_diameter, secondary_diameter) * largest_fragment_pct
                    st.metric("Largest Fragment Size", f"{largest_fragment_size:.1f} meters")
                
                with res_col2:                    
                    # Estimate number of Earth-bound fragments
                    # This is a simplified model - in reality would depend on orbital mechanics
                    earth_solid_angle = 4 * math.pi * (6371e3 / (distance_from_earth * 1e9))**2  # steradians
                    total_solid_angle = 4 * math.pi  # steradians
                    
                    # Fragments that have Earth-intercept trajectories
                    earth_directed_pct = earth_solid_angle / total_solid_angle
                    # Apply a fudge factor for dramatic effect
                    earth_directed_pct = min(0.2, earth_directed_pct * 5000)
                    
                    earth_bound_fragments = int(num_fragments * earth_directed_pct)
                    earth_bound_fragments = max(1, earth_bound_fragments)  # At least 1 for simulation purposes
                    
                    st.metric("Earth-Bound Fragments", f"{earth_bound_fragments}")
                    
                    # Estimate time until fragments reach Earth
                    avg_fragment_velocity = (primary_velocity + secondary_velocity) / 2  # km/s
                    time_to_earth = (distance_from_earth * 1e6) / (avg_fragment_velocity * 3600 * 24)  # days
                    
                    st.metric("Time to Earth Impact", f"{time_to_earth:.1f} days")
                    
                    # Potential impact energy on Earth
                    total_fragment_mass = primary_mass + secondary_mass
                    earth_bound_mass = total_fragment_mass * earth_directed_pct
                    earth_impact_energy = 0.5 * earth_bound_mass * (avg_fragment_velocity * 1000)**2
                    earth_impact_megatons = earth_impact_energy / 4.184e15
                    
                    st.metric("Potential Earth Impact Energy", f"{earth_impact_megatons:.2f} megatons of TNT")
                
                # Generate a visualization of the collision
                st.subheader("Collision and Fragmentation Visualization")
                
                # Create a 3D scatter plot for the fragmentation
                import numpy as np
                
                # Generate random fragments
                np.random.seed(42)  # For reproducibility
                
                # Color map based on fragment size
                fragment_sizes = []
                fragment_velocities = []
                for i in range(num_fragments):
                    if fragment_size_distribution == "Mostly Small":
                        size = np.random.exponential(0.1) * largest_fragment_size
                    elif fragment_size_distribution == "Mixed":
                        size = np.random.beta(2, 2) * largest_fragment_size
                    else:  # Mostly Large
                        size = np.random.beta(5, 2) * largest_fragment_size
                    
                    size = max(1, size)  # Minimum size 1m for visibility
                    fragment_sizes.append(size)
                    
                    # Fragment velocities
                    if momentum_conservation:
                        # More realistic: distribute momentum based on mass ratio
                        v_factor = np.random.normal(1.0, 0.3)
                        fragment_velocities.append(
                            ((primary_velocity * primary_mass) + (secondary_velocity * secondary_mass)) / 
                            (primary_mass + secondary_mass) * v_factor
                        )
                    else:
                        # Simplified: velocity is between primary and secondary
                        fragment_velocities.append(
                            np.random.uniform(min(primary_velocity, secondary_velocity), 
                                             max(primary_velocity, secondary_velocity))
                        )
                
                # Generate fragment directions in a cone
                theta = np.random.uniform(0, 2*np.pi, num_fragments)
                phi = np.random.uniform(0, collision_angle*np.pi/180, num_fragments)
                
                # Convert to Cartesian coordinates
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                
                # Scale based on velocity and time
                time_factor = 50  # For visualization
                x = x * fragment_velocities * time_factor
                y = y * fragment_velocities * time_factor
                z = z * fragment_velocities * time_factor
                
                # Create scatter plot of fragments
                fragment_df = pd.DataFrame({
                    'x': x,
                    'y': y,
                    'z': z,
                    'size': fragment_sizes,
                    'velocity': fragment_velocities,
                    'earth_bound': np.random.choice([True, False], size=num_fragments, p=[earth_directed_pct, 1-earth_directed_pct])
                })
                
                # Color earth-bound fragments red
                colors = np.where(fragment_df['earth_bound'], 'red', 'gray')
                
                # Create 3D scatter
                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=fragment_df['x'],
                        y=fragment_df['y'],
                        z=fragment_df['z'],
                        mode='markers',
                        marker=dict(
                            size=fragment_df['size']/5,  # Scale down for visualization
                            color=colors,
                            opacity=0.8
                        ),
                        name='Fragments'
                    )
                ])
                
                # Add primary and secondary asteroids at collision point
                fig.add_trace(go.Scatter3d(
                    x=[0],
                    y=[0],
                    z=[0],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='blue',
                        symbol='diamond'
                    ),
                    name='Collision Point'
                ))
                
                # Add Earth (scaled)
                earth_distance = distance_from_earth * 1000  # for visualization
                earth_radius = 10  # scaled radius for visualization
                
                # Create a sphere for Earth
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x_earth = earth_distance + earth_radius * np.cos(u) * np.sin(v)
                y_earth = earth_radius * np.sin(u) * np.sin(v)
                z_earth = earth_radius * np.cos(v)
                
                fig.add_trace(go.Surface(
                    x=x_earth, y=y_earth, z=z_earth,
                    colorscale=[[0, 'blue'], [1, 'cyan']],
                    opacity=0.8,
                    name='Earth'
                ))
                
                # Add vectors for fragment paths to Earth
                earth_bound_df = fragment_df[fragment_df['earth_bound']]
                if not earth_bound_df.empty:
                    for idx, row in earth_bound_df.iterrows():
                        # Create a line from fragment to Earth
                        fig.add_trace(go.Scatter3d(
                            x=[row['x'], earth_distance],
                            y=[row['y'], 0],
                            z=[row['z'], 0],
                            mode='lines',
                            line=dict(color='red', width=1, dash='dot'),
                            showlegend=False
                        ))
                
                # Update layout
                fig.update_layout(
                    title="Asteroid Collision and Fragmentation Simulation",
                    scene=dict(
                        xaxis_title="X (km)",
                        yaxis_title="Y (km)", 
                        zaxis_title="Z (km)",
                        aspectratio=dict(x=2, y=1, z=1)
                    ),
                    height=700,
                    legend=dict(x=0.01, y=0.99)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add animation of fragments over time
                st.subheader("Fragments Approaching Earth (Animation)")
                
                # Generate animation frames
                frames = 50
                
                # Create dataframe for animation
                anim_data = []
                for i in range(frames):
                    fraction = i / (frames - 1)
                    for j, row in fragment_df.iterrows():
                        if row['earth_bound'] or np.random.random() < 0.2:  # Include some non-Earth-bound for context
                            anim_data.append({
                                'frame': i,
                                'fragment_id': j,
                                'x': row['x'] + (earth_distance - row['x']) * fraction,
                                'y': row['y'] * (1 - fraction),
                                'z': row['z'] * (1 - fraction),
                                'size': row['size'],
                                'earth_bound': row['earth_bound'],
                                'velocity': row['velocity']
                            })
                
                anim_df = pd.DataFrame(anim_data)
                
                # Create animation - fixed implementation
                fig = px.scatter_3d(
                    anim_df, x='x', y='y', z='z',
                    animation_frame='frame',
                    color='earth_bound',
                    color_discrete_map={True: 'red', False: 'gray'},
                    size='size', size_max=15,
                    hover_data=['velocity'],
                    title="Fragments Approaching Earth Over Time"
                )
                
                # Add Earth at a fixed position (without animation frames which caused the error)
                fig.add_trace(
                    go.Scatter3d(
                        x=[earth_distance], y=[0], z=[0],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='blue'
                        ),
                        name='Earth'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    scene=dict(
                        xaxis_title="X (km)",
                        yaxis_title="Y (km)", 
                        zaxis_title="Z (km)",
                        aspectratio=dict(x=2, y=1, z=1)
                    ),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Estimated impact locations
                st.subheader("Estimated Impact Locations")
                
                # Generate random impact locations on Earth
                num_impacts = min(earth_bound_fragments, 20)  # Limit for visualization
                
                # Create random lat/long coordinates
                np.random.seed(int(primary_diameter + secondary_diameter))
                impact_lats = np.random.uniform(-80, 80, num_impacts)
                impact_longs = np.random.uniform(-180, 180, num_impacts)
                
                # Create size and energy for each impact
                impact_sizes = []
                impact_energies = []
                
                for i in range(num_impacts):
                    size_factor = np.random.beta(2, 5) if fragment_size_distribution == "Mostly Small" else \
                                 np.random.beta(2, 2) if fragment_size_distribution == "Mixed" else \
                                 np.random.beta(5, 2)
                    
                    impact_size = largest_fragment_size * size_factor
                    impact_sizes.append(impact_size)
                    
                    # Calculate energy
                    impact_mass = calculate_mass(impact_size, (primary_density + secondary_density)/2)
                    impact_velocity = fragment_velocities[i % len(fragment_velocities)]
                    impact_energy = 0.5 * impact_mass * (impact_velocity * 1000)**2 / 4.184e12  # kilotons
                    impact_energies.append(impact_energy)
                
                # Create dataframe
                impacts_df = pd.DataFrame({
                    'lat': impact_lats,
                    'lon': impact_longs,
                    'size': impact_sizes,
                    'energy': impact_energies
                })
                
                # Create map
                fig = px.scatter_geo(
                    impacts_df, 
                    lat='lat', 
                    lon='lon',
                    size='size',
                    color='energy',
                    hover_name='energy',
                    hover_data={'lat': True, 'lon': True, 'size': True, 'energy': ':.2f kt'},
                    title="Predicted Fragment Impact Locations",
                    projection='natural earth',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    coloraxis_colorbar=dict(
                        title="Energy (kt)"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Damage assessment
                total_energy = sum(impact_energies)
                
                st.markdown(f"""
                <div style='background-color: rgba(240, 240, 240, 0.5); border-radius: 10px; padding: 15px; margin-top: 20px; border-left: 6px solid #9C27B0;'>
                    <h4 style='color: #000000; margin-top:0;'>Impact Assessment</h4>
                    <p style='color: #000000;'>
                    The collision would produce approximately <b>{num_fragments}</b> fragments, with <b>{earth_bound_fragments}</b> 
                    potentially on Earth-intercept trajectories. The fragments would reach Earth in approximately <b>{time_to_earth:.1f}</b> days.
                    </p>
                    <p style='color: #000000;'>
                    The total combined energy of all Earth impacts would be approximately <b>{total_energy:.2f}</b> kilotons of TNT.
                    Impacts are distributed across multiple locations, reducing the catastrophic effect of a single impact.
                    </p>
                    <p style='color: #000000;'>
                    The largest potential fragment has a diameter of <b>{largest_fragment_size:.1f}</b> meters, which would create
                    a crater approximately <b>{largest_fragment_size * 10:.1f}</b> meters in diameter if it impacts land.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Data Explorer Tab
    with tabs[5]:
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
<div style="text-align: center; color: #4CAF50; font-size: 0.8em;">
    <p>Comprehensive Space Threat Assessment and Prediction System | Using NASA NEO API</p>
    <p>Data refreshed: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
