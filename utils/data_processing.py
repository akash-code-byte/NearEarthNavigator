import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def fetch_neo_data(api_key, start_date, end_date):
    """
    Fetch Near Earth Object data from NASA API
    
    Parameters:
    -----------
    api_key : str
        NASA API key
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
        
    Returns:
    --------
    dict or None
        JSON response from NASA API or None if request failed
    """
    base_url = "https://api.nasa.gov/neo/rest/v1/feed"
    
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "api_key": api_key
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching NEO data: {e}")
        return None

def preprocess_data(neo_data):
    """
    Preprocess the NEO data from NASA API response
    
    Parameters:
    -----------
    neo_data : dict
        JSON response from NASA API
        
    Returns:
    --------
    pandas.DataFrame
        Processed data in a DataFrame
    """
    # Initialize empty list for NEO data
    neo_list = []
    
    # Process data for each day in the response
    for date, daily_neos in neo_data["near_earth_objects"].items():
        for neo in daily_neos:
            try:
                # Extract relevant information
                neo_info = {
                    'id': neo.get('id', ''),
                    'name': neo.get('name', ''),
                    'absolute_magnitude': neo.get('absolute_magnitude_h', np.nan),
                    'estimated_diameter_min': neo.get('estimated_diameter', {}).get('kilometers', {}).get('estimated_diameter_min', np.nan),
                    'estimated_diameter_max': neo.get('estimated_diameter', {}).get('kilometers', {}).get('estimated_diameter_max', np.nan),
                    'is_potentially_hazardous': neo.get('is_potentially_hazardous_asteroid', False),
                }
                
                # Check if there's close approach data
                if neo.get('close_approach_data') and len(neo['close_approach_data']) > 0:
                    approach = neo['close_approach_data'][0]
                    neo_info.update({
                        'relative_velocity': float(approach.get('relative_velocity', {}).get('kilometers_per_second', 0)),
                        'miss_distance': float(approach.get('miss_distance', {}).get('kilometers', 0)),
                        'orbiting_body': approach.get('orbiting_body', ''),
                        'close_approach_date': approach.get('close_approach_date', ''),
                    })
                    
                    # Parse date for additional features
                    try:
                        approach_date = datetime.strptime(approach.get('close_approach_date', ''), '%Y-%m-%d')
                        neo_info['approach_month'] = approach_date.month
                        neo_info['approach_day'] = approach_date.day
                        neo_info['close_approach_date_display'] = approach_date.strftime('%Y-%m-%d')
                    except ValueError:
                        neo_info['approach_month'] = np.nan
                        neo_info['approach_day'] = np.nan
                        neo_info['close_approach_date_display'] = ''
                else:
                    # No approach data
                    neo_info.update({
                        'relative_velocity': np.nan,
                        'miss_distance': np.nan,
                        'orbiting_body': '',
                        'close_approach_date': '',
                        'approach_month': np.nan,
                        'approach_day': np.nan,
                        'close_approach_date_display': '',
                    })
                
                # Check for missing data
                has_missing = (
                    np.isnan(neo_info['absolute_magnitude']) or
                    np.isnan(neo_info['estimated_diameter_min']) or
                    np.isnan(neo_info['estimated_diameter_max']) or
                    np.isnan(neo_info['relative_velocity']) or
                    np.isnan(neo_info['miss_distance'])
                )
                neo_info['has_missing_data'] = has_missing
                
                # Add to list if no critical data is missing
                if not has_missing:
                    neo_list.append(neo_info)
            except Exception as e:
                print(f"Error processing NEO {neo.get('id', 'unknown')}: {e}")
                continue
    
    # Create DataFrame
    df = pd.DataFrame(neo_list)
    
    return df

def feature_engineering(df):
    """
    Perform feature engineering on the NEO data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed NEO data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with engineered features
    """
    # Calculate size uncertainty (ratio of max to min diameter)
    df['size_uncertainty'] = np.sqrt(df['estimated_diameter_max'] / df['estimated_diameter_min'])
    
    # Calculate difference between min and max diameter
    df['diameter_diff'] = df['estimated_diameter_max'] - df['estimated_diameter_min']
    
    # Calculate ratio of diameter to velocity (small objects with high velocity can be dangerous)
    df['diameter_velocity_ratio'] = df['estimated_diameter_max'] / df['relative_velocity']
    
    # Calculate a simple energy proxy (proportional to kinetic energy)
    # E ~ diameter^3 * velocity^2 (assuming density is constant)
    df['energy_proxy'] = df['estimated_diameter_max']**3 * df['relative_velocity']**2
    
    # Calculate proximity risk (inversely proportional to miss distance)
    # Normalized to [0, 1] range for better interpretability
    max_distance = df['miss_distance'].max()
    if max_distance > 0:
        df['proximity_risk'] = 1 - (df['miss_distance'] / max_distance)
    else:
        df['proximity_risk'] = 0
    
    # Calculate orbital stability index (a measure of how stable the orbit is)
    # This is a simplified version considering velocity and miss distance
    df['orbital_stability_index'] = 1 / (df['relative_velocity'] * df['miss_distance'])
    
    return df

def apply_scaling(df, method='StandardScaler'):
    """
    Apply feature scaling to the numerical columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features
    method : str
        Scaling method: 'StandardScaler', 'MinMaxScaler', or 'RobustScaler'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with scaled features
    """
    # Columns to scale (numerical columns only, excluding target and ID columns)
    cols_to_scale = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cols_to_scale = [col for col in cols_to_scale if col not in ['id', 'is_potentially_hazardous']]
    
    # Select the appropriate scaler
    if method == 'StandardScaler':
        scaler = StandardScaler()
    elif method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif method == 'RobustScaler':
        scaler = RobustScaler()
    else:
        return df  # Return original data if no valid method specified
    
    # Apply scaling
    scaled_df = df.copy()
    scaled_df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    return scaled_df

def remove_outliers_zscore(df, threshold=3.0):
    """
    Remove outliers using Z-score method
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features
    threshold : float
        Z-score threshold for outlier detection
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    # Columns to check for outliers (numerical columns only, excluding target and ID columns)
    cols_to_check = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cols_to_check = [col for col in cols_to_check if col not in ['id', 'is_potentially_hazardous']]
    
    # Initialize mask for all rows
    mask = np.ones(len(df), dtype=bool)
    
    # Check each column for outliers
    for col in cols_to_check:
        # Calculate z-scores
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        
        # Update mask to exclude outliers
        mask = mask & (z_scores < threshold)
    
    # Return filtered DataFrame
    return df[mask].reset_index(drop=True)

def load_cached_data(filename):
    """
    Load cached NEO data from a CSV file
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame or None
        Loaded data or None if file doesn't exist or error occurs
    """
    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            return df
        return None
    except Exception as e:
        print(f"Error loading cached data: {e}")
        return None

def save_data_cache(df, filename):
    """
    Save NEO data to a CSV file for caching
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to save
    filename : str
        Path to save the CSV file
    """
    try:
        df.to_csv(filename, index=False)
    except Exception as e:
        print(f"Error saving data cache: {e}")
