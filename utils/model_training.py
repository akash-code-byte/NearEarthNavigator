import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)
import xgboost as xgb

def prepare_data_for_training(df):
    """
    Prepare data for model training
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with asteroid data
        
    Returns:
    --------
    tuple
        X, y, feature_names for model training
    """
    # Check if target variable exists
    if 'is_potentially_hazardous' not in df.columns:
        raise ValueError("Target variable 'is_potentially_hazardous' not found in dataframe")
    
    # Select features
    feature_cols = [
        'absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max',
        'relative_velocity', 'miss_distance', 'diameter_diff', 'diameter_velocity_ratio',
        'energy_proxy', 'proximity_risk', 'orbital_stability_index', 'size_uncertainty'
    ]
    
    # Ensure all feature columns exist
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if len(available_cols) < 5:  # Require at least 5 features
        raise ValueError(f"Not enough features available. Found only {len(available_cols)}")
    
    # Prepare data
    X = df[available_cols].values
    y = df['is_potentially_hazardous'].values
    
    return X, y, available_cols

def train_and_evaluate_models(df, model_type='Random Forest', model_params=None):
    """
    Train and evaluate models with hyperparameter tuning and cross-validation
    """
    """
    Train and evaluate machine learning models for asteroid hazard prediction
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with asteroid data
    model_type : str
        Type of model to train: 'Random Forest', 'Logistic Regression', 'XGBoost', 'k-Nearest Neighbors', 'Neural Network'
    model_params : dict
        Additional parameters for the model
        
    Returns:
    --------
    tuple
        Trained model, metrics dictionary, feature importance dictionary
    """
    # Default parameters if not provided
    if model_params is None:
        model_params = {}
    
    # Prepare data
    X, y, feature_names = prepare_data_for_training(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select and train model
    if model_type == 'Random Forest':
        model = train_random_forest(X_train_scaled, y_train, model_params)
    elif model_type == 'Logistic Regression':
        model = train_logistic_regression(X_train_scaled, y_train, model_params)
    elif model_type == 'XGBoost':
        model = train_xgboost(X_train_scaled, y_train, model_params)
    elif model_type == 'k-Nearest Neighbors':
        model = train_knn(X_train_scaled, y_train, model_params)
    elif model_type == 'Neural Network':
        model = train_neural_network(X_train_scaled, y_train, model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Evaluate model
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # Get feature importance (if applicable)
    feature_importance = extract_feature_importance(model, feature_names, model_type)
    
    # Make predictions on the full dataset for further analysis
    X_full_scaled = scaler.transform(X)
    
    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_full_scaled)[:, 1]
    else:
        # For models without predict_proba, use decision function if available
        if hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_full_scaled)
            # Normalize to [0, 1] range
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        else:
            # Default to binary predictions
            y_prob = model.predict(X_full_scaled).astype(float)
    
    # Add predictions to the dataframe
    df['predicted_hazard'] = model.predict(X_full_scaled)
    df['hazard_probability'] = y_prob
    
    return model, metrics, feature_importance

def train_random_forest(X_train, y_train, params):
    """
    Train a Random Forest classifier
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    params : dict
        Additional parameters for the model
        
    Returns:
    --------
    sklearn.ensemble.RandomForestClassifier
        Trained model
    """
    # Default parameters
    n_estimators = params.get('n_estimators', 200)
    max_depth = params.get('max_depth', 10)
    
    # Check if hyperparameter optimization is requested
    if params.get('optimize', False):
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create base model
        base_model = RandomForestClassifier(random_state=42)
        
        # Perform grid search
        model = GridSearchCV(
            base_model, param_grid, cv=5, scoring='f1', n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Get best model
        return model.best_estimator_
    else:
        # Train with specified parameters
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

def train_logistic_regression(X_train, y_train, params):
    """
    Train a Logistic Regression classifier
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    params : dict
        Additional parameters for the model
        
    Returns:
    --------
    sklearn.linear_model.LogisticRegression
        Trained model
    """
    # Default parameters
    C = params.get('C', 1.0)
    
    # Check if hyperparameter optimization is requested
    if params.get('optimize', False):
        # Define parameter grid
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
        
        # Create base model
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Perform grid search
        model = GridSearchCV(
            base_model, param_grid, cv=5, scoring='f1', n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Get best model
        return model.best_estimator_
    else:
        # Train with specified parameters
        model = LogisticRegression(
            C=C,
            random_state=42,
            max_iter=1000
        )
        model.fit(X_train, y_train)
        return model

def train_xgboost(X_train, y_train, params):
    """
    Train an XGBoost classifier
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    params : dict
        Additional parameters for the model
        
    Returns:
    --------
    xgboost.XGBClassifier
        Trained model
    """
    # Default parameters
    learning_rate = params.get('learning_rate', 0.1)
    max_depth = params.get('max_depth', 6)
    
    # Check if hyperparameter optimization is requested
    if params.get('optimize', False):
        # Define parameter grid
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 6, 9, 12],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        # Create base model
        base_model = xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Perform grid search
        model = GridSearchCV(
            base_model, param_grid, cv=5, scoring='f1', n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Get best model
        return model.best_estimator_
    else:
        # Train with specified parameters
        model = xgb.XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        return model

def train_knn(X_train, y_train, params):
    """
    Train a k-Nearest Neighbors classifier
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    params : dict
        Additional parameters for the model
        
    Returns:
    --------
    sklearn.neighbors.KNeighborsClassifier
        Trained model
    """
    # Default parameters
    n_neighbors = params.get('n_neighbors', 5)
    
    # Check if hyperparameter optimization is requested
    if params.get('optimize', False):
        # Define parameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        # Create base model
        base_model = KNeighborsClassifier()
        
        # Perform grid search
        model = GridSearchCV(
            base_model, param_grid, cv=5, scoring='f1', n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Get best model
        return model.best_estimator_
    else:
        # Train with specified parameters
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors
        )
        model.fit(X_train, y_train)
        return model

def train_neural_network(X_train, y_train, params):
    """
    Train a Neural Network classifier
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    params : dict
        Additional parameters for the model
        
    Returns:
    --------
    sklearn.neural_network.MLPClassifier
        Trained model
    """
    # Default parameters
    hidden_layers = params.get('hidden_layers', 2)
    neurons_per_layer = params.get('neurons_per_layer', 32)
    
    # Construct hidden layer sizes based on parameters
    hidden_layer_sizes = tuple([neurons_per_layer] * hidden_layers)
    
    # Check if hyperparameter optimization is requested
    if params.get('optimize', False):
        # Define parameter grid
        param_grid = {
            'hidden_layer_sizes': [(32,), (64,), (32, 32), (64, 64), (32, 32, 32)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        
        # Create base model
        base_model = MLPClassifier(
            random_state=42,
            max_iter=1000
        )
        
        # Perform grid search
        model = GridSearchCV(
            base_model, param_grid, cv=5, scoring='f1', n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Get best model
        return model.best_estimator_
    else:
        # Train with specified parameters
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=42,
            max_iter=1000
        )
        model.fit(X_train, y_train)
        return model

def evaluate_model(model, X_test, y_test, X_train=None, y_train=None):
    """Enhanced model evaluation with additional metrics"""
    metrics = {}
    
    # Basic metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred)
    metrics['f1'] = f1_score(y_test, y_pred)
    
    # ROC and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    metrics['auc'] = auc(fpr, tpr)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    metrics['precision_curve'] = precision_curve
    metrics['recall_curve'] = recall_curve
    metrics['avg_precision'] = average_precision_score(y_test, y_prob)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics.update({
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'training_samples': len(X_train) if X_train is not None else None,
        'test_samples': len(X_test)
    })
    
    # Calculate cross-validation scores if training data is provided
    if X_train is not None and y_train is not None:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
    
    # Add class distribution
    metrics['class_distribution'] = dict(zip(*np.unique(y_test, return_counts=True)))
    
    # Add model-specific metrics
    if hasattr(model, 'feature_importances_'):
        metrics['feature_importances'] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        metrics['feature_importances'] = model.coef_[0]
    
    return metrics
    """
    Evaluate model performance
    
    Parameters:
    -----------
    model : object
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test target values
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate probability predictions if available
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # For models without predict_proba, use decision function if available
        if hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
            # Normalize to [0, 1] range
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        else:
            # Default to binary predictions
            y_prob = y_pred.astype(float)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Add cross-validation scores
    cv_scores = cross_val_score(model, X_test, y_test, cv=5)
    
    # Calculate metrics at different probability thresholds
    thresholds = [0.3, 0.5, 0.7]
    precision_at_thresholds = {}
    recall_at_thresholds = {}
    f1_at_thresholds = {}
    
    for threshold in thresholds:
        y_pred_at_threshold = (y_prob >= threshold).astype(int)
        precision_at_thresholds[threshold] = precision_score(y_test, y_pred_at_threshold)
        recall_at_thresholds[threshold] = recall_score(y_test, y_pred_at_threshold)
        f1_at_thresholds[threshold] = f1_score(y_test, y_pred_at_threshold)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Return metrics as dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'avg_precision': avg_precision,
        'confusion_matrix': {
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp
        }
    }
    
    # Add class distribution
    class_distribution = dict(zip(*np.unique(y_test, return_counts=True)))
    
    # Add all metrics to the dictionary
    metrics.update({
        'cross_val_scores': cv_scores,
        'class_distribution': class_distribution,
        'precision_at_thresholds': precision_at_thresholds,
        'recall_at_thresholds': recall_at_thresholds,
        'f1_at_thresholds': f1_at_thresholds
    })
    
    return metrics

def extract_feature_importance(model, feature_names, model_type):
    """
    Extract feature importance from the model if available
    
    Parameters:
    -----------
    model : object
        Trained model
    feature_names : list
        List of feature names
    model_type : str
        Type of model
        
    Returns:
    --------
    dict
        Dictionary mapping feature names to importance values
    """
    feature_importance = {}
    
    try:
        # Extract feature importance based on model type
        if model_type == 'Random Forest':
            importances = model.feature_importances_
            for feature, importance in zip(feature_names, importances):
                feature_importance[feature] = importance
                
        elif model_type == 'Logistic Regression':
            # For logistic regression, we use the absolute values of coefficients
            importances = np.abs(model.coef_[0])
            for feature, importance in zip(feature_names, importances):
                feature_importance[feature] = importance
                
        elif model_type == 'XGBoost':
            # XGBoost has built-in feature importance
            importance_type = 'weight'  # Alternative: 'gain', 'cover', 'total_gain', 'total_cover'
            importances = model.get_booster().get_score(importance_type=importance_type)
            
            # XGBoost uses feature index, so we need to map back to feature names
            for i, feature in enumerate(feature_names):
                feature_idx = f"f{i}"
                if feature_idx in importances:
                    feature_importance[feature] = importances[feature_idx]
                else:
                    feature_importance[feature] = 0
                    
        elif model_type == 'Neural Network':
            # For neural networks, we use a permutation importance approach
            # This is just a placeholder - in a real application, you would compute
            # permutation importance or use techniques like SHAP
            # For simplicity, we assign equal importance to all features
            for feature in feature_names:
                feature_importance[feature] = 1.0 / len(feature_names)
                
        elif model_type == 'k-Nearest Neighbors':
            # KNN doesn't have built-in feature importance
            # We could use permutation importance, but for simplicity, equal importance
            for feature in feature_names:
                feature_importance[feature] = 1.0 / len(feature_names)
                
        else:
            # Default case: equal importance
            for feature in feature_names:
                feature_importance[feature] = 1.0 / len(feature_names)
    
    except Exception as e:
        print(f"Error extracting feature importance: {e}")
        # Fallback to equal importance
        for feature in feature_names:
            feature_importance[feature] = 1.0 / len(feature_names)
    
    # Normalize to sum to 1
    total_importance = sum(feature_importance.values())
    if total_importance > 0:
        for feature in feature_importance:
            feature_importance[feature] /= total_importance
    
    return feature_importance

def save_model(model, filepath):
    """
    Save trained model to disk
    
    Parameters:
    -----------
    model : object
        Trained model
    filepath : str
        Path to save the model
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"Error saving model: {e}")

def load_cached_model(filepath):
    """
    Load cached model from disk
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    object or None
        Loaded model or None if file doesn't exist or error occurs
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            return model
        return None
    except Exception as e:
        print(f"Error loading cached model: {e}")
        return None
