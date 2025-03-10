"""
Feature engineering module for credit card fraud detection.
Includes functions for creating, selecting, and transforming features.
"""
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
import sys
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.DATA_DIR, 'feature_engineering.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_interaction_features(X):
    """
    Create interaction features between selected pairs of features.
    
    Args:
        X (pd.DataFrame): Input features
    
    Returns:
        pd.DataFrame: DataFrame with added interaction features
    """
    logger.info("Creating interaction features...")
    
    # Create a copy to avoid modifying the original dataframe
    X_new = X.copy()
    
    # Select top features based on correlation with the target
    # (This would normally be done with the target variable, but here we're just using a predefined list)
    top_features = ['V4', 'V12', 'V14', 'V17', 'Amount']
    
    # Create interaction features (products of pairs)
    for i, feat1 in enumerate(top_features):
        if feat1 not in X_new.columns:
            continue
            
        for feat2 in top_features[i+1:]:
            if feat2 not in X_new.columns:
                continue
                
            feature_name = f"{feat1}_{feat2}_interaction"
            X_new[feature_name] = X_new[feat1] * X_new[feat2]
    
    logger.info(f"Created {X_new.shape[1] - X.shape[1]} interaction features")
    return X_new


def create_polynomial_features(X, degree=2, selected_features=None):
    """
    Create polynomial features for selected features.
    
    Args:
        X (pd.DataFrame): Input features
        degree (int, optional): Polynomial degree. Defaults to 2.
        selected_features (list, optional): List of features to transform. 
                                           Defaults to None (use predefined list).
    
    Returns:
        pd.DataFrame: DataFrame with added polynomial features
    """
    logger.info(f"Creating polynomial features of degree {degree}...")
    
    # Create a copy to avoid modifying the original dataframe
    X_new = X.copy()
    
    # Use predefined list if not provided
    if selected_features is None:
        selected_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount']
    
    # Filter to only include features that exist in the dataframe
    selected_features = [f for f in selected_features if f in X_new.columns]
    
    # Create polynomial features
    for feature in selected_features:
        for d in range(2, degree + 1):
            feature_name = f"{feature}_pow{d}"
            X_new[feature_name] = X_new[feature] ** d
    
    logger.info(f"Created {X_new.shape[1] - X.shape[1]} polynomial features")
    return X_new


def create_aggregate_features(X):
    """
    Create aggregate features like mean, std, min, max of groups of features.
    
    Args:
        X (pd.DataFrame): Input features
    
    Returns:
        pd.DataFrame: DataFrame with added aggregate features
    """
    logger.info("Creating aggregate features...")
    
    # Create a copy to avoid modifying the original dataframe
    X_new = X.copy()
    
    # Get all V features
    v_features = [col for col in X_new.columns if col.startswith('V')]
    
    # Create groups of features
    feature_groups = {
        'V1_V5': v_features[0:5] if len(v_features) >= 5 else v_features,
        'V6_V10': v_features[5:10] if len(v_features) >= 10 else v_features[5:] if len(v_features) > 5 else [],
        'V11_V15': v_features[10:15] if len(v_features) >= 15 else v_features[10:] if len(v_features) > 10 else [],
        'V16_V20': v_features[15:20] if len(v_features) >= 20 else v_features[15:] if len(v_features) > 15 else [],
        'V21_V28': v_features[20:] if len(v_features) > 20 else []
    }
    
    # Create aggregate features for each group
    for group_name, features in feature_groups.items():
        if not features:
            continue
            
        # Mean
        X_new[f"{group_name}_mean"] = X_new[features].mean(axis=1)
        
        # Standard deviation
        X_new[f"{group_name}_std"] = X_new[features].std(axis=1)
        
        # Min and Max
        X_new[f"{group_name}_min"] = X_new[features].min(axis=1)
        X_new[f"{group_name}_max"] = X_new[features].max(axis=1)
        
        # Range (max - min)
        X_new[f"{group_name}_range"] = X_new[f"{group_name}_max"] - X_new[f"{group_name}_min"]
    
    logger.info(f"Created {X_new.shape[1] - X.shape[1]} aggregate features")
    return X_new


def select_features(X, y, method='random_forest', n_features=None):
    """
    Select the most important features using various methods.
    
    Args:
        X (pd.DataFrame): Input features
        y (pd.Series): Target variable
        method (str, optional): Feature selection method. 
                               Options: 'random_forest', 'mutual_info'. 
                               Defaults to 'random_forest'.
        n_features (int, optional): Number of features to select. 
                                   Defaults to None (auto-determine).
    
    Returns:
        tuple: Selected features DataFrame and list of selected feature names
    """
    logger.info(f"Selecting features using {method} method...")
    
    if method == 'random_forest':
        # Use Random Forest for feature selection
        selector = RandomForestClassifier(
            n_estimators=100, 
            random_state=config.RANDOM_STATE,
            n_jobs=config.N_JOBS,
            class_weight='balanced'
        )
        selector.fit(X, y)
        
        # Get feature importances
        importances = selector.feature_importances_
        
        # Create a DataFrame of feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Determine number of features to select
        if n_features is None:
            # Select features that account for 95% of cumulative importance
            cumulative_importance = feature_importance_df['Importance'].cumsum() / feature_importance_df['Importance'].sum()
            n_features = (cumulative_importance <= 0.95).sum() + 1
        
        # Select top n_features
        selected_features = feature_importance_df['Feature'].iloc[:n_features].tolist()
        
    elif method == 'mutual_info':
        # Use mutual information for feature selection
        importances = mutual_info_classif(X, y, random_state=config.RANDOM_STATE)
        
        # Create a DataFrame of feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Determine number of features to select
        if n_features is None:
            # Select features that account for 95% of cumulative importance
            cumulative_importance = feature_importance_df['Importance'].cumsum() / feature_importance_df['Importance'].sum()
            n_features = (cumulative_importance <= 0.95).sum() + 1
        
        # Select top n_features
        selected_features = feature_importance_df['Feature'].iloc[:n_features].tolist()
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Select the features
    X_selected = X[selected_features]
    
    logger.info(f"Selected {len(selected_features)} features out of {X.shape[1]}")
    return X_selected, selected_features


def engineer_features(X_train, y_train, X_test=None):
    """
    Apply feature engineering pipeline to the data.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame, optional): Testing features. Defaults to None.
    
    Returns:
        tuple: Engineered training features, engineered testing features (if provided),
               and list of selected feature names
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Create interaction features
    X_train = create_interaction_features(X_train)
    if X_test is not None:
        X_test = create_interaction_features(X_test)
    
    # Create polynomial features
    X_train = create_polynomial_features(X_train)
    if X_test is not None:
        X_test = create_polynomial_features(X_test)
    
    # Create aggregate features
    X_train = create_aggregate_features(X_train)
    if X_test is not None:
        X_test = create_aggregate_features(X_test)
    
    # Select the most important features
    X_train_selected, selected_features = select_features(X_train, y_train)
    
    # Apply the same feature selection to test set if provided
    if X_test is not None:
        X_test_selected = X_test[selected_features]
        logger.info("Feature engineering pipeline completed")
        return X_train_selected, X_test_selected, selected_features
    
    logger.info("Feature engineering pipeline completed")
    return X_train_selected, None, selected_features


def main():
    """Main function to run the feature engineering pipeline."""
    logger.info("Starting feature engineering process")
    
    # Load the processed data
    train_data = pd.read_csv(config.TRAIN_DATA_PATH)
    test_data = pd.read_csv(config.TEST_DATA_PATH)
    
    # Split into features and target
    X_train = train_data.drop('Class', axis=1)
    y_train = train_data['Class']
    X_test = test_data.drop('Class', axis=1)
    y_test = test_data['Class']
    
    # Apply feature engineering
    X_train_engineered, X_test_engineered, selected_features = engineer_features(X_train, y_train, X_test)
    
    # Save the engineered data
    engineered_train_data = pd.concat([X_train_engineered, y_train.reset_index(drop=True)], axis=1)
    engineered_test_data = pd.concat([X_test_engineered, y_test.reset_index(drop=True)], axis=1)
    
    engineered_train_path = os.path.join(config.PROCESSED_DATA_DIR, 'engineered_train.csv')
    engineered_test_path = os.path.join(config.PROCESSED_DATA_DIR, 'engineered_test.csv')
    
    engineered_train_data.to_csv(engineered_train_path, index=False)
    engineered_test_data.to_csv(engineered_test_path, index=False)
    
    # Save the list of selected features
    selected_features_path = os.path.join(config.MODEL_DIR, 'selected_features.joblib')
    joblib.dump(selected_features, selected_features_path)
    
    logger.info(f"Engineered data saved to {config.PROCESSED_DATA_DIR}")
    logger.info(f"Selected features saved to {selected_features_path}")
    logger.info("Feature engineering process completed successfully")


if __name__ == "__main__":
    main() 