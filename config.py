"""
Configuration settings for the Credit Card Fraud Detection project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'creditcard.csv')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.csv')

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Model training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25  # Percentage of training data to use for validation

# Performance optimization settings
N_JOBS = -1  # Use all available cores
CHUNK_SIZE = 10000  # For processing large datasets in chunks

# Model hyperparameters
# These are default values that can be overridden during hyperparameter tuning
MODEL_PARAMS = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,
        "tree_method": "hist",  # For faster training
    },
    "lightgbm": {
        "n_estimators": 100,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "boosting_type": "gbdt",
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
        "class_weight": "balanced",
    }
}

# Web application settings
FLASK_DEBUG = False
FLASK_PORT = 8050
FLASK_HOST = '0.0.0.0'  # Listen on all interfaces

# Directories
LOG_DIR = os.path.join(BASE_DIR, 'logs') 