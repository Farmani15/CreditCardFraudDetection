"""
Data preprocessing module for credit card fraud detection.
Includes optimized functions for loading, cleaning, and preparing data.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from tqdm import tqdm
import logging
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.DATA_DIR, 'preprocessing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data(filepath=None, sample_size=None):
    """
    Load the credit card dataset with performance optimization.
    
    Args:
        filepath (str, optional): Path to the dataset. Defaults to config.RAW_DATA_PATH.
        sample_size (int, optional): Number of samples to load for testing. Defaults to None (load all).
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    filepath = filepath or config.RAW_DATA_PATH
    
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found at {filepath}. Please download it first.")
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    logger.info(f"Loading data from {filepath}")
    
    # For large datasets, use chunking to optimize memory usage
    if sample_size:
        # Load only a sample for testing
        df = pd.read_csv(filepath, nrows=sample_size)
    else:
        # Check file size to determine if chunking is needed
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
        
        if file_size > 500:  # If file is larger than 500MB
            logger.info(f"Large file detected ({file_size:.2f} MB). Using chunked loading.")
            chunks = []
            for chunk in tqdm(pd.read_csv(filepath, chunksize=config.CHUNK_SIZE), 
                             desc="Loading data chunks"):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(filepath)
    
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


def clean_data(df):
    """
    Clean the dataset by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    logger.info("Cleaning data...")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        logger.info(f"Found {missing_values} missing values. Handling them...")
        # For numerical features, fill with median
        for col in df.columns:
            if df[col].dtype != 'object':
                df[col].fillna(df[col].median(), inplace=True)
    
    # Handle outliers in the Amount column using capping
    q1 = df['Amount'].quantile(0.01)
    q3 = df['Amount'].quantile(0.99)
    df['Amount'] = df['Amount'].clip(q1, q3)
    
    logger.info("Data cleaning completed")
    return df


def preprocess_data(df, scale=True, handle_imbalance=True, test_size=None):
    """
    Preprocess the data for model training with performance optimizations.
    
    Args:
        df (pd.DataFrame): Input dataframe
        scale (bool, optional): Whether to scale features. Defaults to True.
        handle_imbalance (bool, optional): Whether to handle class imbalance. Defaults to True.
        test_size (float, optional): Test set size. Defaults to config.TEST_SIZE.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    logger.info("Preprocessing data...")
    
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Extract features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Create a new feature for hour of day from Time
    X['Hour'] = (X['Time'] / 3600) % 24
    
    # Drop the Time column as it's not useful for prediction
    X = X.drop('Time', axis=1)
    
    # Split the data
    test_size = test_size or config.TEST_SIZE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Scale the features
    scaler = None
    if scale:
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )
        
        # Save the scaler
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, config.SCALER_PATH)
        logger.info(f"Scaler saved to {config.SCALER_PATH}")
    
    # Handle class imbalance
    if handle_imbalance:
        logger.info("Handling class imbalance with SMOTE...")
        # Only apply SMOTE to the training data
        smote = SMOTE(random_state=config.RANDOM_STATE)
        
        # For large datasets, use a sample for SMOTE
        if len(X_train) > 100000:
            logger.info("Large dataset detected. Using stratified sampling for SMOTE.")
            # Get indices of fraud and non-fraud samples
            fraud_idx = y_train[y_train == 1].index
            non_fraud_idx = y_train[y_train == 0].index
            
            # Sample non-fraud cases (10 times the number of fraud cases)
            non_fraud_sample_idx = np.random.choice(
                non_fraud_idx, 
                size=min(len(fraud_idx) * 10, len(non_fraud_idx)),
                replace=False
            )
            
            # Combine indices
            combined_idx = np.concatenate([fraud_idx, non_fraud_sample_idx])
            
            # Apply SMOTE to the sample
            X_train_sample = X_train.loc[combined_idx]
            y_train_sample = y_train.loc[combined_idx]
            
            X_resampled, y_resampled = smote.fit_resample(X_train_sample, y_train_sample)
            
            # Replace the original samples with the resampled ones and keep the rest
            X_train_rest = X_train.drop(combined_idx)
            y_train_rest = y_train.drop(combined_idx)
            
            X_train = pd.concat([pd.DataFrame(X_resampled, columns=X_train.columns), X_train_rest])
            y_train = pd.concat([pd.Series(y_resampled), y_train_rest])
        else:
            X_train, y_train = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Preprocessing completed. Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler


def save_processed_data(X_train, X_test, y_train, y_test):
    """
    Save the processed data to disk.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Testing labels
    """
    logger.info("Saving processed data...")
    
    # Create directory if it doesn't exist
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save training data
    train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    train_data.to_csv(config.TRAIN_DATA_PATH, index=False)
    
    # Save testing data
    test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    test_data.to_csv(config.TEST_DATA_PATH, index=False)
    
    logger.info(f"Processed data saved to {config.PROCESSED_DATA_DIR}")


def main():
    """Main function to run the data preprocessing pipeline."""
    logger.info("Starting data preprocessing pipeline")
    
    # Load data
    df = load_data()
    
    # Clean data
    df = clean_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    
    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test)
    
    logger.info("Data preprocessing pipeline completed successfully")


if __name__ == "__main__":
    main() 