"""
Model prediction module for credit card fraud detection.
Includes functions for loading models and making predictions.
"""
import os
import pandas as pd
import numpy as np
import joblib
import logging
import sys
import time

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.MODEL_DIR, 'prediction.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_model(model_path=None):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str, optional): Path to the model file. 
                                   Defaults to None (use config.BEST_MODEL_PATH).
    
    Returns:
        object: Loaded model
    """
    model_path = model_path or config.BEST_MODEL_PATH
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    
    return model


def load_scaler():
    """
    Load the feature scaler from disk.
    
    Returns:
        object: Loaded scaler
    """
    if not os.path.exists(config.SCALER_PATH):
        logger.error(f"Scaler not found at {config.SCALER_PATH}")
        raise FileNotFoundError(f"Scaler not found at {config.SCALER_PATH}")
    
    logger.info(f"Loading scaler from {config.SCALER_PATH}")
    scaler = joblib.load(config.SCALER_PATH)
    logger.info("Scaler loaded successfully")
    
    return scaler


def load_selected_features():
    """
    Load the list of selected features from disk.
    
    Returns:
        list: List of selected feature names
    """
    selected_features_path = os.path.join(config.MODEL_DIR, 'selected_features.joblib')
    
    if not os.path.exists(selected_features_path):
        logger.error(f"Selected features not found at {selected_features_path}")
        raise FileNotFoundError(f"Selected features not found at {selected_features_path}")
    
    logger.info(f"Loading selected features from {selected_features_path}")
    selected_features = joblib.load(selected_features_path)
    logger.info(f"Loaded {len(selected_features)} selected features")
    
    return selected_features


def preprocess_transaction(transaction, scaler=None, selected_features=None):
    """
    Preprocess a single transaction for prediction.
    
    Args:
        transaction (dict): Transaction data
        scaler (object, optional): Feature scaler. Defaults to None (load from disk).
        selected_features (list, optional): List of selected features. 
                                          Defaults to None (load from disk).
    
    Returns:
        pd.DataFrame: Preprocessed transaction data
    """
    logger.info("Preprocessing transaction...")
    
    # Convert transaction to DataFrame
    df = pd.DataFrame([transaction])
    
    # Create a copy to avoid modifying the original dataframe
    X = df.copy()
    
    # Create hour of day feature from Time if it exists
    if 'Time' in X.columns:
        X['Hour'] = (X['Time'] / 3600) % 24
        X = X.drop('Time', axis=1)
    
    # Scale the features if scaler is provided or available
    if scaler is None and os.path.exists(config.SCALER_PATH):
        scaler = load_scaler()
    
    if scaler is not None:
        # Get the column names
        columns = X.columns
        
        # Scale the features
        X = pd.DataFrame(
            scaler.transform(X),
            columns=columns
        )
    
    # Apply feature engineering (simplified version)
    # Create interaction features
    if all(feat in X.columns for feat in ['V4', 'V12', 'V14', 'V17', 'Amount']):
        X['V4_V12_interaction'] = X['V4'] * X['V12']
        X['V4_V14_interaction'] = X['V4'] * X['V14']
        X['V4_V17_interaction'] = X['V4'] * X['V17']
        X['V4_Amount_interaction'] = X['V4'] * X['Amount']
        X['V12_V14_interaction'] = X['V12'] * X['V14']
        X['V12_V17_interaction'] = X['V12'] * X['V17']
        X['V12_Amount_interaction'] = X['V12'] * X['Amount']
        X['V14_V17_interaction'] = X['V14'] * X['V17']
        X['V14_Amount_interaction'] = X['V14'] * X['Amount']
        X['V17_Amount_interaction'] = X['V17'] * X['Amount']
    
    # Create polynomial features
    for feature in ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount']:
        if feature in X.columns:
            X[f"{feature}_pow2"] = X[feature] ** 2
    
    # Select features if provided or available
    if selected_features is None and os.path.exists(os.path.join(config.MODEL_DIR, 'selected_features.joblib')):
        selected_features = load_selected_features()
    
    if selected_features is not None:
        # Keep only the selected features that exist in the dataframe
        existing_features = [f for f in selected_features if f in X.columns]
        X = X[existing_features]
    
    logger.info("Transaction preprocessing completed")
    return X


def predict_transaction(transaction, model=None, scaler=None, selected_features=None, threshold=None):
    """
    Predict whether a transaction is fraudulent.
    
    Args:
        transaction (dict): Transaction data
        model (object, optional): Trained model. Defaults to None (load from disk).
        scaler (object, optional): Feature scaler. Defaults to None (load from disk).
        selected_features (list, optional): List of selected features. 
                                          Defaults to None (load from disk).
        threshold (float, optional): Classification threshold. 
                                    Defaults to None (use threshold from metrics).
    
    Returns:
        dict: Prediction results
    """
    start_time = time.time()
    logger.info("Predicting transaction...")
    
    # Load model if not provided
    if model is None:
        ensemble_path = os.path.join(config.MODEL_DIR, 'ensemble_model.joblib')
        if os.path.exists(ensemble_path):
            model = load_model(ensemble_path)
        else:
            model = load_model()
    
    # Load threshold if not provided
    if threshold is None:
        metrics_path = os.path.join(config.MODEL_DIR, 'metrics.joblib')
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
            threshold = metrics.get('threshold', 0.5)
        else:
            threshold = 0.5
    
    # Preprocess the transaction
    X = preprocess_transaction(transaction, scaler, selected_features)
    
    # Make prediction
    if isinstance(model, dict) and 'models' in model and 'weights' in model:
        # Ensemble model
        weighted_probs = np.zeros(len(X))
        
        for model_type, model_obj in model['models'].items():
            if hasattr(model_obj, 'predict_proba'):
                probs = model_obj.predict_proba(X)[:, 1]
            else:
                probs = model_obj.predict(X)
            
            weighted_probs += probs * model['weights'][model_type]
        
        probability = float(weighted_probs[0])
        prediction = int(probability >= threshold)
    else:
        # Single model
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(X)[0, 1])
            prediction = int(probability >= threshold)
        else:
            prediction = int(model.predict(X)[0])
            probability = float(prediction)
    
    # Create result dictionary
    result = {
        'prediction': prediction,
        'probability': probability,
        'threshold': threshold,
        'is_fraud': bool(prediction == 1),
        'processing_time': time.time() - start_time
    }
    
    logger.info(f"Prediction: {result['is_fraud']} (probability: {result['probability']:.4f})")
    logger.info(f"Processing time: {result['processing_time']:.4f} seconds")
    
    return result


def predict_batch(transactions, batch_size=1000):
    """
    Predict fraud for a batch of transactions with optimized performance.
    
    Args:
        transactions (list): List of transaction dictionaries
        batch_size (int, optional): Batch size for processing. Defaults to 1000.
    
    Returns:
        list: List of prediction results
    """
    logger.info(f"Predicting batch of {len(transactions)} transactions...")
    
    # Load model, scaler, and selected features once
    ensemble_path = os.path.join(config.MODEL_DIR, 'ensemble_model.joblib')
    if os.path.exists(ensemble_path):
        model = load_model(ensemble_path)
    else:
        model = load_model()
    
    scaler = load_scaler()
    selected_features = load_selected_features()
    
    # Load threshold
    metrics_path = os.path.join(config.MODEL_DIR, 'metrics.joblib')
    if os.path.exists(metrics_path):
        metrics = joblib.load(metrics_path)
        threshold = metrics.get('threshold', 0.5)
    else:
        threshold = 0.5
    
    # Process transactions in batches
    results = []
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i+batch_size]
        
        # Preprocess batch
        batch_df = pd.DataFrame(batch)
        
        # Create hour of day feature from Time if it exists
        if 'Time' in batch_df.columns:
            batch_df['Hour'] = (batch_df['Time'] / 3600) % 24
            batch_df = batch_df.drop('Time', axis=1)
        
        # Scale the features
        if scaler is not None:
            # Get the column names
            columns = batch_df.columns
            
            # Scale the features
            batch_df = pd.DataFrame(
                scaler.transform(batch_df),
                columns=columns
            )
        
        # Apply feature engineering (simplified version)
        # Create interaction features
        if all(feat in batch_df.columns for feat in ['V4', 'V12', 'V14', 'V17', 'Amount']):
            batch_df['V4_V12_interaction'] = batch_df['V4'] * batch_df['V12']
            batch_df['V4_V14_interaction'] = batch_df['V4'] * batch_df['V14']
            batch_df['V4_V17_interaction'] = batch_df['V4'] * batch_df['V17']
            batch_df['V4_Amount_interaction'] = batch_df['V4'] * batch_df['Amount']
            batch_df['V12_V14_interaction'] = batch_df['V12'] * batch_df['V14']
            batch_df['V12_V17_interaction'] = batch_df['V12'] * batch_df['V17']
            batch_df['V12_Amount_interaction'] = batch_df['V12'] * batch_df['Amount']
            batch_df['V14_V17_interaction'] = batch_df['V14'] * batch_df['V17']
            batch_df['V14_Amount_interaction'] = batch_df['V14'] * batch_df['Amount']
            batch_df['V17_Amount_interaction'] = batch_df['V17'] * batch_df['Amount']
        
        # Create polynomial features
        for feature in ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount']:
            if feature in batch_df.columns:
                batch_df[f"{feature}_pow2"] = batch_df[feature] ** 2
        
        # Select features
        if selected_features is not None:
            # Keep only the selected features that exist in the dataframe
            existing_features = [f for f in selected_features if f in batch_df.columns]
            batch_df = batch_df[existing_features]
        
        # Make predictions
        if isinstance(model, dict) and 'models' in model and 'weights' in model:
            # Ensemble model
            weighted_probs = np.zeros(len(batch_df))
            
            for model_type, model_obj in model['models'].items():
                if hasattr(model_obj, 'predict_proba'):
                    probs = model_obj.predict_proba(batch_df)[:, 1]
                else:
                    probs = model_obj.predict(batch_df)
                
                weighted_probs += probs * model['weights'][model_type]
            
            probabilities = weighted_probs
            predictions = (probabilities >= threshold).astype(int)
        else:
            # Single model
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(batch_df)[:, 1]
                predictions = (probabilities >= threshold).astype(int)
            else:
                predictions = model.predict(batch_df)
                probabilities = predictions.astype(float)
        
        # Create result dictionaries
        batch_results = [
            {
                'prediction': int(pred),
                'probability': float(prob),
                'threshold': threshold,
                'is_fraud': bool(pred == 1)
            }
            for pred, prob in zip(predictions, probabilities)
        ]
        
        results.extend(batch_results)
    
    logger.info(f"Batch prediction completed. Found {sum(r['is_fraud'] for r in results)} potential frauds.")
    return results


def main():
    """Main function to demonstrate model prediction."""
    logger.info("Starting model prediction demonstration")
    
    # Load the test data
    test_data_path = config.TEST_DATA_PATH
    
    if not os.path.exists(test_data_path):
        logger.error(f"Test data not found at {test_data_path}")
        return
    
    test_data = pd.read_csv(test_data_path)
    
    # Select a few samples for demonstration
    samples = test_data.sample(5).to_dict('records')
    
    # Make predictions
    for i, sample in enumerate(samples):
        logger.info(f"Sample {i+1}:")
        result = predict_transaction(sample)
        logger.info(f"  Actual: {sample.get('Class', 'Unknown')}")
        logger.info(f"  Predicted: {result['prediction']} (probability: {result['probability']:.4f})")
        logger.info(f"  Is fraud: {result['is_fraud']}")
        logger.info(f"  Processing time: {result['processing_time']:.4f} seconds")
    
    logger.info("Model prediction demonstration completed")


if __name__ == "__main__":
    main() 