"""
Model training module for credit card fraud detection.
Includes functions for training, evaluating, and optimizing models.
"""
import os
import pandas as pd
import numpy as np
import joblib
import logging
import sys
import time
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.MODEL_DIR, 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_model(X_train, y_train, model_type='xgboost', params=None, cv=5):
    """
    Train a model with the specified parameters.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        model_type (str, optional): Type of model to train. 
                                   Options: 'xgboost', 'lightgbm', 'random_forest'. 
                                   Defaults to 'xgboost'.
        params (dict, optional): Model parameters. Defaults to None (use default params).
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
    
    Returns:
        object: Trained model
    """
    logger.info(f"Training {model_type} model...")
    start_time = time.time()
    
    # Get default parameters if not provided
    if params is None:
        params = config.MODEL_PARAMS.get(model_type, {})
    
    # Initialize model based on type
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            **params,
            random_state=config.RANDOM_STATE,
            n_jobs=config.N_JOBS,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(
            **params,
            random_state=config.RANDOM_STATE,
            n_jobs=config.N_JOBS
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            **params,
            random_state=config.RANDOM_STATE,
            n_jobs=config.N_JOBS
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    return model


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained model on the test set.
    
    Args:
        model (object): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        threshold (float, optional): Classification threshold. Defaults to 0.5.
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_prob = y_pred  # For models that don't have predict_proba
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    
    # Log the results
    logger.info(f"Evaluation metrics:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            logger.info(f"  {metric}: {value:.4f}")
    
    # Log confusion matrix
    cm = metrics['confusion_matrix']
    logger.info(f"  Confusion Matrix:")
    logger.info(f"    TN: {cm[0][0]}, FP: {cm[0][1]}")
    logger.info(f"    FN: {cm[1][0]}, TP: {cm[1][1]}")
    
    # Log classification report
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return metrics


def optimize_threshold(model, X_val, y_val):
    """
    Find the optimal classification threshold based on F1 score.
    
    Args:
        model (object): Trained model
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation labels
    
    Returns:
        float: Optimal threshold
    """
    logger.info("Optimizing classification threshold...")
    
    # Get predicted probabilities
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # Calculate precision and recall for different thresholds
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find the threshold that maximizes F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    logger.info(f"Optimal threshold: {optimal_threshold:.4f} (F1: {f1_scores[optimal_idx]:.4f})")
    return optimal_threshold


def tune_hyperparameters(X_train, y_train, X_val, y_val, model_type='xgboost', n_iter=20):
    """
    Tune model hyperparameters using randomized search.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation labels
        model_type (str, optional): Type of model to tune. 
                                   Options: 'xgboost', 'lightgbm', 'random_forest'. 
                                   Defaults to 'xgboost'.
        n_iter (int, optional): Number of parameter settings to try. Defaults to 20.
    
    Returns:
        tuple: Best parameters and best model
    """
    logger.info(f"Tuning hyperparameters for {model_type} model...")
    
    # Define parameter grid based on model type
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=config.N_JOBS,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
            'scale_pos_weight': [1, 5, 10, 50, 100]
        }
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=config.N_JOBS
        )
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'num_leaves': [31, 50, 70, 90, 120],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'feature_fraction': [0.6, 0.7, 0.8, 0.9],
            'bagging_fraction': [0.6, 0.7, 0.8, 0.9],
            'bagging_freq': [0, 1, 5],
            'min_child_samples': [5, 10, 20, 30],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0, 0.1, 0.5, 1],
            'scale_pos_weight': [1, 5, 10, 50, 100]
        }
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=config.N_JOBS
        )
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    
    # Set up randomized search
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='f1',
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
        verbose=1
    )
    
    # Fit the search
    search.fit(X_train, y_train)
    
    # Get the best parameters and model
    best_params = search.best_params_
    best_model = search.best_estimator_
    
    # Evaluate the best model on the validation set
    metrics = evaluate_model(best_model, X_val, y_val)
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Validation F1 score: {metrics['f1']:.4f}")
    
    return best_params, best_model


def train_ensemble(X_train, y_train, X_val, y_val):
    """
    Train an ensemble of models for better performance.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation labels
    
    Returns:
        dict: Dictionary of trained models and their weights
    """
    logger.info("Training ensemble of models...")
    
    # Define model types to include in the ensemble
    model_types = ['xgboost', 'lightgbm', 'random_forest']
    
    # Train each model type
    models = {}
    metrics = {}
    
    # Use ProcessPoolExecutor for parallel training
    with ProcessPoolExecutor(max_workers=min(len(model_types), os.cpu_count())) as executor:
        # Define a function to train and evaluate a model
        def train_and_evaluate(model_type):
            # Train the model
            model = train_model(X_train, y_train, model_type=model_type)
            
            # Evaluate the model
            model_metrics = evaluate_model(model, X_val, y_val)
            
            return model_type, model, model_metrics
        
        # Submit training tasks
        futures = [executor.submit(train_and_evaluate, model_type) for model_type in model_types]
        
        # Collect results
        for future in tqdm(futures, desc="Training models"):
            model_type, model, model_metrics = future.result()
            models[model_type] = model
            metrics[model_type] = model_metrics
    
    # Calculate weights based on F1 scores
    weights = {}
    total_f1 = sum(m['f1'] for m in metrics.values())
    
    for model_type, model_metrics in metrics.items():
        weights[model_type] = model_metrics['f1'] / total_f1
        logger.info(f"{model_type} weight: {weights[model_type]:.4f} (F1: {model_metrics['f1']:.4f})")
    
    # Create ensemble dictionary
    ensemble = {
        'models': models,
        'weights': weights
    }
    
    return ensemble


def ensemble_predict(ensemble, X, threshold=0.5):
    """
    Make predictions using an ensemble of models.
    
    Args:
        ensemble (dict): Ensemble dictionary with models and weights
        X (pd.DataFrame): Features to predict on
        threshold (float, optional): Classification threshold. Defaults to 0.5.
    
    Returns:
        tuple: Predicted probabilities and predicted classes
    """
    # Get models and weights
    models = ensemble['models']
    weights = ensemble['weights']
    
    # Initialize weighted probabilities
    weighted_probs = np.zeros(len(X))
    
    # Calculate weighted probabilities
    for model_type, model in models.items():
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.predict(X)
        
        weighted_probs += probs * weights[model_type]
    
    # Convert probabilities to class predictions
    y_pred = (weighted_probs >= threshold).astype(int)
    
    return weighted_probs, y_pred


def save_model(model, model_path=None):
    """
    Save a trained model to disk.
    
    Args:
        model (object): Trained model to save
        model_path (str, optional): Path to save the model. Defaults to None (use config.BEST_MODEL_PATH).
    """
    model_path = model_path or config.BEST_MODEL_PATH
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")


def plot_learning_curve(model, X_train, y_train, cv=5, n_jobs=None):
    """
    Plot the learning curve for a trained model.
    
    Args:
        model (object): Trained model
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to None (use config.N_JOBS).
    """
    from sklearn.model_selection import learning_curve
    
    logger.info("Plotting learning curve...")
    
    # Set number of jobs
    n_jobs = n_jobs or config.N_JOBS
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1'
    )
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.title('Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel('F1 Score')
    plt.grid()
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')
    
    plt.legend(loc='best')
    
    # Save the plot
    os.makedirs(os.path.join(config.MODEL_DIR, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(config.MODEL_DIR, 'plots', 'learning_curve.png'))
    plt.close()
    
    logger.info(f"Learning curve saved to {os.path.join(config.MODEL_DIR, 'plots', 'learning_curve.png')}")


def main():
    """Main function to run the model training pipeline."""
    logger.info("Starting model training pipeline")
    
    # Load the engineered data
    engineered_train_path = os.path.join(config.PROCESSED_DATA_DIR, 'engineered_train.csv')
    engineered_test_path = os.path.join(config.PROCESSED_DATA_DIR, 'engineered_test.csv')
    
    if not os.path.exists(engineered_train_path) or not os.path.exists(engineered_test_path):
        logger.error("Engineered data not found. Please run feature engineering first.")
        return
    
    train_data = pd.read_csv(engineered_train_path)
    test_data = pd.read_csv(engineered_test_path)
    
    # Split into features and target
    X_train = train_data.drop('Class', axis=1)
    y_train = train_data['Class']
    X_test = test_data.drop('Class', axis=1)
    y_test = test_data['Class']
    
    # Split training data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=config.VALIDATION_SIZE, 
        random_state=config.RANDOM_STATE, stratify=y_train
    )
    
    # Train an ensemble of models
    ensemble = train_ensemble(X_train_split, y_train_split, X_val, y_val)
    
    # Find the optimal threshold
    weighted_probs_val, _ = ensemble_predict(ensemble, X_val)
    
    # Calculate precision and recall for different thresholds
    precision, recall, thresholds = precision_recall_curve(y_val, weighted_probs_val)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find the threshold that maximizes F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    logger.info(f"Optimal threshold: {optimal_threshold:.4f} (F1: {f1_scores[optimal_idx]:.4f})")
    
    # Evaluate the ensemble on the test set
    weighted_probs_test, y_pred_test = ensemble_predict(ensemble, X_test, threshold=optimal_threshold)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test),
        'f1': f1_score(y_test, y_pred_test),
        'roc_auc': roc_auc_score(y_test, weighted_probs_test),
        'avg_precision': average_precision_score(y_test, weighted_probs_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
        'threshold': optimal_threshold
    }
    
    # Log the results
    logger.info(f"Ensemble evaluation metrics:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            logger.info(f"  {metric}: {value:.4f}")
    
    # Log confusion matrix
    cm = metrics['confusion_matrix']
    logger.info(f"  Confusion Matrix:")
    logger.info(f"    TN: {cm[0][0]}, FP: {cm[0][1]}")
    logger.info(f"    FN: {cm[1][0]}, TP: {cm[1][1]}")
    
    # Save the ensemble model
    save_model(ensemble, os.path.join(config.MODEL_DIR, 'ensemble_model.joblib'))
    
    # Save the metrics
    metrics_path = os.path.join(config.MODEL_DIR, 'metrics.joblib')
    joblib.dump(metrics, metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, weighted_probs_test)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    
    # Save the plot
    os.makedirs(os.path.join(config.MODEL_DIR, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(config.MODEL_DIR, 'plots', 'roc_curve.png'))
    plt.close()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {metrics["avg_precision"]:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    
    # Save the plot
    plt.savefig(os.path.join(config.MODEL_DIR, 'plots', 'pr_curve.png'))
    plt.close()
    
    logger.info("Model training pipeline completed successfully")


if __name__ == "__main__":
    main() 