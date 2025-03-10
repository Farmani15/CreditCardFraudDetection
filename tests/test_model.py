"""
Tests for the credit card fraud detection model.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data.preprocess import preprocess_data
from src.features.feature_engineering import engineer_features
from src.models.train_model import train_model, evaluate_model
from src.models.predict_model import predict_transaction


class TestModel(unittest.TestCase):
    """Test cases for the credit card fraud detection model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create a small synthetic dataset for testing
        np.random.seed(config.RANDOM_STATE)
        
        # Create features
        n_samples = 1000
        n_features = 28
        
        # Generate random features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target variable with imbalanced classes (1% fraud)
        y = np.zeros(n_samples)
        fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
        y[fraud_indices] = 1
        
        # Create a DataFrame
        feature_names = [f'V{i+1}' for i in range(n_features)]
        cls.df = pd.DataFrame(X, columns=feature_names)
        cls.df['Time'] = np.random.randint(0, 172800, size=n_samples)  # Random time within 2 days
        cls.df['Amount'] = np.random.uniform(1, 1000, size=n_samples)  # Random amount
        cls.df['Class'] = y
    
    def test_data_preprocessing(self):
        """Test data preprocessing."""
        # Preprocess the data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(self.df)
        
        # Check shapes
        self.assertEqual(len(X_train) + len(X_test), len(self.df))
        self.assertEqual(len(y_train) + len(y_test), len(self.df))
        
        # Check that 'Time' column was dropped and 'Hour' was added
        self.assertNotIn('Time', X_train.columns)
        self.assertIn('Hour', X_train.columns)
        
        # Check that scaler was created
        self.assertIsNotNone(scaler)
    
    def test_feature_engineering(self):
        """Test feature engineering."""
        # Preprocess the data
        X_train, X_test, y_train, y_test, _ = preprocess_data(self.df)
        
        # Apply feature engineering
        X_train_engineered, X_test_engineered, selected_features = engineer_features(X_train, y_train, X_test)
        
        # Check that feature engineering added new features
        self.assertGreater(len(X_train_engineered.columns), 0)
        self.assertGreater(len(X_test_engineered.columns), 0)
        
        # Check that selected features were returned
        self.assertIsNotNone(selected_features)
        self.assertGreater(len(selected_features), 0)
    
    def test_model_training(self):
        """Test model training."""
        # Preprocess the data
        X_train, X_test, y_train, y_test, _ = preprocess_data(self.df)
        
        # Train a model
        model = train_model(X_train, y_train, model_type='random_forest')
        
        # Check that model was created
        self.assertIsNotNone(model)
        
        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Check that metrics were calculated
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('roc_auc', metrics)
        self.assertIn('confusion_matrix', metrics)
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Preprocess the data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(self.df)
        
        # Train a model
        model = train_model(X_train, y_train, model_type='random_forest')
        
        # Create a sample transaction
        transaction = X_test.iloc[0].to_dict()
        
        # Make a prediction
        result = predict_transaction(transaction, model, scaler)
        
        # Check that prediction was made
        self.assertIn('prediction', result)
        self.assertIn('probability', result)
        self.assertIn('is_fraud', result)
        self.assertIn('processing_time', result)
        
        # Check that prediction is binary
        self.assertIn(result['prediction'], [0, 1])
        
        # Check that probability is between 0 and 1
        self.assertGreaterEqual(result['probability'], 0)
        self.assertLessEqual(result['probability'], 1)


if __name__ == '__main__':
    unittest.main() 