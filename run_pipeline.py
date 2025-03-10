#!/usr/bin/env python
"""
Script to run the entire credit card fraud detection pipeline.
"""
import os
import sys
import logging
import argparse
import time
from datetime import datetime

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_data_preprocessing():
    """Run the data preprocessing step."""
    logger.info("Starting data preprocessing...")
    start_time = time.time()
    
    try:
        from src.data.preprocess import main as preprocess_main
        preprocess_main()
        logger.info(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        return False


def run_feature_engineering():
    """Run the feature engineering step."""
    logger.info("Starting feature engineering...")
    start_time = time.time()
    
    try:
        from src.features.feature_engineering import main as feature_engineering_main
        feature_engineering_main()
        logger.info(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        return False


def run_model_training():
    """Run the model training step."""
    logger.info("Starting model training...")
    start_time = time.time()
    
    try:
        from src.models.train_model import main as train_model_main
        train_model_main()
        logger.info(f"Model training completed in {time.time() - start_time:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        return False


def run_model_prediction_test():
    """Run a test of the model prediction."""
    logger.info("Testing model prediction...")
    
    try:
        from src.models.predict_model import main as predict_model_main
        predict_model_main()
        logger.info("Model prediction test completed")
        return True
    except Exception as e:
        logger.error(f"Error in model prediction test: {str(e)}")
        return False


def run_web_app():
    """Run the web application."""
    logger.info("Starting web application...")
    
    try:
        import app
        app.app.run_server(debug=False)
        return True
    except Exception as e:
        logger.error(f"Error starting web application: {str(e)}")
        return False


def run_full_pipeline():
    """Run the full pipeline from data preprocessing to model training."""
    logger.info("Starting full pipeline...")
    start_time = time.time()
    
    # Run data preprocessing
    if not run_data_preprocessing():
        logger.error("Pipeline stopped due to error in data preprocessing")
        return False
    
    # Run feature engineering
    if not run_feature_engineering():
        logger.error("Pipeline stopped due to error in feature engineering")
        return False
    
    # Run model training
    if not run_model_training():
        logger.error("Pipeline stopped due to error in model training")
        return False
    
    # Run model prediction test
    if not run_model_prediction_test():
        logger.warning("Model prediction test failed, but continuing pipeline")
    
    logger.info(f"Full pipeline completed in {time.time() - start_time:.2f} seconds")
    return True


def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description='Run the credit card fraud detection pipeline.')
    parser.add_argument('--step', type=str, choices=['preprocess', 'features', 'train', 'predict', 'webapp', 'all'],
                        default='all', help='Pipeline step to run')
    
    args = parser.parse_args()
    
    if args.step == 'preprocess':
        run_data_preprocessing()
    elif args.step == 'features':
        run_feature_engineering()
    elif args.step == 'train':
        run_model_training()
    elif args.step == 'predict':
        run_model_prediction_test()
    elif args.step == 'webapp':
        run_web_app()
    elif args.step == 'all':
        if run_full_pipeline():
            logger.info("Full pipeline completed successfully")
            logger.info("Starting web application...")
            run_web_app()
        else:
            logger.error("Full pipeline failed")
            sys.exit(1)


if __name__ == "__main__":
    main() 