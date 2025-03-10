#!/usr/bin/env python
"""
Main entry point for the Credit Card Fraud Detection web application.
"""
import os
import sys
import logging
import argparse
from datetime import datetime

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'webapp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def check_model_availability():
    """Check if the trained model is available."""
    ensemble_path = os.path.join(config.MODEL_DIR, 'ensemble_model.joblib')
    best_model_path = config.BEST_MODEL_PATH
    
    if os.path.exists(ensemble_path):
        logger.info(f"Ensemble model found at {ensemble_path}")
        return True
    elif os.path.exists(best_model_path):
        logger.info(f"Best model found at {best_model_path}")
        return True
    else:
        logger.warning("No trained model found. Please run the training pipeline first.")
        return False


def check_data_availability():
    """Check if the processed data is available."""
    train_data_path = config.TRAIN_DATA_PATH
    test_data_path = config.TEST_DATA_PATH
    
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        logger.info("Processed data found.")
        return True
    else:
        logger.warning("Processed data not found. Please run the data preprocessing pipeline first.")
        return False


def create_assets_directory():
    """Create the assets directory for the web application."""
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Copy plot images to assets directory if they exist
    plots_dir = os.path.join(config.MODEL_DIR, 'plots')
    if os.path.exists(plots_dir):
        import shutil
        
        for plot_file in ['roc_curve.png', 'pr_curve.png']:
            plot_path = os.path.join(plots_dir, plot_file)
            if os.path.exists(plot_path):
                shutil.copy(plot_path, os.path.join(assets_dir, plot_file))
                logger.info(f"Copied {plot_file} to assets directory.")
    
    return assets_dir


def run_web_app(debug=False):
    """Run the web application."""
    logger.info("Starting web application...")
    
    # Check if model and data are available
    model_available = check_model_availability()
    data_available = check_data_availability()
    
    if not model_available or not data_available:
        logger.warning("Running with limited functionality due to missing model or data.")
    
    # Create assets directory
    create_assets_directory()
    
    # Import and run the web application
    try:
        from src.web.app import app
        
        # Set debug mode
        debug_mode = debug or config.FLASK_DEBUG
        
        # Run the app
        app.run_server(
            debug=debug_mode,
            host=config.FLASK_HOST,
            port=config.FLASK_PORT
        )
        
        return True
    except Exception as e:
        logger.error(f"Error starting web application: {str(e)}")
        return False


def main():
    """Main function to parse arguments and run the web application."""
    parser = argparse.ArgumentParser(description='Run the Credit Card Fraud Detection web application.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--port', type=int, default=None, help='Port to run the web application on')
    
    args = parser.parse_args()
    
    # Override port if specified
    if args.port is not None:
        config.FLASK_PORT = args.port
    
    # Run the web application
    success = run_web_app(debug=args.debug)
    
    if not success:
        logger.error("Web application failed to start.")
        sys.exit(1)


if __name__ == "__main__":
    main() 