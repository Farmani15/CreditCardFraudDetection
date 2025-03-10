#!/bin/bash

# Run the enhanced web application for Credit Card Fraud Detection

# Create necessary directories
mkdir -p data models/plots logs assets

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if the dataset exists
if [ ! -f "data/creditcard.csv" ]; then
    echo "Dataset not found. Please download it first."
    echo "See instructions in data/README.md"
    echo ""
    echo "You can still run the application, but some features may not work."
fi

# Check if the model exists
if [ ! -f "models/ensemble_model.joblib" ] && [ ! -f "models/best_model.joblib" ]; then
    echo "No trained model found. You may want to run the training pipeline first:"
    echo "python run_pipeline.py --step all"
    echo ""
    echo "You can still run the application, but prediction functionality will be limited."
fi

# Run the web application
echo "Starting the web application..."
python webapp.py "$@" 