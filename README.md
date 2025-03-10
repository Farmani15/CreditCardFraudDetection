# Credit Card Fraud Detection System

A machine learning-based system for detecting fraudulent credit card transactions with high accuracy and performance optimization.

## Features

- **Data Preprocessing**: Handles imbalanced data and prepares it for model training
- **Feature Engineering**: Creates relevant features to improve model performance
- **Model Training**: Implements multiple ML algorithms with hyperparameter tuning
- **Performance Optimization**: Uses techniques to improve both accuracy and computational efficiency
- **Web Interface**: Provides an interactive dashboard for visualizing results and testing the model

## Project Structure

```
├── data/                  # Data directory (download dataset separately)
├── models/                # Saved trained models
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── data/              # Data processing modules
│   ├── features/          # Feature engineering modules
│   ├── models/            # Model training and evaluation
│   ├── visualization/     # Data visualization utilities
│   └── web/               # Web application
├── tests/                 # Unit tests
├── app.py                 # Legacy web application
├── webapp.py              # Enhanced web application
├── run_pipeline.py        # Pipeline execution script
├── config.py              # Configuration settings
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download the dataset (see instructions in data/README.md)

## Usage

### Running the Pipeline

```bash
# Run the full pipeline (preprocessing, feature engineering, model training)
python run_pipeline.py --step all

# Run individual steps
python run_pipeline.py --step preprocess
python run_pipeline.py --step features
python run_pipeline.py --step train
python run_pipeline.py --step predict
```

### Running the Web Application

```bash
# Run the enhanced web application
python webapp.py

# Run with debug mode
python webapp.py --debug

# Run on a specific port
python webapp.py --port 8080
```

Then open your browser to http://localhost:8050 (or the port you specified)

### Running Tests

```bash
python -m unittest discover tests
```

## Web Application

The system includes a modern, responsive web application with the following features:

- **Interactive Dashboard**: View model performance metrics and visualizations
- **Real-time Fraud Detection**: Test the model with transaction data
- **Analytics**: Visualize fraud patterns and trends
- **Prediction History**: Track and review previous predictions

The web application is built using:
- Dash and Plotly for interactive visualizations
- Bootstrap for responsive design
- Custom CSS and JavaScript for enhanced user experience

## Performance Optimizations

This project implements several performance optimizations:
- Feature selection to reduce dimensionality
- Efficient data preprocessing pipeline
- Model ensembling for improved accuracy
- Parallel processing for faster training
- Memory optimization for handling large datasets
- Optimized web application for responsive user experience

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle, which contains transactions made by credit cards in September 2013 by European cardholders.

## License

MIT 