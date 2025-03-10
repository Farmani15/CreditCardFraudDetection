"""
Web application for credit card fraud detection.
"""
import os
import pandas as pd
import numpy as np
import joblib
import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime
import random
import time
import logging

# Import project modules
import config
from src.models.predict_model import predict_transaction, load_model, load_scaler, load_selected_features

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load model, scaler, and selected features
try:
    ensemble_path = os.path.join(config.MODEL_DIR, 'ensemble_model.joblib')
    if os.path.exists(ensemble_path):
        model = load_model(ensemble_path)
    else:
        model = load_model()
    
    scaler = load_scaler()
    selected_features = load_selected_features()
    
    # Load metrics
    metrics_path = os.path.join(config.MODEL_DIR, 'metrics.joblib')
    if os.path.exists(metrics_path):
        metrics = joblib.load(metrics_path)
    else:
        metrics = {}
    
    model_loaded = True
    logger.info("Model and dependencies loaded successfully")
except Exception as e:
    model_loaded = False
    logger.error(f"Error loading model: {str(e)}")
    metrics = {}

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "Credit Card Fraud Detection"
server = app.server

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Credit Card Fraud Detection System", className="text-center my-4"),
            html.Hr(),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Model Performance", className="card-title")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Accuracy", className="card-title text-center"),
                                    html.H3(f"{metrics.get('accuracy', 0):.4f}", className="text-center text-primary")
                                ])
                            ], className="mb-4")
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Precision", className="card-title text-center"),
                                    html.H3(f"{metrics.get('precision', 0):.4f}", className="text-center text-success")
                                ])
                            ], className="mb-4")
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Recall", className="card-title text-center"),
                                    html.H3(f"{metrics.get('recall', 0):.4f}", className="text-center text-info")
                                ])
                            ], className="mb-4")
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("F1 Score", className="card-title text-center"),
                                    html.H3(f"{metrics.get('f1', 0):.4f}", className="text-center text-warning")
                                ])
                            ], className="mb-4")
                        ], width=3)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("ROC AUC", className="card-title text-center"),
                                    html.H3(f"{metrics.get('roc_auc', 0):.4f}", className="text-center text-danger")
                                ])
                            ], className="mb-4")
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Average Precision", className="card-title text-center"),
                                    html.H3(f"{metrics.get('avg_precision', 0):.4f}", className="text-center text-secondary")
                                ])
                            ], className="mb-4")
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Threshold", className="card-title text-center"),
                                    html.H3(f"{metrics.get('threshold', 0.5):.4f}", className="text-center text-dark")
                                ])
                            ], className="mb-4")
                        ], width=4)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Img(src="assets/roc_curve.png" if os.path.exists(os.path.join(config.MODEL_DIR, 'plots', 'roc_curve.png')) else "", 
                                        className="img-fluid")
                            ], id="roc-curve-container", className="text-center")
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.Img(src="assets/pr_curve.png" if os.path.exists(os.path.join(config.MODEL_DIR, 'plots', 'pr_curve.png')) else "", 
                                        className="img-fluid")
                            ], id="pr-curve-container", className="text-center")
                        ], width=6)
                    ])
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Transaction Fraud Detection", className="card-title")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Amount"),
                            dbc.Input(id="amount-input", type="number", placeholder="Enter transaction amount", value=100.0),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Generate Random Features"),
                            dbc.Button("Generate", id="generate-button", color="primary", className="w-100"),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Predict"),
                            dbc.Button("Predict", id="predict-button", color="success", className="w-100"),
                        ], width=4)
                    ], className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="features-container", className="mb-4", style={"maxHeight": "300px", "overflow": "auto"})
                        ], width=12)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Spinner(html.Div(id="prediction-result"))
                        ], width=12)
                    ])
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Recent Predictions", className="card-title")),
                dbc.CardBody([
                    html.Div(id="recent-predictions-container")
                ])
            ])
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Credit Card Fraud Detection System © 2023", className="text-center text-muted")
        ], width=12)
    ])
], fluid=True)


# Define callback to generate random features
@app.callback(
    Output("features-container", "children"),
    Input("generate-button", "n_clicks"),
    State("amount-input", "value"),
    prevent_initial_call=True
)
def generate_random_features(n_clicks, amount):
    if n_clicks is None:
        return dash.no_update
    
    # Generate random features
    features = {
        'Time': random.randint(0, 172800),  # Random time within 2 days (in seconds)
        'Amount': amount or random.uniform(1, 1000)
    }
    
    # Generate V1-V28 features
    for i in range(1, 29):
        features[f'V{i}'] = random.uniform(-5, 5)
    
    # Create a DataFrame for display
    df = pd.DataFrame([features])
    
    # Create a data table
    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '5px',
            'minWidth': '100px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    )
    
    return table


# Store for recent predictions
recent_predictions = []

# Define callback to make predictions
@app.callback(
    [Output("prediction-result", "children"),
     Output("recent-predictions-container", "children")],
    Input("predict-button", "n_clicks"),
    State("features-container", "children"),
    prevent_initial_call=True
)
def make_prediction(n_clicks, features_table):
    if n_clicks is None or features_table is None or not model_loaded:
        return dash.no_update, dash.no_update
    
    # Extract features from the table
    if isinstance(features_table, dict) and 'props' in features_table:
        data = features_table['props'].get('data', [])
        if data:
            transaction = data[0]
            
            # Make prediction
            try:
                start_time = time.time()
                result = predict_transaction(transaction, model, scaler, selected_features)
                processing_time = time.time() - start_time
                
                # Create result card
                if result['is_fraud']:
                    alert_color = "danger"
                    alert_heading = "Fraud Detected!"
                else:
                    alert_color = "success"
                    alert_heading = "Transaction Legitimate"
                
                result_card = dbc.Alert(
                    [
                        html.H4(alert_heading, className="alert-heading"),
                        html.P(f"Fraud Probability: {result['probability']:.4f}"),
                        html.P(f"Threshold: {result['threshold']:.4f}"),
                        html.P(f"Processing Time: {processing_time:.4f} seconds"),
                        html.Hr(),
                        html.P(f"Transaction Amount: ${transaction.get('Amount', 0):.2f}", className="mb-0")
                    ],
                    color=alert_color,
                    className="mt-3"
                )
                
                # Add to recent predictions
                global recent_predictions
                recent_predictions.insert(0, {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'amount': transaction.get('Amount', 0),
                    'is_fraud': result['is_fraud'],
                    'probability': result['probability'],
                    'processing_time': processing_time
                })
                
                # Keep only the 10 most recent predictions
                recent_predictions = recent_predictions[:10]
                
                # Create recent predictions table
                recent_table = dash_table.DataTable(
                    data=recent_predictions,
                    columns=[
                        {'name': 'Timestamp', 'id': 'timestamp'},
                        {'name': 'Amount', 'id': 'amount', 'type': 'numeric', 'format': {'specifier': '$.2f'}},
                        {'name': 'Fraud', 'id': 'is_fraud', 'type': 'boolean'},
                        {'name': 'Probability', 'id': 'probability', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                        {'name': 'Processing Time (s)', 'id': 'processing_time', 'type': 'numeric', 'format': {'specifier': '.4f'}}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '5px'
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{is_fraud} eq true'},
                            'backgroundColor': 'rgba(255, 0, 0, 0.1)'
                        }
                    ]
                )
                
                return result_card, recent_table
                
            except Exception as e:
                logger.error(f"Error making prediction: {str(e)}")
                error_alert = dbc.Alert(
                    f"Error making prediction: {str(e)}",
                    color="danger",
                    className="mt-3"
                )
                return error_alert, dash.no_update
    
    return dash.no_update, dash.no_update


# Run the app
if __name__ == "__main__":
    # Create assets directory for images
    os.makedirs("assets", exist_ok=True)
    
    # Copy plot images to assets directory if they exist
    roc_curve_path = os.path.join(config.MODEL_DIR, 'plots', 'roc_curve.png')
    pr_curve_path = os.path.join(config.MODEL_DIR, 'plots', 'pr_curve.png')
    
    if os.path.exists(roc_curve_path):
        import shutil
        shutil.copy(roc_curve_path, "assets/roc_curve.png")
    
    if os.path.exists(pr_curve_path):
        import shutil
        shutil.copy(pr_curve_path, "assets/pr_curve.png")
    
    app.run_server(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT) 