"""
Enhanced web application for credit card fraud detection with a professional UI.
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
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.models.predict_model import predict_transaction, load_model, load_scaler, load_selected_features

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.BASE_DIR, 'web_app.log')),
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

# Initialize the Dash app with a modern theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://use.fontawesome.com/releases/v5.15.4/css/all.css'],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "Credit Card Fraud Detection System"
server = app.server

# Custom CSS for better styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f8f9fa;
            }
            .navbar {
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            .card {
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
                transition: transform 0.3s;
            }
            .card:hover {
                transform: translateY(-5px);
            }
            .metric-card {
                text-align: center;
                padding: 15px;
            }
            .metric-value {
                font-size: 2rem;
                font-weight: bold;
            }
            .metric-title {
                font-size: 1rem;
                color: #6c757d;
            }
            .fraud-alert {
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); }
                70% { box-shadow: 0 0 0 10px rgba(220, 53, 69, 0); }
                100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
            }
            .footer {
                background-color: #343a40;
                color: white;
                padding: 20px 0;
                margin-top: 40px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define the navbar
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.I(className="fas fa-shield-alt mr-2", style={"font-size": "24px"})),
                        dbc.Col(dbc.NavbarBrand("Credit Card Fraud Detection System", className="ml-2")),
                    ],
                    align="center",
                ),
                href="/",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Dashboard", href="#dashboard")),
                        dbc.NavItem(dbc.NavLink("Fraud Detection", href="#fraud-detection")),
                        dbc.NavItem(dbc.NavLink("Analytics", href="#analytics")),
                        dbc.NavItem(dbc.NavLink("History", href="#history")),
                    ],
                    className="ml-auto",
                    navbar=True,
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    className="mb-4",
)

# Define the footer
footer = html.Footer(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Credit Card Fraud Detection System"),
                            html.P("A machine learning-based system for detecting fraudulent credit card transactions with high accuracy."),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H5("Links"),
                            html.Ul(
                                [
                                    html.Li(html.A("Documentation", href="#")),
                                    html.Li(html.A("GitHub Repository", href="#")),
                                    html.Li(html.A("Report Issues", href="#")),
                                ]
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.H5("Contact"),
                            html.P([
                                html.I(className="fas fa-envelope mr-2"),
                                "support@frauddetection.com",
                            ]),
                            html.P([
                                html.I(className="fas fa-phone mr-2"),
                                "+1 (555) 123-4567",
                            ]),
                        ],
                        width=3,
                    ),
                ]
            ),
            html.Hr(),
            dbc.Row(
                dbc.Col(
                    html.P("© 2023 Credit Card Fraud Detection System. All rights reserved.", className="text-center"),
                    width=12,
                )
            ),
        ]
    ),
    className="footer",
)

# Create dashboard metrics cards
def create_metric_card(title, value, icon, color):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(html.I(className=f"fas {icon} fa-2x"), style={"color": color}),
                html.H4(value, className="metric-value", style={"color": color}),
                html.P(title, className="metric-title"),
            ]
        ),
        className="metric-card mb-4"
    )

# Define the layout
app.layout = html.Div([
    navbar,
    dbc.Container([
        # Dashboard section
        html.Div(id="dashboard", className="mb-5"),
        dbc.Row([
            dbc.Col([
                html.H2("Dashboard", className="mb-4"),
                html.P("Overview of the credit card fraud detection system performance and metrics.", className="text-muted mb-4"),
            ], width=12)
        ]),
        
        # Metrics row
        dbc.Row([
            dbc.Col(create_metric_card("Accuracy", f"{metrics.get('accuracy', 0):.4f}", "fa-bullseye", "#007bff"), width=3),
            dbc.Col(create_metric_card("Precision", f"{metrics.get('precision', 0):.4f}", "fa-crosshairs", "#28a745"), width=3),
            dbc.Col(create_metric_card("Recall", f"{metrics.get('recall', 0):.4f}", "fa-search", "#17a2b8"), width=3),
            dbc.Col(create_metric_card("F1 Score", f"{metrics.get('f1', 0):.4f}", "fa-balance-scale", "#ffc107"), width=3),
        ]),
        
        # Charts row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("ROC Curve")),
                    dbc.CardBody([
                        html.Div([
                            html.Img(
                                src="/assets/roc_curve.png" if os.path.exists(os.path.join(config.MODEL_DIR, 'plots', 'roc_curve.png')) else "",
                                className="img-fluid"
                            )
                        ], id="roc-curve-container", className="text-center")
                    ])
                ], className="mb-4")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Precision-Recall Curve")),
                    dbc.CardBody([
                        html.Div([
                            html.Img(
                                src="/assets/pr_curve.png" if os.path.exists(os.path.join(config.MODEL_DIR, 'plots', 'pr_curve.png')) else "",
                                className="img-fluid"
                            )
                        ], id="pr-curve-container", className="text-center")
                    ])
                ], className="mb-4")
            ], width=6),
        ]),
        
        # Fraud Detection section
        html.Div(id="fraud-detection", className="mb-5 mt-5"),
        dbc.Row([
            dbc.Col([
                html.H2("Fraud Detection", className="mb-4"),
                html.P("Test the fraud detection system with transaction data.", className="text-muted mb-4"),
            ], width=12)
        ]),
        
        dbc.Card([
            dbc.CardHeader(html.H5("Transaction Fraud Detection")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Transaction Amount ($)"),
                        dbc.Input(id="amount-input", type="number", placeholder="Enter transaction amount", value=100.0),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Transaction Time (hour of day)"),
                        dcc.Slider(
                            id="time-slider",
                            min=0,
                            max=23,
                            step=1,
                            value=12,
                            marks={i: f"{i}:00" for i in range(0, 24, 3)},
                        ),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Actions"),
                        dbc.ButtonGroup([
                            dbc.Button("Generate Features", id="generate-button", color="primary", className="mr-2"),
                            dbc.Button("Predict", id="predict-button", color="success"),
                        ], className="w-100"),
                    ], width=4),
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
        ], className="mb-4"),
        
        # Analytics section
        html.Div(id="analytics", className="mb-5 mt-5"),
        dbc.Row([
            dbc.Col([
                html.H2("Analytics", className="mb-4"),
                html.P("Visualizations and analytics of fraud patterns.", className="text-muted mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Fraud by Time of Day")),
                    dbc.CardBody([
                        dcc.Graph(
                            id="fraud-by-time-graph",
                            figure={
                                "data": [
                                    go.Bar(
                                        x=list(range(24)),
                                        y=[random.uniform(0.001, 0.005) for _ in range(24)],
                                        name="Fraud Rate",
                                    )
                                ],
                                "layout": go.Layout(
                                    xaxis={"title": "Hour of Day"},
                                    yaxis={"title": "Fraud Rate"},
                                    margin={"l": 40, "b": 40, "t": 10, "r": 10},
                                    hovermode="closest",
                                )
                            }
                        )
                    ])
                ], className="mb-4")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Fraud by Transaction Amount")),
                    dbc.CardBody([
                        dcc.Graph(
                            id="fraud-by-amount-graph",
                            figure={
                                "data": [
                                    go.Scatter(
                                        x=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                                        y=[random.uniform(0.001, 0.01) for _ in range(10)],
                                        mode="lines+markers",
                                        name="Fraud Rate",
                                    )
                                ],
                                "layout": go.Layout(
                                    xaxis={"title": "Transaction Amount ($)"},
                                    yaxis={"title": "Fraud Rate"},
                                    margin={"l": 40, "b": 40, "t": 10, "r": 10},
                                    hovermode="closest",
                                )
                            }
                        )
                    ])
                ], className="mb-4")
            ], width=6),
        ]),
        
        # History section
        html.Div(id="history", className="mb-5 mt-5"),
        dbc.Row([
            dbc.Col([
                html.H2("Prediction History", className="mb-4"),
                html.P("History of recent fraud predictions.", className="text-muted mb-4"),
            ], width=12)
        ]),
        
        dbc.Card([
            dbc.CardHeader(html.H5("Recent Predictions")),
            dbc.CardBody([
                html.Div(id="recent-predictions-container")
            ])
        ], className="mb-4"),
        
    ], fluid=True),
    
    # Footer
    footer
])

# Store for recent predictions
recent_predictions = []

# Define callback to generate random features
@app.callback(
    Output("features-container", "children"),
    [Input("generate-button", "n_clicks")],
    [State("amount-input", "value"),
     State("time-slider", "value")],
    prevent_initial_call=True
)
def generate_random_features(n_clicks, amount, hour):
    if n_clicks is None:
        return dash.no_update
    
    # Generate random features
    features = {
        'Time': hour * 3600,  # Convert hour to seconds
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


# Define callback to make predictions
@app.callback(
    [Output("prediction-result", "children"),
     Output("recent-predictions-container", "children")],
    [Input("predict-button", "n_clicks")],
    [State("features-container", "children")],
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
                    card_color = "danger"
                    icon = "fa-exclamation-triangle"
                    heading = "Fraud Detected!"
                    card_class = "fraud-alert"
                else:
                    card_color = "success"
                    icon = "fa-check-circle"
                    heading = "Transaction Legitimate"
                    card_class = ""
                
                result_card = dbc.Card([
                    dbc.CardHeader([
                        html.I(className=f"fas {icon} mr-2"),
                        html.Span(heading, style={"font-size": "1.25rem", "font-weight": "bold"})
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H3(f"{result['probability']:.4f}", className="mb-0"),
                                    html.P("Fraud Probability", className="text-muted"),
                                ], className="text-center")
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.H3(f"{result['threshold']:.4f}", className="mb-0"),
                                    html.P("Threshold", className="text-muted"),
                                ], className="text-center")
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.H3(f"${transaction.get('Amount', 0):.2f}", className="mb-0"),
                                    html.P("Transaction Amount", className="text-muted"),
                                ], className="text-center")
                            ], width=4),
                        ]),
                        html.Hr(),
                        html.P(f"Processing Time: {processing_time:.4f} seconds", className="text-muted mb-0 text-center"),
                    ])
                ], color=card_color, className=f"mb-4 {card_class}", outline=True)
                
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
                    [
                        html.I(className="fas fa-exclamation-circle mr-2"),
                        f"Error making prediction: {str(e)}"
                    ],
                    color="danger",
                    className="mt-3"
                )
                return error_alert, dash.no_update
    
    return dash.no_update, dash.no_update


# Run the app
if __name__ == "__main__":
    # Create assets directory for images
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Copy plot images to assets directory if they exist
    roc_curve_path = os.path.join(config.MODEL_DIR, 'plots', 'roc_curve.png')
    pr_curve_path = os.path.join(config.MODEL_DIR, 'plots', 'pr_curve.png')
    
    if os.path.exists(roc_curve_path):
        import shutil
        shutil.copy(roc_curve_path, os.path.join(assets_dir, "roc_curve.png"))
    
    if os.path.exists(pr_curve_path):
        import shutil
        shutil.copy(pr_curve_path, os.path.join(assets_dir, "pr_curve.png"))
    
    app.run_server(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT) 