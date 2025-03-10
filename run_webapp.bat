@echo off
REM Run the enhanced web application for Credit Card Fraud Detection

REM Create necessary directories
mkdir data 2>nul
mkdir models\plots 2>nul
mkdir logs 2>nul
mkdir assets 2>nul

REM Check if virtual environment exists
if exist venv (
    echo Activating virtual environment...
    call venv\Scripts\activate
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Check if the dataset exists
if not exist data\creditcard.csv (
    echo Dataset not found. Please download it first.
    echo See instructions in data\README.md
    echo.
    echo You can still run the application, but some features may not work.
)

REM Check if the model exists
if not exist models\ensemble_model.joblib (
    if not exist models\best_model.joblib (
        echo No trained model found. You may want to run the training pipeline first:
        echo python run_pipeline.py --step all
        echo.
        echo You can still run the application, but prediction functionality will be limited.
    )
)

REM Run the web application
echo Starting the web application...
python webapp.py %* 