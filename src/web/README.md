# Credit Card Fraud Detection Web Application

This directory contains the web application for the Credit Card Fraud Detection system. The web application provides a user-friendly interface for interacting with the fraud detection model, visualizing results, and analyzing fraud patterns.

## Features

- **Interactive Dashboard**: View model performance metrics and visualizations
- **Real-time Fraud Detection**: Test the model with transaction data
- **Analytics**: Visualize fraud patterns and trends
- **Prediction History**: Track and review previous predictions
- **Responsive Design**: Works on desktop and mobile devices

## Directory Structure

```
web/
├── app.py                # Main Dash application
├── assets/               # Static assets
│   ├── custom.css        # Custom CSS styles
│   ├── custom.js         # Custom JavaScript
│   ├── favicon.ico       # Favicon
│   └── ...               # Other assets (images, etc.)
└── templates/            # HTML templates
    └── index.html        # Main HTML template
```

## Running the Web Application

The web application can be run directly using the main `webapp.py` script in the project root:

```bash
python webapp.py
```

Or you can run it directly:

```bash
python src/web/app.py
```

By default, the application will run on `http://localhost:8050`.

## Configuration

The web application uses the configuration settings from the main `config.py` file. You can modify the following settings:

- `FLASK_DEBUG`: Set to `True` to enable debug mode
- `FLASK_PORT`: Port to run the application on (default: 8050)
- `FLASK_HOST`: Host to run the application on (default: "0.0.0.0")

## Customization

### Styling

The web application uses Bootstrap 5 for styling, with custom CSS in `assets/custom.css`. You can modify this file to change the appearance of the application.

### JavaScript

Custom JavaScript functionality is defined in `assets/custom.js`. You can modify this file to add or change interactive features.

### Templates

The main HTML template is defined in `templates/index.html`. You can modify this file to change the structure of the application.

## Integration with the Model

The web application integrates with the fraud detection model through the `predict_transaction` function in `src/models/predict_model.py`. This function takes a transaction as input and returns a prediction result.

## Performance Considerations

- The web application is designed to be lightweight and responsive
- Static assets are cached for improved performance
- The application uses asynchronous callbacks to prevent blocking
- Large datasets are processed in chunks to minimize memory usage

## Browser Compatibility

The web application is compatible with modern browsers:

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Security Considerations

- The web application does not store sensitive transaction data
- Input validation is performed to prevent injection attacks
- CSRF protection is enabled
- Debug mode should be disabled in production 