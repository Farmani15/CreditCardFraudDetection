# Deployment Guide

This guide provides instructions for deploying the Credit Card Fraud Detection System in various environments.

## Local Deployment

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd credit-card-fraud-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset (see instructions in data/README.md)

5. Run the pipeline to preprocess data and train the model:
   ```bash
   python run_pipeline.py --step all
   ```

6. Start the web application:
   ```bash
   python webapp.py
   ```

7. Access the application at http://localhost:8050

## Docker Deployment

### Prerequisites

- Docker
- Docker Compose (optional, for multi-container deployment)

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t credit-card-fraud-detection .
   ```

2. Run the container:
   ```bash
   docker run -p 8050:8050 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models credit-card-fraud-detection
   ```

3. Access the application at http://localhost:8050

### Using Docker Compose

1. Start the services:
   ```bash
   docker-compose up -d
   ```

2. Access the application at http://localhost:8050

3. Stop the services:
   ```bash
   docker-compose down
   ```

## Cloud Deployment

### AWS Elastic Beanstalk

1. Install the EB CLI:
   ```bash
   pip install awsebcli
   ```

2. Initialize EB application:
   ```bash
   eb init -p python-3.8 credit-card-fraud-detection
   ```

3. Create an environment:
   ```bash
   eb create credit-card-fraud-detection-env
   ```

4. Deploy the application:
   ```bash
   eb deploy
   ```

5. Open the application:
   ```bash
   eb open
   ```

### Heroku

1. Install the Heroku CLI and login:
   ```bash
   heroku login
   ```

2. Create a Heroku app:
   ```bash
   heroku create credit-card-fraud-detection
   ```

3. Add Redis add-on (optional):
   ```bash
   heroku addons:create heroku-redis:hobby-dev
   ```

4. Deploy the application:
   ```bash
   git push heroku main
   ```

5. Open the application:
   ```bash
   heroku open
   ```

## Production Considerations

### Security

- Set `FLASK_DEBUG=False` in production
- Use HTTPS for all communications
- Implement proper authentication if needed
- Regularly update dependencies

### Performance

- Use a production WSGI server like Gunicorn
- Configure appropriate worker processes
- Use Redis for caching
- Consider using a CDN for static assets

### Monitoring

- Set up logging to a centralized service
- Monitor application performance
- Set up alerts for errors and performance issues

### Scaling

- Use a load balancer for horizontal scaling
- Consider containerization for easier scaling
- Use a database for storing prediction history instead of in-memory storage

## Troubleshooting

### Common Issues

1. **Application fails to start**
   - Check if all dependencies are installed
   - Verify that the correct Python version is being used
   - Check the logs for specific error messages

2. **Model not found**
   - Run the pipeline to train the model
   - Check if the model file paths are correct

3. **Dataset not found**
   - Download the dataset according to instructions in data/README.md
   - Check if the dataset file paths are correct

4. **Out of memory errors**
   - Reduce the chunk size in config.py
   - Increase the memory allocation for the container
   - Use a machine with more RAM

### Getting Help

If you encounter issues not covered in this guide, please:
1. Check the project's issue tracker
2. Create a new issue with detailed information about the problem
3. Contact the project maintainers 