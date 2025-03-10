FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data models/plots logs assets

# Expose the port the app runs on
EXPOSE 8050

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "webapp:server"] 