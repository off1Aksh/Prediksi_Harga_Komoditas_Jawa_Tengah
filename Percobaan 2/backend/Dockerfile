FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn openpyxl

# Copy application code and models
COPY app.py .
COPY .env .
COPY models_univariate/ models_univariate/
COPY models_vision/ models_vision/
COPY Dataset_2017_2026/dataset_harga_komoditas_2017_2026.csv Dataset_2017_2026/dataset_harga_komoditas_2017_2026.csv

# Expose port
EXPOSE 5000

# Run with Gunicorn (production server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "--threads", "2", "app:app"]
