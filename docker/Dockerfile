# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only what setup needs (cache-friendly)
COPY requirements.txt .
     setup.sh .

# Run setup
RUN chmod +x setup.sh && ./setup.sh

# Copy application code
COPY . .

# Create necessary directories
# RUN mkdir -p data/raw data/processed models logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD sh -c "uvicorn api.main:app --host 0.0.0.0 --port $PORT"


