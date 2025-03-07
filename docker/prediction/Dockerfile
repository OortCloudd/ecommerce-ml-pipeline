# Use Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models \
    FEATURE_PATH=/app/features \
    METRICS_PATH=/app/metrics

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/models /app/features /app/metrics /data /models /features /cache

# Copy requirements files
COPY docker/requirements/base.txt /tmp/base.txt
COPY docker/requirements/prediction.txt /tmp/prediction.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/base.txt && \
    pip install --no-cache-dir -r /tmp/prediction.txt

# Copy source code
COPY src /app/src

# Create non-root user
RUN useradd -m -u 1000 app && \
    chown -R app:app /app /data /models /features /cache

USER app

WORKDIR /app

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "src.prediction.app:app", "--host", "0.0.0.0", "--port", "8000"]
