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
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/models /app/features /app/metrics

# Copy requirements
COPY docker/requirements/base.txt /app/requirements/base.txt
COPY docker/requirements/prediction.txt /app/requirements/prediction.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements/prediction.txt

# Copy source code
COPY src /app/src

# Create non-root user
RUN useradd -m -u 1000 app && \
    chown -R app:app /app

USER app

WORKDIR /app

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "src.prediction.app:app", "--host", "0.0.0.0", "--port", "8000"]
