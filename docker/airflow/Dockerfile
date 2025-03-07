# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /opt/airflow

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements/airflow.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create airflow user
RUN groupadd -g 50000 airflow && \
    useradd -u 50000 -g airflow airflow && \
    mkdir -p /home/airflow && \
    chown -R airflow:airflow /home/airflow && \
    chown -R airflow:airflow /opt/airflow

# Copy source code and configuration
COPY airflow/dags dags/
COPY airflow/plugins plugins/
COPY airflow/config config/
COPY src src/
COPY airflow/airflow.env .
COPY airflow/scripts/init.sh /init.sh

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow \
    PYTHONPATH=/opt/airflow

USER airflow

# Default command (will be overridden by docker-compose)
CMD ["airflow", "webserver"]
