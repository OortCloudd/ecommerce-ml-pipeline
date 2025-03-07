FROM apache/airflow:2.7.2

USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python packages
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
