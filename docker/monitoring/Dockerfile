# Use Prometheus base image
FROM prom/prometheus:v2.50.1

# Copy Prometheus configuration
COPY docker/monitoring/prometheus.yml /etc/prometheus/prometheus.yml

# Create and set permissions for Prometheus data directory
USER root
RUN mkdir -p /prometheus && \
    chown -R nobody:nobody /prometheus

# Set user for security
USER nobody

# Expose Prometheus port
EXPOSE 9090

# Default command
CMD [ "--config.file=/etc/prometheus/prometheus.yml", \
      "--storage.tsdb.path=/prometheus", \
      "--web.console.libraries=/usr/share/prometheus/console_libraries", \
      "--web.console.templates=/usr/share/prometheus/consoles" ]
