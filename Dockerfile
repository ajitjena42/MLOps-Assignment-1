# Multi-stage / slim production Dockerfile for Heart Disease Serving API
FROM python:3.11-slim as runtime

# Set environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install minimal OS dependencies & curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user for enterprise security
RUN groupadd -r mlopsgroup && useradd -r -g mlopsgroup -d /app -s /sbin/nologin mlopsuser

WORKDIR /app

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, trained model artifacts, and minimal necessary assets
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Ensure non-root ownership
RUN chown -R mlopsuser:mlopsgroup /app

# Switch to non-root execution context
USER mlopsuser

# Expose HTTP serving port
EXPOSE 8000

# Container liveness check
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server entrypoint
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
