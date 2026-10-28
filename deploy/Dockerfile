# Multi-arch production Dockerfile for M6 Ultrasound AI & BreastVision Platform
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8008

WORKDIR /app

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Model Development codebase, source files, and inference app
COPY MODEL_DEVELOPMENT/src /app/MODEL_DEVELOPMENT/src
COPY MODEL_DEVELOPMENT/checkpoints/M6_FINAL /app/MODEL_DEVELOPMENT/checkpoints/M6_FINAL
COPY MODEL_DEVELOPMENT/step8_ablation_predictions.csv /app/MODEL_DEVELOPMENT/step8_ablation_predictions.csv
COPY MODEL_DEVELOPMENT/app /app/MODEL_DEVELOPMENT/app

# Copy BreastVision React Native / Expo exported web bundle
COPY breastvision/dist /app/breastvision/dist

# Working directory set to the backend app
WORKDIR /app/MODEL_DEVELOPMENT/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8008

# Start FastAPI Uvicorn server using dynamic PORT environment variable
CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port ${PORT:-8008}"]
