# ── Build stage ──────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# System libs needed at runtime (libGL for OpenCV headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY app.py .
COPY config_api.yaml .
COPY src/ ./src/
COPY trained_models/ ./trained_models/

# Railway uses $PORT env variable
ENV PORT=8000
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/')" || exit 1

# Start server
CMD uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
