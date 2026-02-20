# Single-stage build — avoids the expensive multi-stage "importing" step
# that causes Railway free tier to timeout at 10 minutes
FROM python:3.10-slim

WORKDIR /app

# System deps (build + runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only FIRST (smallest version, ~185MB)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove build-only packages to save space
RUN apt-get purge -y gcc g++ && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY app.py config_api.yaml ./
COPY src/ ./src/
COPY trained_models/ ./trained_models/

# Railway uses $PORT env variable
ENV PORT=8000
EXPOSE $PORT

CMD ["python", "app.py"]
