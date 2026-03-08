# Brain-Inspired AI Docker Image
# 类脑人工智能Docker镜像

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package
RUN pip install -e .

# Create directories for weights and data
RUN mkdir -p /app/weights /app/data /app/logs

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["python", "main.py", "server", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base as dev

RUN pip install pytest pytest-asyncio black flake8 mypy

CMD ["bash"]

# Production stage
FROM base as prod

# Download weights during build (optional)
# RUN python scripts/download_weights.py --cache-dir /app/weights

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["python", "main.py", "server", "--host", "0.0.0.0", "--port", "8000"]
