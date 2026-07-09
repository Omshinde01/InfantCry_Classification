FROM python:3.10-slim

# Set optimization and path environments
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/appuser \
    TFHUB_CACHE_DIR=/home/appuser/.cache/tfhub_modules

# Install essential system audio utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create a secure non-root user (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 appuser
WORKDIR $HOME/app

# Copy and install dependencies first to maximize Docker layer caching
COPY --chown=appuser requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Switch context to our secure user
USER appuser

# Cache YAMNet directly into the image layer during build time
RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1')"

# Copy the remaining application scripts and weights
COPY --chown=appuser . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]