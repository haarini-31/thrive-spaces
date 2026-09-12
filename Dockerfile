FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (Hugging Face Spaces runs as user 1000)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache the Hugging Face emotion transformer model and TextBlob corpora
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k=1)" && \
    python -m textblob.download_corpora

# Copy application files with appropriate ownership
COPY --chown=user:user . /app

# Ensure SQLite instance folder is writable by non-root user
RUN mkdir -p /app/instance && \
    chown -R user:user /app /home/user && \
    chmod -R 775 /app/instance

USER user

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Production startup with Gunicorn (timeout 120s for safe startup)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
