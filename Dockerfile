# Dockerfile for Hugging Face Spaces Training
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_hf.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_hf.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /data /output /models

# Set environment variables
ENV PYTHONPATH=/app
ENV HF_HOME=/cache/huggingface
ENV WANDB_CACHE_DIR=/cache/wandb

# Expose port for Gradio (if needed)
EXPOSE 7860

# Default command
CMD ["python", "scripts/hf_cloud_training.py"]







