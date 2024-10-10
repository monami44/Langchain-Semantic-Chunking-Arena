# Dockerfile

# Use official Python image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project
COPY . .

# Create logs and results directories
RUN mkdir -p logs results chunk_size_evaluator results/retrieval_quality_metrics results/scores results/chunk_sizes

# Set entrypoint
ENTRYPOINT ["python", "src/main.py"]
