# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Copy pyproject.toml and install Python dependencies with uv
COPY pyproject.toml .
RUN uv sync

# Copy the source code
COPY src/ ./src/
COPY streamlit_app.py .

# Create necessary directories
RUN mkdir -p vector_db

# Activate the virtual environment by updating PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose ports
EXPOSE 8000 8501

# Default command runs the FastAPI server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] 