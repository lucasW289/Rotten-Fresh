FROM python:3.9-slim

WORKDIR /app

# 1. Install dependencies
COPY requirements.txt .
# FIX IS HERE: Added --default-timeout=1000 to prevent "Read timed out"
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Copy the folders structure
COPY configs/ ./configs/
COPY dataset/ ./dataset/
COPY src/ ./src/

# Create output dirs
RUN mkdir models logs

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# 3. Run using the config file
CMD ["python", "src/train.py", "--config", "configs/base_config.yaml"]