FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY configs/ ./configs/
COPY src/ ./src/

RUN mkdir models logs

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

CMD ["python", "src/train.py", "--config", "configs/base_config.yaml"]