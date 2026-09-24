FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY ramp ./ramp
COPY config ./config
COPY dashboard ./dashboard
COPY scripts ./scripts

RUN pip install --no-cache-dir -e .

EXPOSE 8000 8501

CMD ["uvicorn", "ramp.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
