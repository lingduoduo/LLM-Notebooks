FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN python -m pip install --no-cache-dir ".[documents,media,data]"
RUN useradd --create-home --uid 1000 mcpuser \
    && chown -R mcpuser:mcpuser /app

USER mcpuser
ENTRYPOINT ["perception-tools-mcp"]
