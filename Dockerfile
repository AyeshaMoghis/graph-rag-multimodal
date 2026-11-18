FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

# Tesseract for OCR; without it every image falls back to BLIP captions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first so the pinned version does not pull CUDA wheels.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

RUN useradd --create-home app && mkdir -p /models /app/output /app/data \
    && chown -R app /models /app
USER app

EXPOSE 8000
CMD ["uvicorn", "graphrag.api.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
