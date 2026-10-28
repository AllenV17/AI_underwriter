# One image for both services: the backend (uvicorn) and the frontend (streamlit),
# selected by the command in docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

# Tesseract enables OCR of scanned pages; libgl/libglib are needed by Docling's layout models.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
# Separate layers so a network hiccup during the large downloads doesn't discard the torch layer.
RUN pip install --timeout 180 --retries 10 torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --timeout 180 --retries 10 -r /app/backend/requirements.txt

COPY backend /app/backend
COPY frontend /app/frontend

# Non-root user
RUN useradd -m appuser && mkdir -p /app/data /models && chown -R appuser /app /models
USER appuser

WORKDIR /app/backend
EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
