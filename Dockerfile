# One image for both the API and the Streamlit UI (docker-compose.yml picks the command).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# libgl1 + libglib2.0-0: needed by OpenCV, which RapidOCR uses
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first: the default Linux wheel bundles ~2 GB of CUDA libraries we never use
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1

COPY requirements.txt .
RUN pip install -r requirements.txt

# bake the embedding model into the image so the first upload doesn't wait for a download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
COPY ui ./ui
COPY sql ./sql
COPY samples ./samples

# run as a normal user, not root
RUN useradd --create-home appuser && mkdir -p data/uploads && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
CMD ["sh", "-c", "python -m app.preload samples && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
