# CardioTwin web app in a container.
#
#   docker build -t cardiotwin .
#   docker run -p 8000:8000 cardiotwin        # then open http://localhost:8000
#
# Works on any host that runs Docker images (Render, Railway, Fly.io, Google Cloud Run, a VPS...).
# The host's PORT environment variable is respected.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only PyTorch: ~200 MB instead of ~3 GB of GPU libraries. Override the index if it is blocked.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url ${TORCH_INDEX_URL}

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Trained models + demo ECGs (MUSIC, ODbL) and the example uploads (PTB-XL, CC BY 4.0).
# Never add MIMIC-derived data or models to a public image.
COPY models ./models
COPY examples ./examples

RUN useradd --create-home --uid 1000 cardiotwin
USER cardiotwin

ENV HOST=0.0.0.0 \
    PORT=8000 \
    CARDIOTWIN_MODELS=/app/models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ['PORT'])" || exit 1
CMD ["cardiotwin", "serve", "--no-browser"]
