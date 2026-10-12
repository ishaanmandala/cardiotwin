# Dockerfile for a Hugging Face Space (https://huggingface.co/spaces).
# Copy this file into your Space; it downloads CardioTwin from GitHub while building.
# To pick up new code later: Space -> Settings -> "Factory rebuild".
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only PyTorch (~200 MB instead of ~3 GB of GPU libraries).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Which code to deploy (change BRANCH to "main" once the code is merged there).
ARG REPO=https://github.com/ishaanmandala/cardiotwin
ARG BRANCH=claude/cardiotwin-ecg-ef-model-urtqpe
ADD ${REPO}/archive/refs/heads/${BRANCH}.tar.gz /tmp/cardiotwin.tar.gz
RUN mkdir /app && tar -xzf /tmp/cardiotwin.tar.gz -C /app --strip-components=1 && rm /tmp/cardiotwin.tar.gz \
    && pip install /app

# Hugging Face runs Spaces as user 1000.
RUN useradd --create-home --uid 1000 user && chown -R user /app
USER user
WORKDIR /app

ENV HOST=0.0.0.0 \
    PORT=7860 \
    CARDIOTWIN_MODELS=/app/models
EXPOSE 7860
CMD ["cardiotwin", "serve", "--no-browser"]
