FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir uv==0.12.19 \
    && apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 floor \
    && mkdir /data && chown floor:floor /data

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    FLOOR_STATE_DIR=/data \
    FLOOR_DESK_DB=/data/floor.sqlite3

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev
COPY app.py floor.py experiment.json ./
COPY scripts/ scripts/
COPY prompts/ prompts/
COPY static/ static/
# Only tokenizer/config files are fetched; model weights stay on Tinker.
RUN python -c 'from scripts.common import setup; from scripts.rendering import build_renderer; build_renderer(setup())' \
    && chown -R floor:floor /app/.cache

USER floor
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c 'import os, urllib.request; urllib.request.urlopen("http://127.0.0.1:" + os.environ.get("PORT", "8000") + "/health", timeout=3)' || exit 1
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
