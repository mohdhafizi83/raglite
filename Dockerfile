# raglite — offline hybrid RAG service
# Build:  docker build -t raglite .
# Run:    docker compose up -d   (see docker-compose.yml; index persists in a volume)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first for layer caching (pinned versions, pip-audited in CI).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code only — no tests, no dotfiles, no venvs (see .dockerignore).
COPY raglite/ ./raglite/

# Run as an unprivileged user; the index lives on a mounted volume at /data.
RUN useradd --system --uid 10001 raglite \
    && mkdir -p /data && chown raglite /data
USER raglite

ENV RAGLITE_INDEX_PATH=/data/index.db RAGLITE_UI=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "raglite.app:app", "--host", "0.0.0.0", "--port", "8000"]
