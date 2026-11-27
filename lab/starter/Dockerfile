# syntax=docker/dockerfile:1
# Dockerfile API harga rumah — pola multi-stage yang SAMA dengan Day 46.
# Bedanya: TIDAK ada folder model/ di dalam image. Model diambil dari MLflow Registry saat start,
# jadi ganti model (promote / rollback) tidak perlu build ulang image.

# ---------- Stage 1: builder — install dependency ke virtualenv ----------
FROM python:3.11-slim AS builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
# install -> buang pip/setuptools -> buang folder tests/ bawaan library (lebih kecil)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y pip setuptools \
    && find /opt/venv -type d -name tests -prune -exec rm -rf {} +

# ---------- Stage 2: runtime — hanya yang perlu untuk serving ----------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MLFLOW_DISABLE_TELEMETRY=true \
    DO_NOT_TRACK=true \
    GIT_PYTHON_REFRESH=quiet
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# yang jarang berubah dulu, yang sering berubah (app/) paling akhir -> cache build awet
COPY config.yaml make_dataset.py ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app/ ./app/
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD python -c \
    "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
