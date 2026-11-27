"""Paket pipeline ML harga rumah (Day 47)."""
import os

# Matikan telemetri anonim MLflow (data penggunaan tidak dikirim ke luar).
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")
# Laptop/container tanpa git: jangan banjiri layar dengan warning GitPython.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
