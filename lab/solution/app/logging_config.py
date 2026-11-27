"""app/logging_config.py — log terstruktur: SATU baris JSON per kejadian (bisa dibaca mesin)."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))  # isi dari extra={"fields": {...}}
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(level: str | None = None) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level or os.getenv("LOG_LEVEL", "INFO"))
    # log uvicorn & mlflow ikut format JSON yang sama (bukan teks bebas)
    for name in ("uvicorn", "uvicorn.error", "mlflow"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True
    # access log bawaan uvicorn dimatikan: request sudah dicatat sendiri (1 baris JSON per /predict)
    access = logging.getLogger("uvicorn.access")
    access.handlers[:] = []
    access.propagate = False
    logging.captureWarnings(True)  # warnings Python -> log JSON juga


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
